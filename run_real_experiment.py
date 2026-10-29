"""Прогон реальных LLM по тестовому окну (E3/E4/E5).

Особенности:
- 60 последних торговых дней на тикер (по умолчанию), 6 тикеров;
- два режима retrieval: standard и regime-aware;
- ответы моделей сохраняются в llm_responses.jsonl и переиспользуются
  при повторном запуске (резюм), поэтому прогон можно прерывать;
- ансамбли majority/confidence-weighted считаются из одних и тех же
  ответов, без дополнительных вызовов API;
- circuit breaker: модель с серией ошибок отключается до конца прогона,
  чтобы не жечь квоту на 429/401.

Ключи берутся только из переменных окружения:
OPENROUTER_API_KEY, GEMINI_API_KEY.
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ensemble import (
    CONFIDENCE_WEIGHTED_VOTING,
    MAJORITY_VOTING,
    make_ensemble_decision,
)
from llm_predictor import (
    AnymodelProvider,
    GeminiProvider,
    OpenRouterProvider,
    predict_for_day,
    safe_name,
)
from pipeline import FEATURES_FOR_LLM, REGIME_LABEL_COLUMN, evaluate_predictions
from retrieval import find_similar_days

DEFAULT_MODELS = [
    "gemini:gemini-3.1-flash-lite",
    "gemini:gemini-3.5-flash-lite",
    "gemini:gemma-4-31b-it",
    "openrouter:liquid/lfm-2.5-2.6b:free",
    "openrouter:nvidia/nemotron-3.5-lightning:free",
]


class RateLimiter:
    """Простейший per-model лимитер запросов в минуту (free-tier)."""

    def __init__(self, rpm):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self.next_slot = {}
        self.lock = threading.Lock()

    def wait(self, key):
        if self.interval <= 0:
            return
        with self.lock:
            now = time.time()
            slot = max(now, self.next_slot.get(key, now))
            self.next_slot[key] = slot + self.interval
        delay = slot - time.time()
        if delay > 0:
            time.sleep(delay)

AUTH_ERROR_MARKERS = (
    "401",
    "invalid auth key",
    "api key not valid",
    "unauthenticated",
    "permission denied",
    "403",
)

_write_lock = threading.Lock()


def build_provider(name):
    kind, model = name.split(":", 1)
    if kind == "gemini":
        return GeminiProvider(model_name=model)
    if kind == "openrouter":
        return OpenRouterProvider(model_name=model)
    if kind == "anymodel":
        if model.startswith(("am/nemotron", "am/gpt-oss")):
            max_tokens = 4096
        else:
            max_tokens = 512
        return AnymodelProvider(model_name=model, max_tokens=max_tokens)
    raise ValueError(f"Неизвестный провайдер: {name!r}")


def preflight(provider):
    """Проверяет, что модель вообще отвечает. Возвращает (ok, reason)."""
    prompt = (
        'Answer only with JSON: {"prediction":"HOLD","confidence":0.5,'
        '"probability_up":0.5,"explanation":"ping"}'
    )
    try:
        raw = provider.complete(prompt, temperature=0.0, max_retries=0)
    except Exception as exc:
        text = str(exc)
        if any(m in text.lower() for m in AUTH_ERROR_MARKERS):
            return False, text[:200]
        return True, f"kept despite: {text[:120]}"
    return True, raw[:80]


def payload_for(row, data, retrieval_mode, top_k):
    ticker = str(row["Ticker"])
    date = pd.Timestamp(row["Date"]).strftime("%Y-%m-%d")
    regime = row.get(REGIME_LABEL_COLUMN)
    regime = None if pd.isna(regime) else str(regime)

    retrieval = find_similar_days(
        current_day=pd.Timestamp(date),
        ticker=ticker,
        top_k=top_k,
        data=data,
        regime_aware=(retrieval_mode == "regime-aware"),
    )
    similar_days = [
        {
            "date": item["date"],
            "similarity": item["similarity"],
            "regime": item.get("regime"),
        }
        for item in retrieval["similar_days"]
    ]
    features = {}
    for source_col, payload_name in FEATURES_FOR_LLM.items():
        if source_col in row.index and pd.notna(row[source_col]):
            features[payload_name] = float(row[source_col])

    return {
        "ticker": ticker,
        "date": date,
        "features": features,
        "market_regime": regime,
        "similar_days": similar_days,
    }


def load_store(path):
    done = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("status") == "ok":
                done[(rec["date"], rec["ticker"], rec["retrieval_mode"], rec["model"])] = rec
    return done


class Runner:
    def __init__(self, store_path, cache_dir, log_dir, max_fail_streak,
                 limiter=None, max_retries=1):
        self.store_path = store_path
        self.cache_dir = Path(cache_dir)
        self.log_dir = Path(log_dir)
        self.max_fail_streak = max_fail_streak
        self.limiter = limiter
        self.max_retries = max_retries
        self.lock = threading.Lock()
        self.fail_streak = {}
        self.disabled = {}
        self.stats = {"ok": 0, "error": 0, "cached": 0}

    def is_disabled(self, model):
        with self.lock:
            return model in self.disabled

    def record(self, rec):
        with _write_lock:
            with open(self.store_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def run_one(self, provider, payload, retrieval_mode, use_cache=True):
        key = provider.name()
        if self.is_disabled(key):
            return None
        try:
            if self.limiter is not None:
                self.limiter.wait(key)
            response = predict_for_day(
                payload,
                provider,
                retrieval_mode=retrieval_mode,
                cache_dir=self.cache_dir / safe_name(key),
                log_dir=self.log_dir,
                use_cache=use_cache,
                max_retries=self.max_retries,
            )
        except Exception as exc:
            with self.lock:
                self.fail_streak[key] = self.fail_streak.get(key, 0) + 1
                self.stats["error"] += 1
                if self.fail_streak[key] >= self.max_fail_streak:
                    self.disabled[key] = str(exc)[:160]
            self.record({
                "date": payload["date"],
                "ticker": payload["ticker"],
                "retrieval_mode": retrieval_mode,
                "model": key,
                "status": "error",
                "error": str(exc)[:300],
            })
            return None

        with self.lock:
            self.fail_streak[key] = 0
            self.stats["ok"] += 1
        rec = {
            "date": payload["date"],
            "ticker": payload["ticker"],
            "retrieval_mode": retrieval_mode,
            "model": key,
            "status": "ok",
            "response": response,
            "payload": payload,
        }
        self.record(rec)
        return rec


def build_predictions(jobs, store, ensemble_method, retrieval_mode):
    rows = []
    for job in jobs:
        if job["retrieval_mode"] != retrieval_mode:
            continue
        p = job["payload"]
        responses = []
        failed = []
        for model in job["models"]:
            rec = store.get((p["date"], p["ticker"], retrieval_mode, model))
            if rec is None:
                failed.append({"model": model, "error": "missing"})
                continue
            r = dict(rec["response"])
            r["model"] = r.get("model") or model
            responses.append(r)
        if not responses:
            continue
        decision = make_ensemble_decision(
            responses, method=ensemble_method, regime=p["market_regime"],
        )
        rows.append({
            "Date": p["date"],
            "Ticker": p["ticker"],
            "Prediction": decision["prediction"],
            "Probability": round(float(pd.Series(
                [r.get("probability_up", 0.5) for r in responses]
            ).mean()), 4),
            "Score": round(float(decision["score"]), 4),
            "Method": decision["method"],
            "Regime": p["market_regime"],
            "RetrievalMode": retrieval_mode,
            "NAnalogues": len(p["similar_days"]),
            "FallbackUsed": False,
            "ModelsFailed": json.dumps(failed, ensure_ascii=False),
            "ModelPredictions": json.dumps(
                [{"model": r["model"], "prediction": r["prediction"],
                  "confidence": r["confidence"]} for r in responses],
                ensure_ascii=False,
            ),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/market_data_with_regime.csv")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--max-fail-streak", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=12,
                    help="Лимит запросов в минуту на модель (0 = без лимита)")
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--out-dir", default="results_real")
    ap.add_argument("--cache-dir", default="llm_cache_real")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    store_path = out_dir / "llm_responses.jsonl"

    data = pd.read_csv(args.data)
    data["Date"] = pd.to_datetime(data["Date"])
    tickers = args.tickers or sorted(data["Ticker"].astype(str).unique())

    jobs = []
    for ticker in tickers:
        td = data[data["Ticker"].astype(str) == ticker].sort_values("Date")
        has_next = td["Close"].shift(-1).notna()
        dates = td.loc[has_next, "Date"].sort_values().unique()
        window = [pd.Timestamp(d) for d in dates[-args.days:]]
        for date in window:
            row = td[td["Date"] == date].iloc[-1]
            for mode in ("standard", "regime-aware"):
                payload = payload_for(row, data, mode, args.top_k)
                jobs.append({
                    "payload": payload,
                    "retrieval_mode": mode,
                    "models": args.models,
                })
    print(f"Задач (день x режим): {len(jobs)}; моделей: {len(args.models)}", flush=True)

    store = load_store(store_path)
    print(f"Уже в кэше успешных ответов: {len(store)}", flush=True)

    providers = []
    for name in args.models:
        provider = build_provider(name)
        if not args.skip_preflight:
            ok, reason = preflight(provider)
            status = "OK " if ok else "DROP"
            print(f"[preflight] {status} {provider.name()}: {reason}", flush=True)
            if not ok:
                continue
        providers.append(provider)
    if not providers:
        raise SystemExit("Ни одна модель не прошла preflight")

    runner = Runner(store_path, cache_dir, out_dir / "llm", args.max_fail_streak,
                    limiter=RateLimiter(args.rpm), max_retries=args.max_retries)

    tasks = []
    for job in jobs:
        for provider in providers:
            key = (job["payload"]["date"], job["payload"]["ticker"],
                   job["retrieval_mode"], provider.name())
            if key in store:
                continue
            tasks.append((provider, job["payload"], job["retrieval_mode"]))
    print(f"Нужно API-вызовов (без кэша): {len(tasks)}", flush=True)

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(runner.run_one, p, pl, m, not args.no_cache): (p.name(), pl["date"], pl["ticker"], m)
            for p, pl, m in tasks
        }
        done = 0
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 25 == 0 or done == len(tasks):
                elapsed = time.time() - started
                print(
                    f"progress {done}/{len(tasks)} ok={runner.stats['ok']} "
                    f"err={runner.stats['error']} elapsed={elapsed:.0f}s "
                    f"disabled={list(runner.disabled)}",
                    flush=True,
                )

    for model, reason in runner.disabled.items():
        print(f"[circuit-breaker] отключена {model}: {reason}", flush=True)

    store = load_store(store_path)
    providers_used = [p.name() for p in providers]

    metadata = {
        "data": args.data,
        "days": args.days,
        "tickers": tickers,
        "top_k": args.top_k,
        "models_requested": args.models,
        "models_used": providers_used,
        "models_disabled": runner.disabled,
        "n_successful_responses": len(store),
        "elapsed_sec": round(time.time() - started, 1),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    configs = [
        ("standard", MAJORITY_VOTING),
        ("standard", CONFIDENCE_WEIGHTED_VOTING),
        ("regime-aware", MAJORITY_VOTING),
        ("regime-aware", CONFIDENCE_WEIGHTED_VOTING),
    ]
    summary = []
    for mode, method in configs:
        preds = build_predictions(jobs, store, method, mode)
        tag = f"{mode}_{method}"
        pred_path = out_dir / f"predictions_{tag}.csv"
        preds.to_csv(pred_path, index=False)
        if preds.empty:
            print(f"{tag}: нет прогнозов", flush=True)
            continue
        res_dir = out_dir / f"eval_{tag}"
        cls, trd = evaluate_predictions(
            predictions_path=str(pred_path),
            data_path=args.data,
            output_alignment=str(res_dir),
        )
        acc = cls["accuracy"].mean() if "accuracy" in cls else None
        mcc = cls["mcc"].mean() if "mcc" in cls else None
        summary.append({
            "config": tag,
            "n_predictions": len(preds),
            "mean_accuracy": None if acc is None else round(float(acc), 4),
            "mean_mcc": None if mcc is None else round(float(mcc), 4),
        })
        print(f"\n=== {tag} (n={len(preds)}) ===")
        print(cls.to_string(index=False))
        print(trd.to_string(index=False))

    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    print("\nИтог сохранён в", out_dir, flush=True)


if __name__ == "__main__":
    main()
