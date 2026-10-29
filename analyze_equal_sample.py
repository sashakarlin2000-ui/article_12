"""Анализ равного набора ответов LLM: только кейсы, где ответили все модели."""

import argparse
import json
from pathlib import Path

import pandas as pd

from ensemble import (
    CONFIDENCE_WEIGHTED_VOTING,
    MAJORITY_VOTING,
    make_ensemble_decision,
)
from pipeline import evaluate_predictions

DEFAULT_MODELS = [
    "gemini:gemini-3.1-flash-lite",
    "gemini:gemini-3.5-flash-lite",
    "gemini:gemma-4-31b-it",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="results_real/llm_responses.jsonl")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--data", default="data/processed/market_data_with_regime.csv")
    ap.add_argument("--out-dir", default="results_equal")
    ap.add_argument("--min-days", type=int, default=10,
                    help="Оставлять только тикеры с >= N дней в равном наборе")
    args = ap.parse_args()

    records = []
    with open(args.store, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("status") == "ok" and rec.get("model") in args.models:
                records.append(rec)

    by_model = {m: set() for m in args.models}
    indexed = {}
    for rec in records:
        key = (rec["date"], rec["ticker"], rec["retrieval_mode"])
        by_model[rec["model"]].add(key)
        indexed[(key, rec["model"])] = rec

    counts = {m: len(by_model[m]) for m in args.models}
    common = set.intersection(*by_model.values()) if by_model else set()
    print("Ответов по моделям:", counts)
    print("Общих кейсов (все модели):", len(common))
    if not common:
        raise SystemExit("Нет общих кейсов")

    rows = []
    for (date, ticker, mode) in sorted(common):
        responses = []
        payload = None
        for model in args.models:
            rec = indexed[((date, ticker, mode), model)]
            payload = rec["payload"]
            r = dict(rec["response"])
            r["model"] = r.get("model") or model
            responses.append(r)
        rows.append({
            "Date": date, "Ticker": ticker, "RetrievalMode": mode,
            "Regime": payload.get("market_regime"),
            "Responses": responses,
            "Payload": payload,
        })

    ticker_days = {}
    for r in rows:
        if r["RetrievalMode"] == "standard":
            ticker_days[r["Ticker"]] = ticker_days.get(r["Ticker"], 0) + 1
    keep = {t for t, n in ticker_days.items() if n >= args.min_days}
    dropped = {t: n for t, n in ticker_days.items() if n < args.min_days}
    if dropped:
        print("Отброшены тикеры (мало дней):", dropped)
    rows = [r for r in rows if r["Ticker"] in keep]
    print("Кейсов после фильтра:", len(rows), "тикеры:", sorted(keep))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    def eval_frame(frame, tag):
        pred_path = out_dir / f"predictions_{tag}.csv"
        frame.to_csv(pred_path, index=False)
        cls, trd = evaluate_predictions(
            predictions_path=str(pred_path),
            data_path=args.data,
            output_alignment=str(out_dir / f"eval_{tag}"),
        )
        summary.append({
            "config": tag,
            "n_predictions": len(frame),
            "mean_accuracy": round(float(cls["accuracy"].mean()), 4),
            "mean_balanced_accuracy": round(float(cls["balanced_accuracy"].mean()), 4),
            "mean_f1": round(float(cls["f1"].mean()), 4),
            "mean_mcc": round(float(cls["mcc"].mean()), 4),
            "mean_cumulative_return": round(float(trd["cumulative_return"].mean()), 4),
            "mean_sharpe": round(float(trd["sharpe_ratio"].mean()), 4),
        })
        print(f"\n=== {tag} (n={len(frame)}) ===")
        print(cls.to_string(index=False))
        print(trd.to_string(index=False))

    for model in args.models:
        for mode in ("standard", "regime-aware"):
            preds = []
            for r in rows:
                if r["RetrievalMode"] != mode:
                    continue
                resp = next(x for x in r["Responses"] if x["model"] == model)
                preds.append({
                    "Date": r["Date"],
                    "Ticker": r["Ticker"],
                    "Prediction": resp["prediction"],
                    "Probability": resp.get("probability_up", 0.5),
                    "Score": 0.0,
                    "Method": f"single:{model}",
                    "Regime": r["Regime"],
                    "RetrievalMode": r["RetrievalMode"],
                    "NAnalogues": len((r["Payload"] or {}).get("similar_days", [])),
                    "FallbackUsed": False,
                    "ModelsFailed": "[]",
                    "ModelPredictions": json.dumps([{
                        "model": resp["model"],
                        "prediction": resp["prediction"],
                        "confidence": resp["confidence"],
                    }], ensure_ascii=False),
                })
            eval_frame(
                pd.DataFrame(preds),
                f"single_{model.split(':', 1)[1]}_{mode}",
            )

    for mode in ("standard", "regime-aware"):
        subset = [r for r in rows if r["RetrievalMode"] == mode]
        for method in (MAJORITY_VOTING, CONFIDENCE_WEIGHTED_VOTING):
            preds = []
            for r in subset:
                decision = make_ensemble_decision(
                    r["Responses"], method=method, regime=r["Regime"],
                )
                preds.append({
                    "Date": r["Date"],
                    "Ticker": r["Ticker"],
                    "Prediction": decision["prediction"],
                    "Probability": round(float(pd.Series(
                        [x.get("probability_up", 0.5) for x in r["Responses"]]
                    ).mean()), 4),
                    "Score": round(float(decision["score"]), 4),
                    "Method": decision["method"],
                    "Regime": r["Regime"],
                    "RetrievalMode": mode,
                    "NAnalogues": len((r["Payload"] or {}).get("similar_days", [])),
                    "FallbackUsed": False,
                    "ModelsFailed": "[]",
                    "ModelPredictions": json.dumps(
                        [{"model": x["model"], "prediction": x["prediction"],
                          "confidence": x["confidence"]} for x in r["Responses"]],
                        ensure_ascii=False,
                    ),
                })
            eval_frame(pd.DataFrame(preds), f"{mode}_{method}")

    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    print("\n=== SUMMARY ===")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nСохранено в", out_dir)


if __name__ == "__main__":
    main()
