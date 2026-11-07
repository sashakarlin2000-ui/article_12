"""Evaluate validation-selected model ensembles without test-set selection."""

import argparse
import json
from pathlib import Path

import pandas as pd

from ensemble import CONFIDENCE_WEIGHTED_VOTING, REGIME_WEIGHTED_VOTING, make_ensemble_decision
from pipeline import evaluate_predictions


def load_store(path):
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            if rec.get("status") == "ok":
                key = (rec["date"], rec["ticker"], rec["retrieval_mode"], rec["model"])
                records[key] = rec
    return records


def response_rows(store, models, mode="no-retrieval"):
    grouped = {}
    for (date, ticker, retrieval_mode, model), rec in store.items():
        if retrieval_mode == mode and model in models:
            grouped.setdefault((date, ticker), {})[model] = rec
    rows = []
    for (date, ticker), by_model in sorted(grouped.items()):
        if set(by_model) != set(models):
            continue
        payload = next(iter(by_model.values()))["payload"]
        rows.append((date, ticker, payload, by_model))
    return rows


def frame(rows, models, method, weights=None):
    output = []
    for date, ticker, payload, by_model in rows:
        responses = []
        for model in models:
            response = dict(by_model[model]["response"])
            response["model"] = model
            responses.append(response)
        decision = make_ensemble_decision(
            responses, method=method, weights=weights,
            regime=payload.get("market_regime"),
        )
        output.append({
            "Date": date,
            "Ticker": ticker,
            "Prediction": decision["prediction"],
            "Probability": sum(r.get("probability_up", 0.5) for r in responses) / len(responses),
            "Score": decision["score"],
            "Method": method,
            "Regime": payload.get("market_regime"),
            "RetrievalMode": "no-retrieval",
            "NAnalogues": 0,
            "FallbackUsed": False,
            "ModelsFailed": "[]",
            "ModelPredictions": json.dumps(
                [{"model": r["model"], "prediction": r["prediction"],
                  "confidence": r["confidence"]} for r in responses],
                ensure_ascii=False,
            ),
        })
    return pd.DataFrame(output)


def evaluate(frame_data, data_path, output_dir, tag):
    path = Path(output_dir) / f"predictions_{tag}.csv"
    frame_data.to_csv(path, index=False)
    cls, trading = evaluate_predictions(
        predictions_path=str(path),
        data_path=data_path,
        output_alignment=str(Path(output_dir) / f"eval_{tag}"),
    )
    return {
        "config": tag,
        "n": len(frame_data),
        "accuracy": float(cls["accuracy"].mean()),
        "mcc": float(cls["mcc"].mean()),
        "cumulative_return": float(trading["cumulative_return"].mean()),
        "sharpe": float(trading["sharpe_ratio"].mean()),
        "max_drawdown": float(trading["max_drawdown"].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-store", required=True)
    parser.add_argument("--test-store", required=True)
    parser.add_argument("--data", default="data/processed/market_data_with_regime.csv")
    parser.add_argument("--out-dir", default="results_hypotheses/ensembles")
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation = response_rows(load_store(args.validation_store), args.models)
    test = response_rows(load_store(args.test_store), args.models)
    if not validation or not test:
        raise SystemExit("Validation and test must both contain complete common cases.")

    validation_scores = {}
    for model in args.models:
        rows = []
        for date, ticker, payload, by_model in validation:
            rows.append((date, ticker, payload, {model: by_model[model]}))
        single = frame(rows, [model], CONFIDENCE_WEIGHTED_VOTING)
        validation_scores[model] = evaluate(single, args.data, out_dir, f"validation_{model}")

    best_models = sorted(
        args.models,
        key=lambda model: validation_scores[model]["cumulative_return"],
        reverse=True,
    )[:max(2, min(3, len(args.models)))]

    regime_weights = {"default": {}}
    regimes = sorted({payload.get("market_regime") for _, _, payload, _ in validation})
    for regime in regimes:
        regime_weights[regime] = {}
        for model in args.models:
            subset = [
                (date, ticker, payload, {model: by_model[model]})
                for date, ticker, payload, by_model in validation
                if payload.get("market_regime") == regime
            ]
            score = evaluate(
                frame(subset, [model], CONFIDENCE_WEIGHTED_VOTING),
                args.data, out_dir, f"validation_{model}_{regime}",
            )
            regime_weights[regime][model] = max(score["cumulative_return"], 0.0) + 0.01
    regime_weights["default"] = {
        model: max(validation_scores[model]["cumulative_return"], 0.0) + 0.01
        for model in args.models
    }
    (out_dir / "regime_weights.json").write_text(
        json.dumps(regime_weights, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    results = [
        evaluate(
            frame(test, best_models, CONFIDENCE_WEIGHTED_VOTING),
            args.data, out_dir, "test_best_return_models",
        ),
        evaluate(
            frame(test, args.models, REGIME_WEIGHTED_VOTING, regime_weights),
            args.data, out_dir, "test_regime_weighted",
        ),
    ]
    (out_dir / "hypothesis_summary.json").write_text(
        json.dumps(
            {"validation_scores": validation_scores, "best_models": best_models,
             "test_results": results},
            ensure_ascii=False, indent=2,
        ), encoding="utf-8",
    )
    print(json.dumps({"validation_scores": validation_scores,
                      "best_models": best_models, "test_results": results},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
