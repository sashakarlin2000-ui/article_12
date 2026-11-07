"""Run leakage-safe naive and classical ML baselines on the final test window."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import evaluate_predictions


FEATURES = [
    "return_1d",
    "return_5d",
    "volatility_5d",
    "volatility_20d",
    "distance_from_MA20",
    "distance_from_MA200",
    "RSI",
]


def labels(values):
    return pd.Series(values).map({1: "UP", 0: "DOWN"}).fillna("HOLD")


def make_predictions(data, days):
    data = data.sort_values(["Ticker", "Date"]).copy()
    rows = []
    for ticker, group in data.groupby("Ticker", sort=True):
        group = group.dropna(subset=FEATURES + ["target"]).copy()
        group = group[group["Close"].shift(-1).notna()]
        test = group.tail(days)
        train = group.iloc[:-days]
        if train.empty or test.empty:
            continue

        X_train = train[FEATURES]
        y_train = train["target"].astype(int)
        X_test = test[FEATURES]

        majority = int(y_train.mean() >= 0.5)
        models = {
            "naive_majority": None,
            "logistic_regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=42),
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
        }
        for method, model in models.items():
            if model is None:
                predictions = [majority] * len(test)
                probabilities = [float(majority)] * len(test)
            else:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test)[:, 1]
            for row, prediction, probability in zip(
                test.itertuples(index=False), predictions, probabilities
            ):
                rows.append(
                    {
                        "Date": row.Date.strftime("%Y-%m-%d"),
                        "Ticker": ticker,
                        "Prediction": labels([prediction]).iloc[0],
                        "Probability": round(float(probability), 4),
                        "Score": round(float(probability) * 2 - 1, 4),
                        "Method": method,
                        "Regime": row.market_regime_label,
                        "RetrievalMode": "baseline",
                        "NAnalogues": 0,
                        "FallbackUsed": False,
                        "ModelsFailed": "[]",
                        "ModelPredictions": "[]",
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/market_data_with_regime.csv")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--out-dir", default="results_hypotheses/baselines")
    args = parser.parse_args()

    data = pd.read_csv(args.data, parse_dates=["Date"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = make_predictions(data, args.days)
    if predictions.empty:
        raise SystemExit("No baseline predictions were produced.")

    for method, frame in predictions.groupby("Method", sort=True):
        path = out_dir / f"predictions_{method}.csv"
        frame.to_csv(path, index=False)
        cls, trading = evaluate_predictions(
            predictions_path=str(path),
            data_path=args.data,
            output_alignment=str(out_dir / f"eval_{method}"),
        )
        summary = {
            "method": method,
            "n_predictions": len(frame),
            "accuracy": float(cls["accuracy"].mean()),
            "balanced_accuracy": float(cls["balanced_accuracy"].mean()),
            "mcc": float(cls["mcc"].mean()),
            "cumulative_return": float(trading["cumulative_return"].mean()),
            "sharpe": float(trading["sharpe_ratio"].mean()),
            "max_drawdown": float(trading["max_drawdown"].mean()),
        }
        print(summary)


if __name__ == "__main__":
    main()
