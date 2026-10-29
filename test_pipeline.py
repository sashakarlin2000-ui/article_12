"""Интеграционный тест Роли 7 — сквозной запуск всех модулей.

Проверяет связку: data_with_regime -> retrieval -> LLM (mock) ->
ensemble -> predictions -> backtest -> metrics.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import build_results_tables, calculate_trading_results
from backtest import load_data as bt_load_data
from ensemble import MAJORITY_VOTING
from pipeline import predict_one_day, run_pipeline


@pytest.fixture
def integration_data():
    """Синтетический датасет с режимами и полным набором признаков."""
    rng = np.random.default_rng(11)
    dates = pd.date_range("2023-01-02", periods=260, freq="B")
    close = 100 + np.cumsum(rng.normal(0.0005, 0.01, len(dates)))
    df = pd.DataFrame({
        "Date": dates,
        "Ticker": "TEST",
        "Close": close,
        "return_1d": rng.normal(0, 0.01, len(dates)),
        "return_5d": pd.Series(close).pct_change(5).values,
        "volatility_5d": rng.uniform(0.005, 0.03, len(dates)),
        "volatility_20d": rng.uniform(0.005, 0.04, len(dates)),
        "MA20": pd.Series(close).rolling(20).mean().values,
        "MA200": pd.Series(close).rolling(200).mean().values,
        "distance_from_MA20": np.zeros(len(dates)),
        "distance_from_MA200": np.zeros(len(dates)),
        "RSI": rng.uniform(25, 75, len(dates)),
        "target": rng.integers(0, 2, len(dates)),
        "market_regime": rng.integers(0, 4, len(dates)),
        "market_regime_label": rng.choice(
            ["Bull", "Bear", "Sideways", "High Volatility"], len(dates)
        ),
    })

    df["distance_from_MA20"] = df["Close"] / df["MA20"] - 1
    df["distance_from_MA200"] = df["Close"] / df["MA200"] - 1

    return df


def test_predict_one_day_full_chain(integration_data):
    data = integration_data
    df = data.dropna().reset_index(drop=True)
    row = df.iloc[-10]

    from llm_predictor import MockLLMProvider
    providers = [MockLLMProvider(), MockLLMProvider(model_name="mock2")]

    result = predict_one_day(
        row,
        df,
        providers,
        retrieval_mode="standard",
        top_k=3,
        ensemble_method=MAJORITY_VOTING,
        use_cache=False,
    )

    assert result["Prediction"] in {"UP", "DOWN", "HOLD"}
    assert result["Ticker"] == "TEST"
    assert result["NAnalogues"] == 3
    assert result["RetrievalMode"] == "standard"
    model_preds = json.loads(result["ModelPredictions"])
    assert len(model_preds) == len(providers)
    for mp in model_preds:
        assert mp["prediction"] in {"UP", "DOWN", "HOLD"}
        assert 0.0 <= mp["confidence"] <= 1.0


def test_predict_one_day_regime_aware(integration_data):
    df = integration_data
    row = df.dropna().iloc[-10]

    from llm_predictor import MockLLMProvider
    result = predict_one_day(
        row,
        df,
        [MockLLMProvider()],
        retrieval_mode="regime-aware",
        use_cache=False,
    )
    assert result["RetrievalMode"] == "regime-aware"
    # все аналоги того же режима
    model_preds = json.loads(result["ModelPredictions"])
    assert len(model_preds) == 1


def test_run_pipeline_produces_predictions(integration_data, tmp_path):
    csv_path = tmp_path / "data_regime.csv"
    integration_data.to_csv(csv_path, index=False)

    preds = run_pipeline(
        data_path=str(csv_path),
        tickers=["TEST"],
        last_n_days=8,
        providers=None,  # берутся дефолтные mock-модели
        use_cache=False,
        output_predictions=str(tmp_path / "predictions.csv"),
        output_dir=str(tmp_path / "res"),
        cache_dir=str(tmp_path / "llm_cache"),
    )

    out_file = Path(tmp_path / "predictions.csv")
    assert out_file.exists()
    saved = pd.read_csv(out_file)
    assert len(saved) == len(preds) == 8
    assert set(saved["Prediction"]).issubset({"UP", "DOWN", "HOLD"})

    expected_last8 = integration_data["Date"].sort_values().unique()[-9:]
    assert set(saved["Date"].tolist()).issubset(
        {str(pd.Timestamp(d).date()) for d in expected_last8}
    )


def test_run_pipeline_to_backtest_connection(integration_data, tmp_path):
    """Полная связка: pipeline -> backtest -> таблицы результатов."""
    csv_path = tmp_path / "data_regime.csv"
    integration_data.to_csv(csv_path, index=False)

    preds = run_pipeline(
        data_path=str(csv_path),
        tickers=["TEST"],
        last_n_days=30,
        providers=None,
        use_cache=False,
        output_predictions=str(tmp_path / "predictions.csv"),
        output_dir=str(tmp_path / "res"),
        cache_dir=str(tmp_path / "llm_cache"),
    )

    data = bt_load_data(
        market_path=str(csv_path),
        predictions_path=str(tmp_path / "predictions.csv"),
    )
    data = calculate_trading_results(data, allow_short=True)
    results_classification, results_trading = build_results_tables(data)

    assert "accuracy" in results_classification.columns
    assert "sharpe_ratio" in results_trading.columns
    assert results_trading["n_days"].iloc[0] == len(data)
    assert len(data) == 30  # 30 прогнозов → 30 оценённых дней
