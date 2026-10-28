"""Фикстуры для тестов модуля режимов."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Синтетический датафрейм с 2 тикерами и 300 днями."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    rows = []
    for ticker in ["SPX", "AAPL"]:
        close = 100 + np.cumsum(rng.normal(0, 1, len(dates)))
        df = pd.DataFrame({
            "Date": dates,
            "Ticker": ticker,
            "Close": close,
            "Open": close - rng.normal(0, 0.5, len(dates)),
            "High": close + rng.normal(0, 0.5, len(dates)),
            "Low": close - rng.normal(0, 0.5, len(dates)),
            "Volume": rng.integers(1_000_000, 5_000_000, len(dates)),
            "return_1d": rng.normal(0, 0.01, len(dates)),
            "return_5d": rng.normal(0, 0.02, len(dates)),
            "volatility_5d": rng.uniform(0.005, 0.03, len(dates)),
            "volatility_20d": rng.uniform(0.005, 0.04, len(dates)),
            "MA20": close - rng.normal(0, 1, len(dates)),
            "MA200": close - rng.normal(0, 2, len(dates)),
            "distance_from_MA20": rng.normal(0, 0.02, len(dates)),
            "distance_from_MA200": rng.normal(0, 0.05, len(dates)),
            "RSI": rng.uniform(20, 80, len(dates)),
            "target": rng.integers(0, 2, len(dates)),
        })
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def config_path(tmp_path):
    """Временный YAML-конфиг."""
    cfg = tmp_path / "config_regime.yaml"
    cfg.write_text(
        """
n_clusters: 4
method: kmeans
features:
  - return_5d
  - volatility_20d
  - distance_from_MA20
  - distance_from_MA200
scaler: standard
random_state: 42
smoothing_window: 3
train_end_date: "2021-06-30"
kmeans:
  n_init: 5
  max_iter: 100
output:
  data_path: "{tmp}/data.csv"
  tables_dir: "{tmp}/tables"
  figures_dir: "{tmp}/figures"
""".format(tmp=str(tmp_path)),
        encoding="utf-8",
    )
    return cfg
