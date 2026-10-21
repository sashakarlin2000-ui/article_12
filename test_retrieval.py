import pandas as pd
import pytest

from retrieval import FEATURES, find_similar_days


def make_data(n=15):
    dates = pd.bdate_range("2024-01-01", periods=n)
    rows = []
    close = 100.0

    for i, date in enumerate(dates):
        close *= 1 + (0.01 if i % 2 == 0 else -0.003)

        rows.append({
            "Date": date,
            "Ticker": "TEST",
            "Close": close,
            "return_5d": i * 0.01 + 0.01,
            "volatility_5d": 0.01 + i * 0.001,
            "volatility_20d": 0.02 + i * 0.001,
            "RSI": 40 + i,
            "distance_from_MA20": -0.02 + i * 0.005,
            "distance_from_MA200": -0.1 + i * 0.01,
            "market_regime": "BEAR" if i < 7 else "BULL",
        })

    return pd.DataFrame(rows)


def test_only_past_dates_and_completed_windows():
    data = make_data()
    query_date = data.iloc[-1]["Date"]

    result = find_similar_days(
        query_date,
        data=data,
        ticker="TEST",
        top_k=3,
    )

    assert len(result["similar_days"]) == 3

    for item in result["similar_days"]:
        assert pd.Timestamp(item["date"]) < query_date
        assert pd.Timestamp(item["target_end_date"]) < query_date


def test_regime_aware_returns_same_regime():
    data = make_data()
    query_date = data.iloc[-1]["Date"]

    result = find_similar_days(
        query_date,
        data=data,
        ticker="TEST",
        regime_aware=True,
    )

    assert result["current_regime"] == "BULL"

    for item in result["similar_days"]:
        assert item["regime"] == "BULL"


def test_missing_regime_column_raises_error():
    data = make_data().drop(columns="market_regime")

    with pytest.raises(ValueError, match="market_regime"):
        find_similar_days(
            data.iloc[-1]["Date"],
            data=data,
            ticker="TEST",
            regime_aware=True,
        )


def test_ambiguous_ticker_requires_ticker():
    data = pd.concat([
        make_data(),
        make_data().assign(Ticker="OTHER"),
    ], ignore_index=True)

    with pytest.raises(ValueError, match="несколько тикеров"):
        find_similar_days(
            data.iloc[-1]["Date"],
            data=data,
        )


def test_top_k_must_be_positive():
    with pytest.raises(ValueError):
        find_similar_days(
            "2024-01-01",
            data=make_data(),
            ticker="TEST",
            top_k=0,
        )


def test_fallback_when_not_enough_analogues():
    data = make_data(4)

    result = find_similar_days(
        data.iloc[-1]["Date"],
        data=data,
        ticker="TEST",
        top_k=10,
    )

    assert result["returned_count"] < 10
    assert result["fallback_used"] is True


def test_forecast_horizon_excludes_overlapping_windows():
    data = make_data()
    query_date = data.iloc[-1]["Date"]

    result = find_similar_days(
        query_date,
        data=data,
        ticker="TEST",
        top_k=10,
        forecast_horizon=3,
    )

    for item in result["similar_days"]:
        assert pd.Timestamp(item["target_end_date"]) < query_date


def test_forward_return_is_not_a_similarity_feature():
    assert "forward_return" not in FEATURES
    assert "target" not in FEATURES

