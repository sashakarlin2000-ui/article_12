from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "return_5d",
    "volatility_5d",
    "volatility_20d",
    "RSI",
    "distance_from_MA20",
    "distance_from_MA200",
]

REGIME_COLUMNS = ("market_regime", "regime")
DEFAULT_DATA_PATH = Path(__file__).resolve().with_name("market_data.csv")


def _prepare_data(
    data: pd.DataFrame,
    forecast_horizon: int,
) -> pd.DataFrame:
    """Validate market data and calculate future outcomes separately."""
    required = {"Date", "Ticker", "Close", *FEATURES}
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            f"В датасете отсутствуют обязательные колонки: {sorted(missing)}"
        )

    df = data.copy()
    df["Date"] = (
        pd.to_datetime(df["Date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=["Date", "Ticker", "Close"])
    df = df.drop_duplicates(
        subset=["Ticker", "Date"],
        keep="last",
    )
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    grouped = df.groupby("Ticker", sort=False)

    # End date of the historical outcome window.
    df["_target_end_date"] = grouped["Date"].shift(-forecast_horizon)

    # Outcome is metadata only; it is not a retrieval feature.
    future_close = grouped["Close"].shift(-forecast_horizon)
    df["forward_return"] = future_close.div(df["Close"]).sub(1)

    return df


def find_similar_days(
    current_day: str | pd.Timestamp | pd.Series | dict[str, Any],
    top_k: int = 3,
    data: pd.DataFrame | str | Path | None = None,
    *,
    ticker: str | None = None,
    regime_aware: bool = False,
    regime_col: str | None = None,
    forecast_horizon: int = 1,
) -> dict[str, Any]:
    """Return the most similar fully observed historical market states.

    Only dates before current_day are eligible. The historical outcome window
    must also end before current_day. Forward returns are never used to
    calculate similarity.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k должен быть целым числом")
    if top_k < 1:
        raise ValueError("top_k должен быть >= 1")

    if (
        isinstance(forecast_horizon, bool)
        or not isinstance(forecast_horizon, int)
    ):
        raise TypeError("forecast_horizon должен быть целым числом")
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon должен быть >= 1")

    if data is None:
        raw = pd.read_csv(DEFAULT_DATA_PATH)
    elif isinstance(data, (str, Path)):
        raw = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        raw = data.copy()
    else:
        raise TypeError(
            "data должен быть DataFrame, путём к CSV или None"
        )

    df = _prepare_data(raw, forecast_horizon)

    supplied_regime = None

    if isinstance(current_day, (pd.Series, dict)):
        current_date = pd.to_datetime(
            current_day["Date"], errors="coerce"
        )
        ticker = ticker or current_day.get("Ticker")

        if regime_col:
            supplied_regime = current_day.get(regime_col)
    else:
        current_date = pd.to_datetime(
            current_day, errors="coerce"
        )

    if pd.isna(current_date):
        raise ValueError("Не удалось распознать current_day")

    current_date = (
        pd.Timestamp(current_date).tz_localize(None).normalize()
    )

    if ticker is None:
        tickers = df.loc[
            df["Date"].dt.normalize().eq(current_date),
            "Ticker",
        ].unique()

        if len(tickers) == 0:
            raise ValueError(
                f"Дата {current_date.date()} отсутствует в данных"
            )
        if len(tickers) > 1:
            raise ValueError(
                "На эту дату несколько тикеров; передайте ticker явно"
            )

        ticker = str(tickers[0])

    same_ticker = df[
        df["Ticker"].astype(str).eq(str(ticker))
    ].copy()

    current_rows = same_ticker[
        same_ticker["Date"].dt.normalize().eq(current_date)
    ]

    if current_rows.empty:
        raise ValueError(
            f"Для тикера {ticker} нет строки на дату "
            f"{current_date.date()}"
        )

    current = current_rows.iloc[-1]

    if regime_col is None:
        regime_col = next(
            (col for col in REGIME_COLUMNS if col in df.columns),
            None,
        )

    if regime_aware and (
        regime_col is None or regime_col not in df.columns
    ):
        raise ValueError(
            "Для regime-aware retrieval нужна колонка "
            "market_regime или regime"
        )

    current_regime = (
        supplied_regime
        if supplied_regime is not None
        else (
            current[regime_col]
            if regime_col and regime_col in current.index
            else None
        )
    )

    if regime_aware and pd.isna(current_regime):
        raise ValueError("У текущего дня не заполнен режим рынка")

    # Both the analogue and its outcome window must be fully in the past.
    candidates = same_ticker[
        (same_ticker["Date"] < current_date)
        & (same_ticker["_target_end_date"] < current_date)
    ].copy()

    if regime_aware:
        candidates = candidates[
            candidates[regime_col].eq(current_regime)
        ]

    candidates = candidates.dropna(
        subset=FEATURES + ["forward_return", "_target_end_date"]
    )

    current_features = pd.to_numeric(
        current[FEATURES],
        errors="coerce",
    )

    if current_features.isna().any():
        raise ValueError(
            "У текущего дня есть пропуски в одном или нескольких признаках"
        )

    result: dict[str, Any] = {
        "current_date": current_date.strftime("%Y-%m-%d"),
        "ticker": str(ticker),
        "current_regime": (
            None if pd.isna(current_regime) else str(current_regime)
        ),
        "mode": "regime-aware" if regime_aware else "standard",
        "forecast_horizon": forecast_horizon,
        "requested_top_k": top_k,
        "returned_count": 0,
        "fallback_used": False,
        "similar_days": [],
    }

    if candidates.empty:
        result["fallback_used"] = True
        result["message"] = (
            "Нет допустимых аналогов. Будущие даты и неполные "
            "прогнозные окна не подставляются."
        )
        return result

    # Fit only on eligible historical candidates.
    scaler = StandardScaler()
    history_scaled = scaler.fit_transform(
        candidates[FEATURES].astype(float)
    )
    current_scaled = scaler.transform(
        current_features.to_frame().T.astype(float)
    )

    candidates["similarity"] = cosine_similarity(
        current_scaled,
        history_scaled,
    ).ravel()

    candidates = candidates.sort_values(
        ["similarity", "Date"],
        ascending=[False, False],
    ).head(top_k)

    for _, row in candidates.iterrows():
        item = {
            "date": row["Date"].strftime("%Y-%m-%d"),
            "similarity": round(float(row["similarity"]), 4),
            "forward_return": float(row["forward_return"]),
            "target_end_date": row["_target_end_date"].strftime(
                "%Y-%m-%d"
            ),
            "ticker": str(row["Ticker"]),
        }

        if (
            regime_col
            and regime_col in row.index
            and pd.notna(row[regime_col])
        ):
            item["regime"] = str(row[regime_col])

        result["similar_days"].append(item)

    result["returned_count"] = len(result["similar_days"])

    if result["returned_count"] < top_k:
        result["fallback_used"] = True
        result["message"] = (
            f"Найдено {result['returned_count']} из {top_k} "
            "запрошенных аналогов. Возвращены только допустимые "
            "исторические записи."
        )

    return result


def format_results(result: dict[str, Any]) -> str:
    """Format retrieval results for display."""
    lines = [
        f"Current date: {result['current_date']}",
        f"Ticker: {result['ticker']}",
    ]

    if result.get("current_regime") is not None:
        lines.append(
            f"Current regime: {result['current_regime']}"
        )

    lines.extend([
        f"Mode: {result['mode']}",
        "",
        "Similar days:",
    ])

    if not result["similar_days"]:
        lines.append(
            result.get("message", "No similar days found.")
        )
    else:
        for item in result["similar_days"]:
            pct = item["forward_return"] * 100
            lines.append(
                f"{item['date']} | "
                f"similarity {item['similarity']:.4f} | "
                f"forward return "
                f"({result['forecast_horizon']} trading day(s)) "
                f"{pct:+.2f}%"
            )

    if result.get("fallback_used"):
        lines.append(
            f"Fallback: {result.get('message', 'Not enough analogues')}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    sample = pd.read_csv(
        DEFAULT_DATA_PATH,
        usecols=["Date", "Ticker"],
    )
    sample["Date"] = pd.to_datetime(sample["Date"])

    dates = sample.loc[
        sample["Ticker"].eq("SPX"),
        "Date",
    ]

    if dates.empty:
        raise SystemExit("В market_data.csv нет тикера SPX")

    query_date = dates.max().strftime("%Y-%m-%d")

    result = find_similar_days(
        query_date,
        ticker="SPX",
        top_k=3,
    )

    print(format_results(result))

