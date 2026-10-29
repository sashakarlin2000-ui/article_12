"""Генерация retrieval_records.csv: обычный и regime-aware retrieval.

Запускается поверх market_data_with_regime.csv, если он есть,
иначе поверх market_data.csv без режима.
По умолчанию берутся последние N торговых дней каждого тикера.
"""

import argparse

import pandas as pd

from retrieval import find_similar_days

DEFAULT_DATA = "data/processed/market_data_with_regime.csv"
FALLBACK_DATA = "market_data.csv"


def main():
    parser = argparse.ArgumentParser(description="retrieval_records.csv")
    parser.add_argument("--data", default=None)
    parser.add_argument("--last-n-dates", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--regime-aware", action="store_true")
    parser.add_argument(
        "--output", default=None,
        help="Путь вывода; по умолчанию retrieval_records.csv "
             "или retrieval_records_regime.csv для regime-aware",
    )
    args = parser.parse_args()

    data_path = args.data
    if data_path is None:
        import os
        data_path = DEFAULT_DATA if os.path.exists(DEFAULT_DATA) else FALLBACK_DATA

    regime_aware = args.regime_aware
    output = args.output or (
        "retrieval_records_regime.csv" if regime_aware else "retrieval_records.csv"
    )

    data = pd.read_csv(data_path)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    records = []
    for ticker in sorted(data["Ticker"].dropna().unique()):
        dates = sorted(
            data.loc[data["Ticker"] == ticker, "Date"].dropna().unique()
        )
        for date in dates[-args.last_n_dates:]:
            result = find_similar_days(
                current_day=pd.Timestamp(date),
                ticker=str(ticker),
                top_k=args.top_k,
                data=data,
                regime_aware=regime_aware,
            )
            for rank, item in enumerate(result["similar_days"], start=1):
                records.append({
                    "current_date": result["current_date"],
                    "ticker": result["ticker"],
                    "mode": result["mode"],
                    "current_regime": result.get("current_regime"),
                    "rank": rank,
                    "analogue_date": item["date"],
                    "analogue_ticker": item.get("ticker"),
                    "similarity": item["similarity"],
                    "forward_return": item["forward_return"],
                    "target_end_date": item["target_end_date"],
                    "forecast_horizon": result["forecast_horizon"],
                    "fallback_used": result["fallback_used"],
                })

    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)

    print(f"Сохранено аналогов: {len(frame)}")
    print(f"Файл: {output}")
    print(frame.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
