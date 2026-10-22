
import pandas as pd

from retrieval import (
    DEFAULT_DATA_PATH,
    find_similar_days,
)

data = pd.read_csv(DEFAULT_DATA_PATH)
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

records = []

for ticker in sorted(data["Ticker"].dropna().unique()):
    dates = sorted(
        data.loc[data["Ticker"] == ticker, "Date"].dropna().unique()
    )

    for date in dates[-3:]:
        result = find_similar_days(
            current_day=pd.Timestamp(date),
            ticker=str(ticker),
            top_k=3,
            data=data,
            regime_aware=False,
        )

        for rank, item in enumerate(
            result["similar_days"], start=1
        ):
            records.append({
                "current_date": result["current_date"],
                "ticker": result["ticker"],
                "mode": result["mode"],
                "rank": rank,
                "analogue_date": item["date"],
                "similarity": item["similarity"],
                "forward_return": item["forward_return"],
                "target_end_date": item["target_end_date"],
                "forecast_horizon": result["forecast_horizon"],
                "fallback_used": result["fallback_used"],
            })

output = pd.DataFrame(records)
output.to_csv("retrieval_records.csv", index=False)

print(f"Сохранено аналогов: {len(output)}")
print("Файл: retrieval_records.csv")
print(output.head(10).to_string(index=False))
