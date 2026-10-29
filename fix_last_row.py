"""
Разовая правка уже собранного market_data.csv БЕЗ повторной загрузки из yfinance.

Удаляет последнюю строку каждого тикера (у неё ложный target=0, т.к. Close(t+1)
неизвестен) и пересобирает data_manifest.json. Остальные строки не меняются,
поэтому кэш ответов LLM, режимы и retrieval по этим датам остаются валидными.

Запуск:  python fix_last_row.py
"""
import json
from pathlib import Path

import pandas as pd

from features import build_manifest

CSV = Path("market_data.csv")

df = pd.read_csv(CSV, parse_dates=["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

last_idx = df.groupby("Ticker").tail(1).index
print("Удаляются последние строки:")
print(df.loc[last_idx, ["Ticker", "Date", "Close", "target"]].to_string(index=False))

df = df.drop(index=last_idx).reset_index(drop=True)
df.to_csv(CSV, index=False)

manifest = build_manifest(df, csv_path=CSV)
Path("data_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"\nГотово: {len(df)} строк, период {manifest['date_start']} — {manifest['date_end']}")