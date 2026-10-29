import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import fetch_market_data


def calculate_rsi(series, period=14):
    """Классический расчет индикатора RSI (Relative Strength Index) через Pandas"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def add_features(df):
    """
    Рассчитывает все технические признаки и таргет для каждого тикера отдельно.
    Используем groupby('Ticker'), чтобы данные разных акций не смешивались в окнах!
    """
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
    processed_groups = []

    for ticker, group in df.groupby('Ticker'):
        group = group.copy()

        # 1. Доходности (returns)
        group['return_1d'] = group['Close'].pct_change(1)
        group['return_5d'] = group['Close'].pct_change(5)

        # 2. Волатильность (скользящее стандартное отклонение доходностей)
        group['volatility_5d'] = group['return_1d'].rolling(5).std()
        group['volatility_20d'] = group['return_1d'].rolling(20).std()

        # 3. Скользящие средние (Moving Averages)
        group['MA20'] = group['Close'].rolling(20).mean()
        group['MA200'] = group['Close'].rolling(200).mean()

        # 4. Дистанция от цены до скользящих средних
        group['distance_from_MA20'] = (group['Close'] - group['MA20']) / group['MA20']
        group['distance_from_MA200'] = (group['Close'] - group['MA200']) / group['MA200']

        # 5. Индикатор RSI
        group['RSI'] = calculate_rsi(group['Close'], period=14)

        # 6. Целевая переменная (Target):
        # Если завтрашний Close > сегодняшнего Close, то 1, иначе 0.
        # shift(-1) «подтягивает» завтрашний день на текущую строчку.
        group['next_close'] = group['Close'].shift(-1)

        # ИСПРАВЛЕНО: сравнение с NaN в pandas даёт False, и .astype(int) превращал
        # последнюю строку тикера в target=0 (ложный «день без роста»).
        # Теперь там, где next_close неизвестен, target = NaN, и clean_dataset()
        # удалит такую строку через dropna().
        group['target'] = np.where(
            group['next_close'].isna(),
            np.nan,
            (group['next_close'] > group['Close']).astype(float),
        )

        # Удаляем временную вспомогательную колонку
        group.drop(columns=['next_close'], inplace=True)
        processed_groups.append(group)

    return pd.concat(processed_groups, ignore_index=True)


def clean_dataset(df):
    """
    Очистка датасета от дубликатов и NaN (которые неизбежно возникают
    из-за MA200 в начале истории и shift(-1) на самой последней строке).
    """
    # Убираем дубликаты по дате и тикеру, если они вдруг есть
    df = df.drop_duplicates(subset=['Date', 'Ticker'])
    initial_rows = len(df)

    # Удаляем строки с пропущенными значениями (NaN),
    # в том числе последнюю строку каждого тикера (target = NaN)
    df = df.dropna().reset_index(drop=True)

    # После удаления NaN target можно безопасно вернуть к целому типу
    df['target'] = df['target'].astype(int)

    print(f"Очистка: удалено {initial_rows - len(df)} строк с NaN (из-за оконных расчетов MA200 и таргета).")
    return df


def check_no_label_on_last_day(df, raw_df):
    """
    Защита от повторения ошибки: для последней даты каждого тикера в сырых
    данных не должно быть строки с target в итоговом датасете.
    """
    last_dates = raw_df.groupby('Ticker')['Date'].max()
    for ticker, last_date in last_dates.items():
        mask = (df['Ticker'] == ticker) & (df['Date'] == last_date)
        if mask.any():
            raise AssertionError(
                f"{ticker}: последняя дата {last_date.date()} осталась в датасете "
                "с target, хотя Close(t+1) неизвестен."
            )


def build_manifest(df, csv_path="market_data.csv"):
    """
    data_manifest.json: схема колонок, период, тикеры, пропуски, хеш файла.
    Результаты связаны с конкретной версией датасета.
    """
    csv_path = Path(csv_path)
    content = csv_path.read_bytes()
    return {
        "file": csv_path.name,
        "dataset_sha256": hashlib.sha256(content).hexdigest(),
        "n_rows": int(len(df)),
        "n_tickers": int(df["Ticker"].nunique()),
        "tickers": sorted(df["Ticker"].unique().tolist()),
        "date_start": str(df["Date"].min().date()),
        "date_end": str(df["Date"].max().date()),
        "columns": list(df.columns),
        "missing_per_column": {
            col: int(df[col].isna().sum()) for col in df.columns
        },
        "duplicates": int(df.duplicated(subset=["Date", "Ticker"]).sum()),
        "generated_at": str(pd.Timestamp.now()),
    }


if __name__ == "__main__":
    print("--- Шаг 1: Загрузка данных ---")
    raw_data = fetch_market_data()

    print("\n--- Шаг 2: Расчет признаков ---")
    df_features = add_features(raw_data)

    print("\n--- Шаг 3: Очистка данных ---")
    final_df = clean_dataset(df_features)
    check_no_label_on_last_day(final_df, raw_data)

    print("\n--- Шаг 4: Сохранение в файл market_data.csv ---")
    final_df.to_csv("market_data.csv", index=False)

    print("\n--- Шаг 5: Манифест датасета ---")
    manifest = build_manifest(final_df)
    Path("data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Готово! Итоговый файл сохранен. Размер датасета: {final_df.shape}")
    print("\nПервые 5 строк готового датасета:")
    print(final_df.head())