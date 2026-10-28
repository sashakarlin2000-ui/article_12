"""Вспомогательные функции для модуля определения режимов рынка."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

REGIME_COLORS = {
    "Bull": "#2ecc71",
    "Bear": "#e74c3c",
    "Sideways": "#95a5a6",
    "High Volatility": "#f39c12",
}


def compute_cluster_profiles(
    df: pd.DataFrame,
    label_col: str = "market_regime",
    feature_cols: List[str] | None = None,
) -> pd.DataFrame:
    """Средние значения признаков по кластерам + count и share."""
    if feature_cols is None:
        feature_cols = ["return_5d", "volatility_20d",
                        "distance_from_MA20", "distance_from_MA200"]
    profiles = df.groupby(label_col)[feature_cols].mean()
    counts = df[label_col].value_counts().sort_index()
    profiles["count"] = counts
    profiles["share"] = counts / counts.sum()
    return profiles.reset_index().rename(columns={label_col: "cluster"})


def assign_regime_names(profiles: pd.DataFrame) -> Dict[int, str]:
    """
    Присваивает имена кластерам на основе средних значений.
    Правила:
      - Bull: return_5d > 0 и volatility_20d ниже медианы
      - Bear: return_5d < 0 и volatility_20d выше медианы
      - High Volatility: volatility_20d максимальная
      - Sideways: остальные (|return_5d| мал)
    """
    vol_median = profiles["volatility_20d"].median()
    vol_max = profiles["volatility_20d"].max()
    names: Dict[int, str] = {}
    used = set()

    # 1) High Volatility — кластер с максимальной волатильностью
    hv_cluster = int(profiles.loc[profiles["volatility_20d"].idxmax(), "cluster"])
    names[hv_cluster] = "High Volatility"
    used.add(hv_cluster)

    # 2) Bear — отрицательная доходность + высокая волатильность
    bear_candidates = profiles[
        (profiles["return_5d"] < 0)
        & (profiles["volatility_20d"] >= vol_median)
        & (~profiles["cluster"].isin(used))
    ]
    if not bear_candidates.empty:
        bear_cluster = int(bear_candidates["volatility_20d"].idxmax()
                           if bear_candidates["volatility_20d"].idxmax() in bear_candidates.index
                           else bear_candidates.index[0])
        # idxmax возвращает индекс строки, а нам нужен cluster id
        bear_row = bear_candidates.loc[bear_candidates["volatility_20d"].idxmax()]
        names[int(bear_row["cluster"])] = "Bear"
        used.add(int(bear_row["cluster"]))

    # 3) Bull — положительная доходность + низкая волатильность
    bull_candidates = profiles[
        (profiles["return_5d"] > 0)
        & (~profiles["cluster"].isin(used))
    ]
    if not bull_candidates.empty:
        bull_row = bull_candidates.loc[bull_candidates["return_5d"].idxmax()]
        names[int(bull_row["cluster"])] = "Bull"
        used.add(int(bull_row["cluster"]))

    # 4) Остальные — Sideways
    for cid in profiles["cluster"]:
        if int(cid) not in used:
            names[int(cid)] = "Sideways"

    return names


def compute_regime_distribution(
    df: pd.DataFrame, label_col: str = "market_regime_label"
) -> pd.DataFrame:
    """Доли, count, средняя длительность каждого режима."""
    rows = []
    for regime, g in df.groupby(label_col):
        # длительность: считаем серии подряд идущих меток
        series = (df[label_col] == regime).astype(int)
        groups = (series.diff() != 0).cumsum()
        durations = series.groupby(groups).sum()
        durations = durations[durations > 0]
        rows.append({
            "regime_label": regime,
            "count": len(g),
            "share": len(g) / len(df),
            "start_date": g["Date"].min(),
            "end_date": g["Date"].max(),
            "mean_duration_days": float(durations.mean()) if len(durations) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("share", ascending=False)


def validate_no_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Проверяет, что даты train и test не пересекаются."""
    train_max = pd.to_datetime(train_df["Date"]).max()
    test_min = pd.to_datetime(test_df["Date"]).min()
    if train_max >= test_min:
        raise ValueError(
            f"Утечка данных: train_max={train_max} >= test_min={test_min}"
        )
    logger.info("Leakage check passed: train_max=%s < test_min=%s", train_max, test_min)


def plot_regime_timeline(
    df: pd.DataFrame,
    price_col: str = "Close",
    label_col: str = "market_regime_label",
    ticker: str = "SPX",
    output_path: str | Path = "results/figures/regime_timeline.png",
) -> None:
    """График цены с раскраской фона по режимам."""
    sub = df[df["Ticker"] == ticker].sort_values("Date").copy()
    sub["Date"] = pd.to_datetime(sub["Date"])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(sub["Date"], sub[price_col], color="black", lw=1, label=f"{ticker} Close")

    # фон по режимам
    for regime, color in REGIME_COLORS.items():
        mask = sub[label_col] == regime
        if mask.any():
            ax.fill_between(
                sub["Date"], sub[price_col].min(), sub[price_col].max(),
                where=mask, color=color, alpha=0.15, label=regime,
            )

    ax.set_title(f"{ticker}: режимы рынка")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Цена")
    ax.legend(loc="upper left", ncol=5, fontsize=8)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved %s", output_path)


def plot_cluster_heatmap(
    profiles: pd.DataFrame,
    output_path: str | Path = "results/figures/cluster_heatmap.png",
) -> None:
    """Тепловая карта профилей кластеров (z-score по признакам)."""
    feature_cols = ["return_5d", "volatility_20d",
                    "distance_from_MA20", "distance_from_MA200"]
    mat = profiles.set_index("cluster")[feature_cols]
    mat_z = (mat - mat.mean()) / mat.std(ddof=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(mat_z, annot=mat.round(4), fmt="", cmap="RdBu_r",
                center=0, ax=ax)
    ax.set_title("Профили кластеров (z-score, подписи — исходные средние)")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved %s", output_path)


def plot_kmeans_vs_gmm(
    df: pd.DataFrame,
    kmeans_col: str = "market_regime",
    gmm_col: str = "market_regime_gmm",
    output_path: str | Path = "results/figures/kmeans_vs_gmm.png",
) -> None:
    """Сравнение меток KMeans и GMM во времени."""
    sub = df[df["Ticker"] == "SPX"].sort_values("Date").copy()
    sub["Date"] = pd.to_datetime(sub["Date"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].scatter(sub["Date"], sub[kmeans_col], s=4, c=sub[kmeans_col], cmap="tab10")
    axes[0].set_title("KMeans")
    axes[1].scatter(sub["Date"], sub[gmm_col], s=4, c=sub[gmm_col], cmap="tab10")
    axes[1].set_title("GMM")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved %s", output_path)
