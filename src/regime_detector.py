"""Модуль определения режимов рынка: KMeans / GMM / HMM."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

from regime_utils import (
    assign_regime_names,
    compute_cluster_profiles,
    compute_regime_distribution,
    plot_cluster_heatmap,
    plot_kmeans_vs_gmm,
    plot_regime_timeline,
    validate_no_leakage,
)

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Определение режимов рынка с защитой от утечки будущей информации."""

    def __init__(self, config_path: str | Path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.features: List[str] = self.config["features"]
        self.random_state: int = self.config["random_state"]
        self.scaler: StandardScaler | None = None
        self.model = None
        self.cluster_names: Dict[int, str] = {}

    # ---------- данные ----------
    def load_data(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["Date"])
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Нет признаков: {missing}")
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        logger.info("Loaded %d rows, %d tickers", len(df), df["Ticker"].nunique())
        return df

    def split_train_test(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        cutoff = pd.to_datetime(self.config["train_end_date"])
        train = df[df["Date"] <= cutoff].copy()
        test = df[df["Date"] > cutoff].copy()
        validate_no_leakage(train, test)
        return train, test

    # ---------- нормализация ----------
    def fit_scaler(self, train_df: pd.DataFrame) -> None:
        self.scaler = StandardScaler()
        self.scaler.fit(train_df[self.features])
        logger.info("Scaler fitted on train (n=%d)", len(train_df))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Scaler не обучен: вызовите fit_scaler()")
        return self.scaler.transform(df[self.features])

    # ---------- модели ----------
    def fit_kmeans(self, X_train: np.ndarray) -> None:
        cfg = self.config.get("kmeans", {})
        self.model = KMeans(
            n_clusters=self.config["n_clusters"],
            n_init=cfg.get("n_init", 10),
            max_iter=cfg.get("max_iter", 300),
            random_state=self.random_state,
        )
        self.model.fit(X_train)
        logger.info("KMeans fitted: n_clusters=%d", self.config["n_clusters"])

    def fit_gmm(self, X_train: np.ndarray) -> None:
        cfg = self.config.get("gmm", {})
        self.model = GaussianMixture(
            n_components=self.config["n_clusters"],
            covariance_type=cfg.get("covariance_type", "full"),
            n_init=cfg.get("n_init", 5),
            max_iter=cfg.get("max_iter", 200),
            random_state=self.random_state,
        )
        self.model.fit(X_train)
        logger.info("GMM fitted: n_components=%d", self.config["n_clusters"])

    def fit_hmm(self, X_train: np.ndarray) -> None:
        if not HMM_AVAILABLE:
            raise ImportError("Установите hmmlearn: pip install hmmlearn")
        cfg = self.config.get("hmm", {})
        self.model = GaussianHMM(
            n_components=self.config["n_clusters"],
            covariance_type=cfg.get("covariance_type", "diag"),
            n_iter=cfg.get("n_iter", 100),
            random_state=self.random_state,
        )
        self.model.fit(X_train)
        logger.info("HMM fitted: n_components=%d", self.config["n_clusters"])

    def fit(self, X_train: np.ndarray) -> None:
        method = self.config["method"]
        if method == "kmeans":
            self.fit_kmeans(X_train)
        elif method == "gmm":
            self.fit_gmm(X_train)
        elif method == "hmm":
            self.fit_hmm(X_train)
        else:
            raise ValueError(f"Неизвестный метод: {method}")

    # ---------- предсказание ----------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.transform(df)
        out = df.copy()

        if self.config["method"] == "hmm":
            labels = self.model.predict(X)
            proba = self.model.predict_proba(X)
        else:
            labels = self.model.predict(X)
            proba = self.model.predict_proba(X) if hasattr(self.model, "predict_proba") else None

        out["market_regime"] = labels
        if proba is not None:
            for i in range(proba.shape[1]):
                out[f"regime_prob_{i}"] = proba[:, i]

        # сглаживание
        window = self.config.get("smoothing_window", 1)
        if window and window > 1:
            out = self._smooth(out, window)

        return out

    def _smooth(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Rolling majority vote по каждому тикеру отдельно."""
        def _majority(s: pd.Series) -> pd.Series:
            return s.rolling(window, min_periods=1, center=True).apply(
                lambda x: pd.Series(x).mode().iloc[0], raw=False
            )
        df["market_regime"] = (
            df.groupby("Ticker")["market_regime"]
              .transform(lambda s: _majority(s).astype(int))
        )
        return df

    # ---------- интерпретация ----------
    def interpret(self, df: pd.DataFrame) -> pd.DataFrame:
        profiles = compute_cluster_profiles(df, "market_regime", self.features)
        self.cluster_names = assign_regime_names(profiles)
        df["market_regime_label"] = df["market_regime"].map(self.cluster_names)
        logger.info("Cluster names: %s", self.cluster_names)
        return df, profiles

    # ---------- сохранение ----------
    def save(self, df: pd.DataFrame, profiles: pd.DataFrame) -> None:
        out_cfg = self.config["output"]
        tables_dir = Path(out_cfg["tables_dir"])
        figures_dir = Path(out_cfg["figures_dir"])
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        data_path = Path(out_cfg["data_path"])
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)

        profiles.to_csv(tables_dir / "cluster_profiles.csv", index=False)
        compute_regime_distribution(df).to_csv(
            tables_dir / "regime_distribution.csv", index=False
        )
        pd.DataFrame({
            "feature": self.features,
            "mean": self.scaler.mean_,
            "scale": self.scaler.scale_,
        }).to_csv(tables_dir / "scaler_params.csv", index=False)

        # графики
        plot_regime_timeline(df, output_path=figures_dir / "regime_timeline.png")
        plot_cluster_heatmap(profiles, output_path=figures_dir / "cluster_heatmap.png")

        logger.info("Saved data → %s", data_path)
        logger.info("Saved tables → %s", tables_dir)
        logger.info("Saved figures → %s", figures_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Определение режимов рынка")
    parser.add_argument("--config", required=True, help="Путь к config_regime.yaml")
    parser.add_argument("--input", default="data/raw/market_data.csv")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    detector = RegimeDetector(args.config)
    df = detector.load_data(args.input)
    train, test = detector.split_train_test(df)

    detector.fit_scaler(train)
    X_train = detector.transform(train)
    detector.fit(X_train)

    df_out = detector.predict(df)
    df_out, profiles = detector.interpret(df_out)
    detector.save(df_out, profiles)


if __name__ == "__main__":
    main()
