"""Тесты модуля определения режимов."""
import numpy as np
import pandas as pd

from regime_detector import RegimeDetector


def test_output_shape(sample_df, config_path, tmp_path):
    det = RegimeDetector(config_path)
    train, test = det.split_train_test(sample_df)
    det.fit_scaler(train)
    det.fit(det.transform(train))

    out = det.predict(sample_df)
    out, profiles = det.interpret(out)

    assert "market_regime" in out.columns
    assert "market_regime_label" in out.columns
    assert len(out) == len(sample_df)
    assert len(profiles) == 4


def test_no_nan_in_labels(sample_df, config_path):
    det = RegimeDetector(config_path)
    train, _ = det.split_train_test(sample_df)
    det.fit_scaler(train)
    det.fit(det.transform(train))
    out = det.predict(sample_df)
    assert out["market_regime"].isna().sum() == 0


def test_label_range(sample_df, config_path):
    det = RegimeDetector(config_path)
    train, _ = det.split_train_test(sample_df)
    det.fit_scaler(train)
    det.fit(det.transform(train))
    out = det.predict(sample_df)
    assert out["market_regime"].between(0, 3).all()


def test_interpretation_all_regimes(sample_df, config_path):
    det = RegimeDetector(config_path)
    train, _ = det.split_train_test(sample_df)
    det.fit_scaler(train)
    det.fit(det.transform(train))
    out = det.predict(sample_df)
    out, _ = det.interpret(out)
    labels = set(out["market_regime_label"].dropna().unique())
    assert labels <= {"Bull", "Bear", "Sideways", "High Volatility"}
    assert len(labels) >= 2


def test_smoothing_reduces_switches(sample_df, config_path):
    det = RegimeDetector(config_path)
    train, _ = det.split_train_test(sample_df)
    det.fit_scaler(train)
    det.fit(det.transform(train))

    det.config["smoothing_window"] = 1
    raw = det.predict(sample_df)
    det.config["smoothing_window"] = 5
    smooth = det.predict(sample_df)

    raw_switches = (raw.groupby("Ticker")["market_regime"]
                    .apply(lambda s: (s.diff() != 0).sum()).sum())
    smooth_switches = (smooth.groupby("Ticker")["market_regime"]
                       .apply(lambda s: (s.diff() != 0).sum()).sum())
    assert smooth_switches <= raw_switches
