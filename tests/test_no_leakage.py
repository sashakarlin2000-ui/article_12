"""Тесты на отсутствие утечки будущей информации."""
import pandas as pd
import pytest

from regime_detector import RegimeDetector
from regime_utils import validate_no_leakage


def test_train_test_split(sample_df, config_path):
    det = RegimeDetector(config_path)
    train, test = det.split_train_test(sample_df)
    assert pd.to_datetime(train["Date"]).max() < pd.to_datetime(test["Date"]).min()


def test_scaler_fit_on_train_only(sample_df, config_path):
    det = RegimeDetector(config_path)
    train, _ = det.split_train_test(sample_df)
    det.fit_scaler(train)
    expected_mean = train[det.features].mean().values
    assert det.scaler.mean_ == pytest.approx(expected_mean)


def test_validate_no_leakage_raises_on_overlap(sample_df, config_path):
    det = RegimeDetector(config_path)
    train, test = det.split_train_test(sample_df)
    # искусственно сдвигаем test в прошлое — должна быть ошибка
    bad_test = test.copy()
    bad_test["Date"] = pd.to_datetime(bad_test["Date"]) - pd.Timedelta(days=400)
    with pytest.raises(ValueError):
        validate_no_leakage(train, bad_test)
