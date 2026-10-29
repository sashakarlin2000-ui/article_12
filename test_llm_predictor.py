"""Тесты Роли 4 — LLM: парсинг, форматы, кэш, обработка ошибок."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_predictor import (
    DEFAULT_PROMPT_TEMPLATE,
    LLMOutputError,
    MockLLMProvider,
    build_llm_payload,
    build_prompt,
    parse_llm_output,
    predict_for_day,
)


def test_build_payload_format():
    payload = build_llm_payload(
        ticker="AAPL",
        date="2024-03-10",
        features={"return_5d": 0.02, "rsi": 58.4},
        market_regime="Bear",
    )
    assert set(payload) == {
        "ticker", "date", "features", "market_regime", "similar_days"
    }
    assert payload["date"] == "2024-03-10"
    assert payload["similar_days"] == []


def test_prompt_contains_all_sections():
    payload = build_llm_payload(
        ticker="SPX",
        date="2024-03-10",
        features={"return_5d": 0.01},
        market_regime="Bull",
        similar_days=[{"date": "2020-03-12", "forward_return": -0.021}],
    )
    prompt = build_prompt(payload)
    for fragment in ("Ticker: SPX", "Date: 2024-03-10",
                     "Market regime: Bull", "return_5d",
                     "2020-03-12"):
        assert fragment in prompt


@pytest.mark.parametrize("raw", [
    json.dumps({"prediction": "UP", "confidence": 0.8,
                "probability_up": 0.64, "explanation": "ok"}),
    '```json\n{"prediction": "DOWN", "confidence": 0.7, '
    '"probability_up": 0.2, "explanation": "x"}\n```',
    'Noise {"prediction": "HOLD", "confidence": 1, '
    '"probability_up": 0.5, "explanation": ""} noise',
])
def test_parse_valid_outputs(raw):
    parsed = parse_llm_output(raw)
    assert parsed["prediction"] in {"UP", "DOWN", "HOLD"}
    assert 0.0 <= parsed["confidence"] <= 1.0
    assert 0.0 <= parsed["probability_up"] <= 1.0


@pytest.mark.parametrize("raw", [
    "",
    "not a json at all",
    '{"prediction": "MAYBE", "confidence": 0.5, "probability_up": 0.5}',
    '{"confidence": 0.5, "probability_up": 0.5}',
    '{"prediction": "UP", "confidence": 1.5, "probability_up": 0.5}',
    '{"prediction": "UP", "confidence": "high", "probability_up": 0.5}',
])
def test_parse_invalid_outputs_raise(raw):
    with pytest.raises(LLMOutputError):
        parse_llm_output(raw)


def test_mock_provider_deterministic_up():
    provider = MockLLMProvider()
    payload = build_llm_payload(
        ticker="SPX",
        date="2024-03-10",
        features={"return_5d": 0.05},
        market_regime="Bull",
        similar_days=[{"date": "2020-03-12", "similarity": 0.9}],
    )
    out = predict_for_day(
        payload, provider, cache_dir=Path("llm_cache_test"),
        log_dir=Path("llm_cache_test"),
    )
    assert out["prediction"] == "UP"
    assert out["confidence"] > 0.5


def test_mock_provider_down_and_hold():
    provider = MockLLMProvider()
    down_payload = build_llm_payload(
        ticker="SPX", date="2024-03-10", features={"return_5d": -0.05},
        market_regime="Bear",
    )
    hold_payload = build_llm_payload(
        ticker="SPX", date="2024-03-11", features={"return_5d": 0.001},
        market_regime=None,
    )
    down = predict_for_day(
        down_payload, provider, cache_dir=Path("llm_cache_test"),
        log_dir=Path("llm_cache_test"),
    )
    hold = predict_for_day(
        hold_payload, provider, cache_dir=Path("llm_cache_test"),
        log_dir=Path("llm_cache_test"),
    )
    assert down["prediction"] == "DOWN"
    assert hold["prediction"] == "HOLD"


def test_mock_provider_does_not_use_forward_returns():
    """Аналоги с будущим исходом не должны менять прогноз заглушки."""
    base = build_llm_payload(
        ticker="SPX", date="2024-03-10", features={"return_5d": -0.05},
        market_regime="Bear",
    )
    with_leak = base.copy()
    with_leak["similar_days"] = [
        {"date": "2020-03-12", "forward_return": 0.10},
        {"date": "2022-06-13", "forward_return": 0.08},
    ]
    clean = base.copy()
    clean["similar_days"] = [
        {"date": "2020-03-12", "similarity": 0.9},
        {"date": "2022-06-13", "similarity": 0.9},
    ]
    provider = MockLLMProvider()
    a = predict_for_day(
        with_leak, provider, cache_dir=Path("llm_cache_test"),
        log_dir=Path("llm_cache_test"),
    )
    b = predict_for_day(
        clean, provider, cache_dir=Path("llm_cache_test"),
        log_dir=Path("llm_cache_test"),
    )
    assert a["prediction"] == b["prediction"] == "DOWN"


def test_cache_reuse(tmp_path):
    provider = MockLLMProvider()
    payload = build_llm_payload(
        ticker="AAPL", date="2024-01-05", features={"return_5d": 0.02},
        market_regime="Bull", similar_days=[{"date": "2020-03-12"}],
    )
    first = predict_for_day(
        payload, provider, cache_dir=tmp_path / "c",
        log_dir=tmp_path / "l",
    )
    second = predict_for_day(
        payload, provider, cache_dir=tmp_path / "c",
        log_dir=tmp_path / "l",
    )
    assert first == second

    log_lines = (tmp_path / "l" / "llm_prompts_log.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert json.loads(log_lines[-1])["cached"] is True


def test_invalid_then_recovery_logged():
    class BrokenThenGoodProvider(MockLLMProvider):
        def __init__(self):
            super().__init__(model_name="flaky")
            self.calls = 0

        def complete(self, prompt, temperature=0.2, max_retries=2):
            self.calls += 1
            if self.calls == 1:
                return ":) garbage :("
            return super().complete(prompt, temperature, max_retries)

    provider = BrokenThenGoodProvider()
    payload = build_llm_payload(
        ticker="AAPL", date="2024-01-05", features={"return_5d": 0.02},
        market_regime="Bull", similar_days=[{"date": "2020-03-12"}],
    )
    out = predict_for_day(
        payload, provider, cache_dir=Path("llm_cache_test"),
        log_dir=Path("llm_cache_test"),
        use_cache=False,
    )
    assert out["prediction"] == "UP"

    log_dir = Path("llm_cache_test")
    log_file = log_dir / "llm_prompts_log.jsonl"
    entry = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["status"] == "ok"
    assert "raw_response" in entry


def test_permanently_invalid_raises():
    class GarbageProvider(MockLLMProvider):
        def complete(self, prompt, temperature=0.2, max_retries=2):
            return "garbage"

    provider = GarbageProvider()
    payload = build_llm_payload(
        ticker="AAPL", date="2024-01-05", features={}, market_regime=None,
        similar_days=[],
    )
    with pytest.raises(LLMOutputError):
        predict_for_day(
            payload, provider, cache_dir=Path("llm_cache_test"),
            log_dir=Path("llm_cache_test"), use_cache=False,
            max_retries=1,
        )
