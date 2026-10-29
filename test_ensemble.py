"""Тесты для механизма объединения прогнозов LLM."""

import math
import unittest

from ensemble import (
    CONFIDENCE_WEIGHTED_VOTING,
    MAJORITY_VOTING,
    confidence_weighted_voting,
    majority_voting,
    make_ensemble_decision,
)


SAMPLE_RESPONSES = [
    {"model": "gpt", "prediction": "UP", "confidence": 0.80},
    {"model": "gemini", "prediction": "UP", "confidence": 0.70},
    {"model": "deepseek", "prediction": "DOWN", "confidence": 0.60},
    {"model": "llama", "prediction": "UP", "confidence": 0.75},
]


class TestMajorityVoting(unittest.TestCase):
    def test_sample_returns_up(self):
        result = majority_voting(SAMPLE_RESPONSES)

        self.assertEqual(result["prediction"], "UP")
        self.assertEqual(result["method"], MAJORITY_VOTING)
        self.assertEqual(result["vote_counts"], {"UP": 3, "DOWN": 1, "HOLD": 0})

    def test_score_is_share_of_votes(self):
        result = majority_voting(SAMPLE_RESPONSES)

        self.assertAlmostEqual(result["score"], 0.75)

    def test_tie_between_up_and_down_returns_hold(self):
        responses = [
            {"model": "gpt", "prediction": "UP", "confidence": 0.80},
            {"model": "gemini", "prediction": "UP", "confidence": 0.70},
            {"model": "deepseek", "prediction": "DOWN", "confidence": 0.60},
            {"model": "llama", "prediction": "DOWN", "confidence": 0.75},
        ]

        result = majority_voting(responses)

        self.assertEqual(result["prediction"], "HOLD")
        self.assertAlmostEqual(result["score"], 0.0)

    def test_hold_can_win_majority(self):
        responses = [
            {"model": "gpt", "prediction": "HOLD", "confidence": 0.80},
            {"model": "gemini", "prediction": "HOLD", "confidence": 0.70},
            {"model": "deepseek", "prediction": "DOWN", "confidence": 0.60},
        ]

        result = majority_voting(responses)

        self.assertEqual(result["prediction"], "HOLD")
        self.assertAlmostEqual(result["score"], 2 / 3)

    def test_single_model(self):
        result = majority_voting(
            [{"model": "gpt", "prediction": "DOWN", "confidence": 0.55}]
        )

        self.assertEqual(result["prediction"], "DOWN")
        self.assertAlmostEqual(result["score"], 1.0)


class TestConfidenceWeightedVoting(unittest.TestCase):
    def test_sample_returns_up_with_expected_score(self):
        result = confidence_weighted_voting(SAMPLE_RESPONSES)

        self.assertEqual(result["prediction"], "UP")
        self.assertEqual(result["method"], CONFIDENCE_WEIGHTED_VOTING)
        self.assertAlmostEqual(result["score"], 0.4125)
        self.assertEqual(
            result["weighted_scores"],
            {"UP": 2.25, "DOWN": 0.60, "HOLD": 0.0},
        )

    def test_confidence_can_override_majority(self):
        responses = [
            {"model": "gpt", "prediction": "UP", "confidence": 0.40},
            {"model": "gemini", "prediction": "UP", "confidence": 0.30},
            {"model": "deepseek", "prediction": "DOWN", "confidence": 0.95},
        ]

        result = confidence_weighted_voting(responses)

        self.assertEqual(result["prediction"], "DOWN")
        self.assertLess(result["score"], 0.0)
        self.assertAlmostEqual(result["score"], (0.70 - 0.95) / 3)

    def test_weighted_tie_uses_majority_vote(self):
        responses = [
            {"model": "gpt", "prediction": "UP", "confidence": 0.60},
            {"model": "gemini", "prediction": "UP", "confidence": 0.20},
            {"model": "deepseek", "prediction": "DOWN", "confidence": 0.80},
        ]

        result = confidence_weighted_voting(responses)

        self.assertEqual(result["weighted_scores"]["UP"], 0.80)
        self.assertEqual(result["weighted_scores"]["DOWN"], 0.80)
        self.assertEqual(result["prediction"], "UP")
        self.assertAlmostEqual(result["score"], 0.0)

    def test_full_tie_returns_hold(self):
        responses = [
            {"model": "gpt", "prediction": "UP", "confidence": 0.80},
            {"model": "gemini", "prediction": "DOWN", "confidence": 0.80},
        ]

        result = confidence_weighted_voting(responses)

        self.assertEqual(result["prediction"], "HOLD")
        self.assertAlmostEqual(result["score"], 0.0)

    def test_hold_with_highest_confidence_wins(self):
        responses = [
            {"model": "gpt", "prediction": "UP", "confidence": 0.50},
            {"model": "gemini", "prediction": "DOWN", "confidence": 0.40},
            {"model": "deepseek", "prediction": "HOLD", "confidence": 0.90},
        ]

        result = confidence_weighted_voting(responses)

        self.assertEqual(result["prediction"], "HOLD")
        self.assertAlmostEqual(result["score"], 0.10 / 3)

    def test_all_hold_returns_zero_score(self):
        responses = [
            {"model": "gpt", "prediction": "HOLD", "confidence": 0.80},
            {"model": "gemini", "prediction": "HOLD", "confidence": 0.20},
        ]

        result = confidence_weighted_voting(responses)

        self.assertEqual(result["prediction"], "HOLD")
        self.assertAlmostEqual(result["score"], 0.0)

    def test_score_is_within_bounds(self):
        responses = [
            {"model": "gpt", "prediction": "UP", "confidence": 1.0},
            {"model": "gemini", "prediction": "UP", "confidence": 1.0},
        ]

        result = confidence_weighted_voting(responses)

        self.assertLessEqual(abs(result["score"]), 1.0)
        self.assertTrue(math.isfinite(result["score"]))


class TestMakeEnsembleDecision(unittest.TestCase):
    def test_default_method_is_confidence_weighted(self):
        result = make_ensemble_decision(SAMPLE_RESPONSES)

        self.assertEqual(result["method"], CONFIDENCE_WEIGHTED_VOTING)
        self.assertAlmostEqual(result["score"], 0.4125)

    def test_majority_method(self):
        result = make_ensemble_decision(
            SAMPLE_RESPONSES, method=MAJORITY_VOTING
        )

        self.assertEqual(result["method"], MAJORITY_VOTING)
        self.assertEqual(result["prediction"], "UP")

    def test_method_aliases(self):
        self.assertEqual(
            make_ensemble_decision(SAMPLE_RESPONSES, method="majority")[
                "method"
            ],
            MAJORITY_VOTING,
        )
        self.assertEqual(
            make_ensemble_decision(SAMPLE_RESPONSES, method="weighted")[
                "method"
            ],
            CONFIDENCE_WEIGHTED_VOTING,
        )

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            make_ensemble_decision(SAMPLE_RESPONSES, method="unknown")

    def test_empty_responses_raise(self):
        for method in (MAJORITY_VOTING, CONFIDENCE_WEIGHTED_VOTING):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    make_ensemble_decision([], method=method)

    def test_non_list_responses_raise(self):
        with self.assertRaises(TypeError):
            make_ensemble_decision({"model": "gpt"})

    def test_non_dict_response_raises(self):
        with self.assertRaises(TypeError):
            make_ensemble_decision(["UP"])

    def test_invalid_prediction_raises(self):
        responses = [
            {"model": "gpt", "prediction": "MAYBE", "confidence": 0.80}
        ]

        with self.assertRaises(ValueError):
            make_ensemble_decision(responses)

    def test_confidence_out_of_range_raises(self):
        for confidence in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(confidence=confidence):
                responses = [
                    {
                        "model": "gpt",
                        "prediction": "UP",
                        "confidence": confidence,
                    }
                ]
                with self.assertRaises(ValueError):
                    make_ensemble_decision(responses)

    def test_boolean_confidence_raises(self):
        responses = [
            {"model": "gpt", "prediction": "UP", "confidence": True}
        ]

        with self.assertRaises(ValueError):
            make_ensemble_decision(responses)

    def test_missing_or_empty_fields_raise(self):
        invalid_responses = [
            {"prediction": "UP", "confidence": 0.8},
            {"model": "  ", "prediction": "UP", "confidence": 0.8},
            {"model": "gpt", "confidence": 0.8},
            {"model": "gpt", "prediction": "UP"},
        ]

        for response in invalid_responses:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    make_ensemble_decision([response])

    def test_lowercase_and_spaces_are_normalised(self):
        responses = [
            {"model": " GPT ", "prediction": " up ", "confidence": 0.8}
        ]

        result = make_ensemble_decision(responses)

        self.assertEqual(result["prediction"], "UP")


if __name__ == "__main__":
    unittest.main()

# --- Режимно-зависимые веса (Роль 5) ---
import pytest as _pytest

from ensemble import (
    REGIME_WEIGHTED_VOTING,
    default_regime_weights,
    make_ensemble_decision,
    normalise_regime_weights,
    regime_weighted_voting,
)


def test_default_regime_weights_structure():
    models = ["gpt", "gemini"]
    table = default_regime_weights(models)
    assert "default" in table
    for regime in ("Bull", "Bear", "Sideways", "High Volatility"):
        assert table[regime] == {"gpt": 1.0, "gemini": 1.0}


def test_regime_weighted_respects_weights():
    responses = [
        {"model": "gpt", "prediction": "UP", "confidence": 0.9},
        {"model": "deepseek", "prediction": "DOWN", "confidence": 0.9},
    ]
    weights = {
        "Bull": {"gpt": 3.0, "deepseek": 0.1},
        "Bear": {"gpt": 0.1, "deepseek": 3.0},
        "default": {"gpt": 1.0, "deepseek": 1.0},
    }
    bull = regime_weighted_voting(responses, weights, regime="Bull")
    bear = regime_weighted_voting(responses, weights, regime="Bear")
    assert bull["prediction"] == "UP"
    assert bear["prediction"] == "DOWN"


def test_regime_fallback_to_default():
    responses = [{"model": "gpt", "prediction": "UP", "confidence": 0.9}]
    weights = {
        "default": {"gpt": 2.0},
        "Bull": {"gpt": 1.0},
    }
    result = regime_weighted_voting(responses, weights, regime="Crash")
    assert result["weights_regime_used"] == "default"
    assert result["prediction"] == "UP"


def test_unknown_model_gets_weight_one():
    responses = [{"model": "new_model", "prediction": "HOLD", "confidence": 0.5}]
    weights = default_regime_weights(["gpt"])
    result = regime_weighted_voting(responses, weights, regime="Bull")
    assert result["prediction"] == "HOLD"
    assert result["applied_weights"][0]["weight"] == 1.0


def test_regime_method_without_weights_raises():
    responses = [{"model": "gpt", "prediction": "UP", "confidence": 0.9}]
    with _pytest.raises(ValueError):
        make_ensemble_decision(responses, method=REGIME_WEIGHTED_VOTING)


def test_normalise_raises_on_missing_default():
    with _pytest.raises(ValueError):
        normalise_regime_weights({"Bull": {"gpt": 1.0}})


def test_normalise_raises_on_negative_weight():
    with _pytest.raises(ValueError):
        normalise_regime_weights(
            {"default": {"gpt": -1.0}}
        )


def test_weights_not_affected_by_confidence_not_breaking():
    responses = [
        {"model": "gpt", "prediction": "UP", "confidence": 0.2},
        {"model": "gpt", "prediction": "DOWN", "confidence": 0.9},
    ]
    weights = {
        "default": {"gpt": 10.0},
        "Bull": {"gpt": 10.0},
    }
    result = regime_weighted_voting(responses, weights, regime="Bull")
    assert result["prediction"] == "DOWN"
