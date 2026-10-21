"""Объединение прогнозов нескольких LLM в единый сигнал."""

import math

VALID_PREDICTIONS = ("UP", "DOWN", "HOLD")

DIRECTION = {
    "UP": 1.0,
    "DOWN": -1.0,
    "HOLD": 0.0,
}

MAJORITY_VOTING = "majority_voting"
CONFIDENCE_WEIGHTED_VOTING = "confidence_weighted_voting"
DEFAULT_METHOD = CONFIDENCE_WEIGHTED_VOTING


def _normalise_responses(responses):
    """Проверяет ответы LLM и приводит их к единому виду."""

    if not isinstance(responses, (list, tuple)):
        raise TypeError("responses должен быть списком ответов моделей.")

    if not responses:
        raise ValueError("Список ответов моделей пуст.")

    normalised = []

    for index, response in enumerate(responses):
        if not isinstance(response, dict):
            raise TypeError(f"Ответ модели №{index} должен быть словарём.")

        model = response.get("model")
        prediction = response.get("prediction")
        confidence = response.get("confidence")

        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"В ответе модели №{index} поле 'model' должно быть непустой строкой."
            )

        if not isinstance(prediction, str):
            raise ValueError(
                f"В ответе модели №{index} поле 'prediction' должно быть строкой."
            )

        prediction = prediction.upper().strip()

        if prediction not in VALID_PREDICTIONS:
            raise ValueError(
                f"В ответе модели №{index} недопустимый prediction: "
                f"{prediction!r}. Допустимо: {VALID_PREDICTIONS}."
            )

        if isinstance(confidence, bool) or not isinstance(
            confidence, (int, float)
        ):
            raise ValueError(
                f"В ответе модели №{index} поле 'confidence' должно быть числом."
            )

        confidence = float(confidence)

        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"В ответе модели №{index} confidence должен быть в диапазоне "
                f"[0, 1], получено: {confidence!r}."
            )

        normalised.append(
            {
                "model": model.strip(),
                "prediction": prediction,
                "confidence": confidence,
            }
        )

    return normalised


def _count_votes(responses):
    """Считает количество голосов каждой модели за каждый класс."""

    counts = {label: 0 for label in VALID_PREDICTIONS}

    for response in responses:
        counts[response["prediction"]] += 1

    return counts


def _select_winner(scores, vote_counts):
    """Выбирает класс с максимальным score; при ничьей использует голоса."""

    max_score = max(scores.values())
    winners = [
        label
        for label, score in scores.items()
        if math.isclose(score, max_score, rel_tol=1e-9, abs_tol=1e-12)
    ]

    if len(winners) == 1:
        return winners[0]

    max_votes = max(vote_counts[label] for label in winners)
    vote_winners = [
        label for label in winners if vote_counts[label] == max_votes
    ]

    if len(vote_winners) == 1:
        return vote_winners[0]

    return "HOLD"


def majority_voting(responses):
    """Объединяет прогнозы простым большинством голосов."""

    responses = _normalise_responses(responses)
    vote_counts = _count_votes(responses)

    prediction = _select_winner(
        {label: float(count) for label, count in vote_counts.items()},
        vote_counts,
    )

    return {
        "prediction": prediction,
        "score": vote_counts[prediction] / len(responses),
        "method": MAJORITY_VOTING,
        "vote_counts": vote_counts,
    }


def confidence_weighted_voting(responses):
    """Объединяет прогнозы с учётом уверенности каждой модели.

    Score — направленный баланс уверенности в диапазоне [-1, 1]:
    положительное значение поддерживает UP, отрицательное — DOWN.
    Итоговый класс выбирается по максимальной сумме confidence.
    """

    responses = _normalise_responses(responses)
    vote_counts = _count_votes(responses)

    weighted_scores = {label: 0.0 for label in VALID_PREDICTIONS}

    for response in responses:
        weighted_scores[response["prediction"]] += response["confidence"]

    prediction = _select_winner(weighted_scores, vote_counts)

    score = (
        weighted_scores["UP"] - weighted_scores["DOWN"]
    ) / len(responses)

    return {
        "prediction": prediction,
        "score": score,
        "method": CONFIDENCE_WEIGHTED_VOTING,
        "weighted_scores": weighted_scores,
        "vote_counts": vote_counts,
    }


def make_ensemble_decision(
    responses,
    method=DEFAULT_METHOD,
):
    """Возвращает итоговое решение ансамбля по ответам нескольких LLM.

    Поддерживаемые методы:
    - "confidence_weighted_voting" — взвешивание по confidence (по умолчанию);
    - "majority_voting" — простое большинство голосов.
    """

    if not isinstance(method, str):
        raise TypeError("method должен быть строкой.")

    method = method.lower().strip()

    if method in {
        CONFIDENCE_WEIGHTED_VOTING,
        "confidence_weighted",
        "weighted",
    }:
        return confidence_weighted_voting(responses)

    if method in {MAJORITY_VOTING, "majority"}:
        return majority_voting(responses)

    raise ValueError(
        f"Неизвестный метод ensemble: {method!r}. "
        f"Допустимо: {MAJORITY_VOTING!r}, {CONFIDENCE_WEIGHTED_VOTING!r}."
    )


if __name__ == "__main__":
    sample_responses = [
        {"model": "gpt", "prediction": "UP", "confidence": 0.80},
        {"model": "gemini", "prediction": "UP", "confidence": 0.70},
        {"model": "deepseek", "prediction": "DOWN", "confidence": 0.60},
        {"model": "llama", "prediction": "UP", "confidence": 0.75},
    ]

    for method in (MAJORITY_VOTING, CONFIDENCE_WEIGHTED_VOTING):
        result = make_ensemble_decision(sample_responses, method=method)
        print(f"\n{method}:")
        print(f"  Final prediction: {result['prediction']}")
        print(f"  Score: {result['score']:.4f}")
