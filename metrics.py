"""Роль 6 — Классификационные метрики прогнозов.

Классификационные и торговые метрики считаются раздельно:
- этот модуль оценивает качество прогноза направления (UP/DOWN/HOLD);
- модуль backtest.py оценивает торговый результат.

Планируемые метрики: Accuracy, Balanced Accuracy, Precision, Recall, F1,
MCC, confusion matrix, Brier Score и Log Loss (при наличии вероятностей).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

BINARY_LABELS = ["UP", "DOWN"]


def _normalise_labels(values):
    return [label.strip().upper() for label in values]


def calculate_metrics(actual, predicted, probabilities=None):
    """Рассчитывает метрики качества предсказаний.

    actual     — реальные направления движения рынка (UP/DOWN/HOLD).
    predicted  — предсказанные направления (UP/DOWN/HOLD).
    probabilities — опционально: probability_up для Brier/LogLoss.

    Для вероятностных метрик сравнение бинарное: UP=1 против остального.
    """

    allowed_labels = {"UP", "DOWN", "HOLD"}
    labels_list = ["UP", "DOWN", "HOLD"]

    actual = _normalise_labels(actual)
    predicted = _normalise_labels(predicted)

    if len(actual) != len(predicted):
        raise ValueError(
            "Количество реальных и предсказанных значений должно совпадать."
        )

    if not actual:
        raise ValueError("Списки actual и predicted не должны быть пустыми.")

    if not all(label in allowed_labels for label in actual + predicted):
        raise ValueError("Допустимые метки: UP, DOWN, HOLD.")

    results = {
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, labels=labels_list,
            average="macro", zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, labels=labels_list,
            average="macro", zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, labels=labels_list,
            average="macro", zero_division=0
        ),
        "mcc": matthews_corrcoef(actual, predicted),
    }

    cm = confusion_matrix(actual, predicted, labels=labels_list)
    results["confusion_matrix"] = {
        f"pred_{pred_col}|actual_{row_label.lower()}": int(cm[row, col])
        for row, row_label in enumerate(labels_list)
        for col, pred_col in enumerate(labels_list)
    }

    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        if len(probabilities) != len(actual):
            raise ValueError(
                "Количество вероятностей не совпадает с числом прогнозов."
            )
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("Вероятности должны быть в диапазоне [0, 1].")

        binary_actual = [1 if label == "UP" else 0 for label in actual]

        results["brier_score"] = float(
            np.mean((probabilities - np.array(binary_actual)) ** 2)
        )

        unique_classes = set(binary_actual)
        if len(unique_classes) == 2:
            results["log_loss"] = float(
                log_loss(binary_actual, np.column_stack(
                    [1 - probabilities, probabilities]
                ))
            )
        else:
            results["log_loss"] = float("nan")

    return results


def calculate_metrics_binary(actual, predicted):
    """Бинарные метрики UP против DOWN (обработано для E1 baseline)."""

    actual = _normalise_labels(actual)
    predicted = _normalise_labels(predicted)

    valid = [
        i for i, (a, p) in enumerate(zip(actual, predicted))
        if a in BINARY_LABELS and p in BINARY_LABELS
    ]

    if not valid:
        raise ValueError("Нет ни одной бинарной пары UP/DOWN.")

    actual_bin = [1 if actual[i] == "UP" else 0 for i in valid]
    predicted_bin = [1 if predicted[i] == "UP" else 0 for i in valid]

    return {
        "n_samples": len(valid),
        "accuracy": accuracy_score(actual_bin, predicted_bin),
        "precision": precision_score(
            actual_bin, predicted_bin, zero_division=0
        ),
        "recall": recall_score(
            actual_bin, predicted_bin, zero_division=0
        ),
        "f1": f1_score(
            actual_bin, predicted_bin, zero_division=0
        ),
        "mcc": matthews_corrcoef(actual_bin, predicted_bin),
    }


if __name__ == "__main__":
    actual = ["UP", "DOWN", "UP", "HOLD", "DOWN"]
    predicted = ["UP", "UP", "UP", "HOLD", "DOWN"]
    probabilities = [0.7, 0.4, 0.65, 0.5, 0.3]

    results = calculate_metrics(
        actual, predicted, probabilities=probabilities
    )

    for metric, value in results.items():
        print(f"{metric}: {value}")
