
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


def calculate_metrics(actual, predicted):
    """
    Рассчитывает метрики качества предсказаний.

    actual — реальные направления движения рынка.
    predicted — предсказанные направления: UP, DOWN или HOLD.
    """

    allowed_labels = {"UP", "DOWN", "HOLD"}

    actual = [label.upper() for label in actual]
    predicted = [label.upper() for label in predicted]

    if len(actual) != len(predicted):
        raise ValueError(
            "Количество реальных и предсказанных значений должно совпадать."
        )

    if not actual:
        raise ValueError("Списки actual и predicted не должны быть пустыми.")

    if not all(label in allowed_labels for label in actual + predicted):
        raise ValueError("Допустимые метки: UP, DOWN, HOLD.")

    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, labels=["UP", "DOWN", "HOLD"],
            average="macro", zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, labels=["UP", "DOWN", "HOLD"],
            average="macro", zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, labels=["UP", "DOWN", "HOLD"],
            average="macro", zero_division=0
        ),
        "mcc": matthews_corrcoef(actual, predicted),
    }


if __name__ == "__main__":
    actual = ["UP", "DOWN", "UP", "HOLD", "DOWN"]
    predicted = ["UP", "UP", "UP", "HOLD", "DOWN"]

    results = calculate_metrics(actual, predicted)

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
