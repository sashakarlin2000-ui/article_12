import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from metrics import calculate_metrics


def load_data(market_path="market_data.csv",
              predictions_path="predictions.csv"):
    """Загружает и проверяет рыночные данные и предсказания."""

    market = pd.read_csv(market_path)
    predictions = pd.read_csv(predictions_path)

    required_market = {"Date", "Ticker", "Close"}
    required_predictions = {"Date", "Ticker", "Prediction"}

    if not required_market.issubset(market.columns):
        raise ValueError(
            f"В market_data.csv нужны столбцы: {required_market}"
        )

    if not required_predictions.issubset(predictions.columns):
        raise ValueError(
            f"В predictions.csv нужны столбцы: {required_predictions}"
        )

    market["Date"] = pd.to_datetime(market["Date"])
    predictions["Date"] = pd.to_datetime(predictions["Date"])

    market["Ticker"] = market["Ticker"].str.upper().str.strip()
    predictions["Ticker"] = predictions["Ticker"].str.upper().str.strip()
    predictions["Prediction"] = (
        predictions["Prediction"].str.upper().str.strip()
    )

    allowed = {"UP", "DOWN", "HOLD"}
    if not predictions["Prediction"].isin(allowed).all():
        raise ValueError("Prediction должен быть UP, DOWN или HOLD.")

    if market.duplicated(["Date", "Ticker"]).any():
        raise ValueError("В рыночных данных есть дубликаты Date + Ticker.")

    if predictions.duplicated(["Date", "Ticker"]).any():
        raise ValueError("В predictions.csv есть дубликаты Date + Ticker.")

    market = market.sort_values(["Ticker", "Date"]).copy()

    # Доходность от текущего закрытия до следующего закрытия
    market["Next_Close"] = market.groupby("Ticker")["Close"].shift(-1)
    market["Future_Return"] = (
        market["Next_Close"] / market["Close"] - 1
    )

    # Последняя строка каждого тикера не имеет будущей доходности
    market = market.dropna(subset=["Future_Return"])

    # Соединяем данные по дате и тикеру
    data = predictions.merge(
        market[["Date", "Ticker", "Future_Return", "Close", "Next_Close"]],
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = data["_merge"] != "both"
    if unmatched.any():
        examples = data.loc[unmatched, ["Date", "Ticker"]].head()
        raise ValueError(
            "Для некоторых предсказаний нет соответствующих рыночных данных. "
            f"Примеры:\n{examples.to_string(index=False)}"
        )

    data = data.drop(columns="_merge")
    return data


def calculate_trading_results(data):
    """Считает доходность стратегии и Buy & Hold."""

    data = data.sort_values(["Ticker", "Date"]).copy()

    # UP = Long, DOWN = Short, HOLD = нет позиции
    position_map = {"UP": 1, "DOWN": -1, "HOLD": 0}
    data["Position"] = data["Prediction"].map(position_map)

    # Доходность сигнала за следующий торговый день
    data["Strategy_Return"] = (
        data["Position"] * data["Future_Return"]
    )

    # Сравнение с Buy & Hold на тех же датах
    data["Buy_Hold_Return"] = data["Future_Return"]

    # Отдельная кривая капитала для каждого тикера.
    # Кумулятивный рост: (1+r1)*(1+r2)*...*(1+rn)
    data["Strategy_Equity"] = (
        data.groupby("Ticker")["Strategy_Return"]
        .transform(lambda returns: (1 + returns).cumprod())
    )

    data["Buy_Hold_Equity"] = (
        data.groupby("Ticker")["Buy_Hold_Return"]
        .transform(lambda returns: (1 + returns).cumprod())
    )

    return data


def calculate_financial_metrics(returns):
    """Рассчитывает доходность, Sharpe, просадку и долю прибыльных дней."""

    returns = pd.Series(returns, dtype=float).dropna()

    if returns.empty:
        raise ValueError("Недостаточно данных для финансовых метрик.")

    equity = (1 + returns).cumprod()
    cumulative_return = equity.iloc[-1] - 1

    # Предполагаем 252 торговых дня в году и нулевую безрисковую ставку
    std = returns.std(ddof=1)
    if pd.isna(std) or std == 0:
        sharpe = 0.0
    else:
        sharpe = np.sqrt(252) * returns.mean() / std

    peak = equity.cummax()
    drawdown = equity / peak - 1
    max_drawdown = drawdown.min()

    win_rate = (returns > 0).mean()

    return {
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


def plot_equity_curve(data, output_path="equity_curve.png"):
    """Строит кривые капитала для каждого тикера."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for ticker, group in data.groupby("Ticker"):
        axes[0].plot(
            group["Date"], group["Strategy_Equity"], label=ticker
        )
        axes[1].plot(
            group["Date"], group["Buy_Hold_Equity"], label=ticker
        )

    axes[0].set_title("Strategy Equity Curve")
    axes[1].set_title("Buy & Hold Equity Curve")

    for ax in axes:
        ax.set_ylabel("Growth of $1")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    data = load_data()
    data = calculate_trading_results(data)

    print("\nФинансовые метрики стратегии по тикеру:")
    for ticker, group in data.groupby("Ticker"):
        metrics = calculate_financial_metrics(group["Strategy_Return"])
        print(f"\n{ticker}:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

    # Классификационные метрики рассчитываем только если есть
    # реальные метки Actual в отдельном источнике.
    # Пока в predictions.csv есть только Prediction.
    plot_equity_curve(data)

    print("\nГрафик сохранён в equity_curve.png")


if __name__ == "__main__":
    main()
