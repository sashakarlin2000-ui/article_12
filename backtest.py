"""Роль 6 — Backtest и оценка торговых результатов.

Торговые правила (зафиксированы заранее, как требует протокол):
- сигнал формируется по закрытию дня T, исполнение — по цене закрытия T,
 доходность считается от закрытия T к закрытию T+1;
- UP   → Long (+1),
- DOWN → Short (-1) в варианте long_short, 0 в варианте long_only,
- HOLD → нет позиции (0);
- комиссия и проскальзывание списываются при изменении позиции:
  |Δposition| * (commission + slippage);
- размер позиции — 100% капитала;
- сравнение с Buy & Hold на тех же датах.

Классификационные метрики считаются в metrics.py, торговые — здесь.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from metrics import calculate_metrics

DEFAULT_COSTS = {"commission": 0.001, "slippage": 0.0005}


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

    # Actual вычисляется по фактическому движению следующего дня —
    # это реальный исход, доступный только при оценке, не при прогнозе.
    data["Actual"] = np.where(
        data["Future_Return"] > 0, "UP", "DOWN"
    )

    return data


def calculate_trading_results(
    data,
    allow_short=True,
    commission=DEFAULT_COSTS["commission"],
    slippage=DEFAULT_COSTS["slippage"],
):
    """Считает доходность стратегии (long/short) и Buy & Hold.

    allow_short=False включает long-only вариант (DOWN трактуется как выход
    в кэш, позиция = 0).
    """

    if not allow_short:
        data = data.copy()

    data = data.sort_values(["Ticker", "Date"]).copy()

    if allow_short:
       position_map = {"UP": 1, "DOWN": -1, "HOLD": 0}
    else:
        position_map = {"UP": 1, "DOWN": 0, "HOLD": 0}

    data["Position"] = data["Prediction"].map(position_map)

    # Изменение позиции определяют издержки
    data["Position_Change"] = (
        data.groupby("Ticker")["Position"].diff().abs().fillna(
            data["Position"].abs()
        )
    )
    data["Cost"] = data["Position_Change"] * (commission + slippage)

    # Доходность сигнала за следующий торговый день (после издержек)
    data["Strategy_Return_Gross"] = (
        data["Position"] * data["Future_Return"]
    )
    data["Strategy_Return"] = data["Strategy_Return_Gross"] - data["Cost"]

    data["Turnover"] = data["Position_Change"]
    data["Commission_Paid"] = data["Position_Change"] * commission
    data["Slippage_Paid"] = data["Position_Change"] * slippage

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


def calculate_financial_metrics(returns, n_days=252):
    """Доходность, Sharpe, просадка, win rate, число периодов."""

    returns = pd.Series(returns, dtype=float).dropna()

    if returns.empty:
        raise ValueError("Недостаточно данных для финансовых метрик.")

    equity = (1 + returns).cumprod()
    cumulative_return = equity.iloc[-1] - 1

    annualized_return = (
        equity.iloc[-1] ** (n_days / max(len(returns), 1)) - 1
    )

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
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "volatility": std * np.sqrt(n_days),
        "win_rate": win_rate,
    }


def build_results_tables(data):
    """Формирует таблицы классификационных и торговых метрик по тикерам."""

    classification_rows = []
    trading_rows = []

    for ticker, group in data.groupby("Ticker"):
        probs = group.get("Probability")
        classification = calculate_metrics(
            group["Actual"],
            group["Prediction"],
            probabilities=probs if probs is not None else None,
        )
        flat = {
            "ticker": ticker,
            "n_days": len(group),
            **{
                k: v for k, v in classification.items()
                if k != "confusion_matrix"
            },
        }
        classification_rows.append(flat)

        returns = group["Strategy_Return"]
        fin = calculate_financial_metrics(returns)
        trading_rows.append({
            "ticker": ticker,
            "n_days": len(group),
            **fin,
            "total_turnover": group["Turnover"].sum(),
            "n_trades": int((group["Position_Change"] > 0).sum()),
            "commissions_total": group["Commission_Paid"].sum(),
            "slippage_total": group["Slippage_Paid"].sum(),
            "buy_hold_cumulative_return": (
                group["Buy_Hold_Equity"].iloc[-1] - 1
            ),
        })

    return (
        pd.DataFrame(classification_rows),
        pd.DataFrame(trading_rows),
    )


def plot_equity_curve(data, output_path="equity_curve.png"):
    """Строит кривые капитала и просадки для каждого тикера."""

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for ticker, group in data.groupby("Ticker"):
        label = f"{ticker}"
        axes[0].plot(group["Date"], group["Strategy_Equity"], label=ticker)
        axes[1].plot(group["Date"], group["Buy_Hold_Equity"], label=ticker)
        eq = group["Strategy_Equity"]
        dd = eq / eq.cummax() - 1
        axes[2].plot(group["Date"], dd, label=ticker)

    axes[0].set_title("Strategy Equity Curve")
    axes[1].set_title("Buy & Hold Equity Curve")
    axes[2].set_title("Strategy Drawdown")
    axes[2].set_ylabel("Drawdown")

    for ax in axes:
        ax.set_ylabel("Growth of $1")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[2].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    data = load_data()
    data = calculate_trading_results(data)

    results_classification, results_trading = build_results_tables(data)

    results_classification.to_csv("results_classification.csv", index=False)
    results_trading.to_csv("results_trading.csv", index=False)

    print("\nКлассификационные метрики по тикеру:")
    print(results_classification.to_string(index=False))

    print("\nТорговые метрики по тикеру:")
    print(results_trading.to_string(index=False))

    plot_equity_curve(data)
    print("\nГрафик сохранён в equity_curve.png")
    print("Таблицы: results_classification.csv, results_trading.csv")


if __name__ == "__main__":
    main()
