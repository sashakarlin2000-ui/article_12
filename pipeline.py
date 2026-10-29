"""Сквозной запуск исследовательского конвейера (Роль 7, Этап 3).

data_loader -> features -> market_data_with_regime -> retrieval -> LLM ->
ensemble -> backtest -> metrics.

Схема работы:
1. Загружает market_data_with_regime.csv (продукт Роли 1 и Роли 2).
2. Для каждого поля (T, ticker) запускает retrieval (обычный или regime-aware).
3. Собирает единый входной JSON и передаёт его всем LLM-моделям.
4. Ансамбль объединяет ответы в итоговый сигнал UP/DOWN/HOLD.
5. Прогнозы сохраняются в predictions.csv, метрики — в результаты CSV.

Аналоги передаются моделям БЕЗ их future outcomes (запрет утечки):
forward_return остаётся только в служебных записях для анализа.
Работа с реальными LLM подключается через OpenAIChatProvider
(ключ в переменной окружения OPENAI_API_KEY); по умолчанию используются
mock-провайдеры, о чём явно сообщается в результатах.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from retrieval import find_similar_days
from llm_predictor import (
    AnymodelProvider,
    GeminiProvider,
    LLMOutputError,
    MockLLMProvider,
    OpenAIChatProvider,
    OpenRouterProvider,
    OrcaRouterProvider,
    predict_for_day,
    safe_name,
)
from ensemble import (
    CONFIDENCE_WEIGHTED_VOTING,
    MAJORITY_VOTING,
    REGIME_WEIGHTED_VOTING,
    default_regime_weights,
    make_ensemble_decision,
)
from backtest import build_results_tables, calculate_trading_results
from metrics import calculate_metrics

FEATURES_FOR_LLM = {
    "return_5d": "return_5d",
    "volatility_20d": "volatility_20d",
    "RSI": "rsi",
    "distance_from_MA200": "distance_from_MA200",
}
REGIME_LABEL_COLUMN = "market_regime_label"
DEFAULT_LABEL_COL = "market_regime_label"


# --------------------------------------------------------------------------
# Провайдеры моделей
# --------------------------------------------------------------------------

def build_providers(provider_names):
    """Создаёт провайдеров по их именам.

    Роль 4: чтобы задать несколько моделей, перечислите их в
    --providers. Mock-стили детерминированы и не дают утечки будущего.
    """
    providers = []
    for name in provider_names:
        if name.startswith("mock:"):
            style = name.split(":", 1)[1]
            providers.append(
                MockLLMProvider(
                    model_name=f"mock-{style}",
                    style=style,
                )
            )
        elif name.startswith("openai:"):
            providers.append(
                OpenAIChatProvider(model_name=name.split(":", 1)[1])
            )
        elif name.startswith("openrouter:"):
            model = name.split(":", 1)[1]
            # Платным моделям на аккаунтах без кредитов OpenRouter требует
            # ограничение max_tokens, иначе отвечает 402
            providers.append(
                OpenRouterProvider(
                    model_name=model,
                    max_tokens=1024 if model.startswith("qwen") else None,
                )
            )
        elif name.startswith("orca:"):
            providers.append(
                OrcaRouterProvider(model_name=name.split(":", 1)[1])
            )
        elif name.startswith("gemini:"):
            providers.append(
                GeminiProvider(model_name=name.split(":", 1)[1])
            )
        elif name.startswith("anymodel:"):
            model = name.split(":", 1)[1]
            # reasoning-модели тратят бюджет на thinking и без запаса
            # возвращают пустой content
            if model.startswith(("am/nemotron", "am/gpt-oss")):
                max_tokens = 4096
            else:
                max_tokens = 512
            providers.append(
                AnymodelProvider(model_name=model, max_tokens=max_tokens)
            )
        else:
            raise ValueError(f"Неизвестный провайдер: {name!r}")
    return providers


# --------------------------------------------------------------------------
# Один прогнозный шаг
# --------------------------------------------------------------------------

def predict_one_day(
    row,
    data,
    providers,
    *,
    retrieval_mode="standard",
    top_k=3,
    ensemble_method=MAJORITY_VOTING,
    weights=None,
    use_cache=True,
    cache_dir=None,
    log_dir=Path("results/llm"),
):
    """Возвращает итоговое решение и подробности для одного дня и тикера.

    row — строка датасета (Date, Ticker, features, market_regime_label).
    """
    ticker = str(row["Ticker"])
    date = pd.Timestamp(row["Date"]).strftime("%Y-%m-%d")
    regime = row.get(REGIME_LABEL_COLUMN)
    regime = None if pd.isna(regime) else str(regime)

    retrieval = find_similar_days(
        current_day=pd.Timestamp(date),
        ticker=ticker,
        top_k=top_k,
        data=data,
        regime_aware=(retrieval_mode == "regime-aware"),
    )

    similar_days_slim = [
        {
            "date": item["date"],
            "similarity": item["similarity"],
            "regime": item.get("regime"),
        }
        for item in retrieval["similar_days"]
    ]

    features = {}
    for source_col, payload_name in FEATURES_FOR_LLM.items():
        if source_col in row.index and pd.notna(row[source_col]):
            features[payload_name] = float(row[source_col])

    payload = {
        "ticker": ticker,
        "date": date,
        "features": features,
        "market_regime": regime,
        "similar_days": similar_days_slim,
    }

    model_responses = []
    failed_models = []
    for provider in providers:
        kwargs = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = Path(cache_dir) / safe_name(provider.name())
        kwargs["retrieval_mode"] = retrieval_mode
        try:
            response = predict_for_day(
                payload, provider, log_dir=log_dir,
                use_cache=use_cache, **kwargs,
            )
            model_responses.append(response)
        except Exception as exc:
            # Отказ модели (квота, сеть, невалидный JSON) не превращается
            # в случайный сигнал: провайдер исключается из ансамбля,
            # факт ошибки логируется.
            failed_models.append(
                {"model": provider.name(), "error": str(exc)[:200]}
            )

    if not model_responses:
        raise LLMOutputError(
            f"{date} {ticker}: ни одна модель не дала валидного ответа; "
            f"ошибки: {failed_models}"
        )

    decision = make_ensemble_decision(
        model_responses,
        method=ensemble_method,
        weights=weights,
        regime=regime,
    )

    probability_up = float(
        np.mean([r.get("probability_up", 0.5) for r in model_responses])
    )

    return {
        "Date": date,
        "Ticker": ticker,
        "Prediction": decision["prediction"],
        "Probability": round(probability_up, 4),
        "Score": round(float(decision["score"]), 4),
        "Method": decision["method"],
        "Regime": regime,
        "RetrievalMode": retrieval_mode,
        "NAnalogues": retrieval["returned_count"],
        "FallbackUsed": retrieval["fallback_used"],
        "ModelsFailed": json.dumps(
            failed_models,
            ensure_ascii=False,
        ),
        "ModelPredictions": json.dumps(
            [
                {
                    "model": r["model"],
                    "prediction": r["prediction"],
                    "confidence": r["confidence"],
                }
                for r in model_responses
            ],
            ensure_ascii=False,
        ),
    }


# --------------------------------------------------------------------------
# Прогон по периоду
# --------------------------------------------------------------------------

def run_pipeline(
    data_path="data/processed/market_data_with_regime.csv",
    tickers=None,
    last_n_days=60,
    start_date=None,
    end_date=None,
    retrieval_mode="standard",
    ensemble_method=MAJORITY_VOTING,
    top_k=3,
    providers=None,
    use_cache=True,
    output_predictions="predictions.csv",
    output_dir="results",
    cache_dir="llm_cache",
):
    """Прогоняет конвейер по всем тикерам и датам, возвращает predictions."""
    data = pd.read_csv(data_path)
    data["Date"] = pd.to_datetime(data["Date"])

    providers = providers or build_providers(
        ["mock:momentum", "mock:reversion", "mock:rsi"]
    )

    if ensemble_method == REGIME_WEIGHTED_VOTING:
        models = [p.name() for p in providers]
        weights = default_regime_weights(models)
    else:
        weights = None

    results = []
    for ticker in sorted(data["Ticker"].unique()) if tickers is None else tickers:
        ticker_data = data[data["Ticker"].astype(str) == ticker].copy()
        ticker_data = ticker_data.sort_values("Date")

        # Последняя строка тикера не имеет известного следующего дня:
        # на ней нельзя ни сформировать target, ни оценить сделку.
        has_next = ticker_data["Close"].shift(-1).notna()
        dates = (
            ticker_data.loc[has_next, "Date"]
            .sort_values()
            .unique()
            .tolist()
        )

        if start_date is not None or end_date is not None:
            window = [
                pd.Timestamp(d) for d in dates
                if (start_date is None or pd.Timestamp(d) >= pd.Timestamp(start_date))
                and (end_date is None or pd.Timestamp(d) <= pd.Timestamp(end_date))
            ]
        else:
            window = [pd.Timestamp(d) for d in dates[-last_n_days:]]

        for date in window:
            sub = ticker_data[ticker_data["Date"] == date]
            row = sub.iloc[-1]
            try:
                result = predict_one_day(
                    row,
                    data,
                    providers,
                    retrieval_mode=retrieval_mode,
                    top_k=top_k,
                    ensemble_method=ensemble_method,
                    weights=weights,
                    use_cache=use_cache,
                    cache_dir=cache_dir,
                    log_dir=Path(output_dir) / "llm",
                )
            except LLMOutputError as exc:
                # Полный отказ всех моделей в этот день — день пропускается
                # (не заполняется случайным сигналом), позже можно перезапустить.
                print(f"SKIP {ticker} {str(date.date())}: {str(exc)[:160]}")
                continue
            results.append(result)

    frame = pd.DataFrame(results)

    # Прогон по частям (по тикеру или по дню) не должен терять
    # предыдущие результаты: объединяем с существующим файлом.
    output_path = Path(output_predictions)
    if output_path.exists():
        try:
            existing = pd.read_csv(output_path)
            existing["Date"] = existing["Date"].astype(str)
            frame["Date"] = frame["Date"].astype(str)
            combined = pd.concat(
                [frame, existing], ignore_index=True
            )
            combined = combined.drop_duplicates(
                subset=["Date", "Ticker"], keep="first"
            )
            combined = combined.sort_values(["Ticker", "Date"])
            frame = combined
        except Exception as exc:
            print(f"Не удалось объединить с {output_predictions}: {exc}")

    frame.to_csv(output_predictions, index=False)

    return frame


def evaluate_predictions(
    predictions_path="predictions.csv",
    data_path="data/processed/market_data_with_regime.csv",
    output_alignment="results",
    allow_short=True,
    commission=0.001,
    slippage=0.0005,
):
    """Присоединяет фактические значения и рассчитывает таблицы метрик."""
    import backtest as bt

    # Загружаем через backtest.load_data, чтобы отработали все проверки
    # и расчёт Future_Return на исходе следующего дня
    data = bt.load_data(market_path=data_path, predictions_path=predictions_path)
    data = bt.calculate_trading_results(
        data,
        allow_short=allow_short,
        commission=commission,
        slippage=slippage,
    )

    results_classification, results_trading = build_results_tables(data)

    out_dir = Path(output_alignment)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_classification.to_csv(out_dir / "results_classification.csv", index=False)
    results_trading.to_csv(out_dir / "results_trading.csv", index=False)

    bt.plot_equity_curve(data, output_path=str(out_dir / "equity_curve.png"))

    return results_classification, results_trading


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Сквозной запуск конвейера")
    parser.add_argument("--data", default="data/processed/market_data_with_regime.csv")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--last-n-days", type=int, default=60)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--retrieval-mode", choices=["standard", "regime-aware"],
        default="standard",
    )
    parser.add_argument(
        "--ensemble-method",
        choices=[
            MAJORITY_VOTING, CONFIDENCE_WEIGHTED_VOTING,
            REGIME_WEIGHTED_VOTING,
        ],
        default=CONFIDENCE_WEIGHTED_VOTING,
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--providers", nargs="*",
        default=["mock:momentum", "mock:reversion", "mock:rsi"],
        help='Провайдеры: "mock:momentum|reversion|rsi" или "openai:model"',
    )
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    providers = build_providers(args.providers)
    print("Провайдеры:", [p.name() for p in providers])

    predictions = run_pipeline(
        data_path=args.data,
        tickers=args.tickers,
        last_n_days=args.last_n_days,
        start_date=args.start_date,
        end_date=args.end_date,
        retrieval_mode=args.retrieval_mode,
        ensemble_method=args.ensemble_method,
        top_k=args.top_k,
        providers=providers,
        use_cache=not args.no_cache,
        output_predictions=args.output,
        output_dir=args.output_dir,
    )

    print(f"\nПрогнозов сохранено: {len(predictions)} (файл {args.output})")
    print(predictions.head(10).to_string(index=False))

    results_classification, results_trading = evaluate_predictions(
        predictions_path=args.output,
        data_path=args.data,
        output_alignment=args.output_dir,
    )

    print("\nКлассификационные метрики (окно прогноза):")
    print(results_classification.to_string(index=False))

    print("\nТорговые метрики (окно прогноза):")
    print(results_trading.to_string(index=False))


if __name__ == "__main__":
    main()
