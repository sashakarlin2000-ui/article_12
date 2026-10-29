"""Роль 4 — Прогнозы через LLM: единый вход/выход, кэш, логирование.

Единый вход:
    {
      "ticker": "AAPL",
      "date": "2024-03-10",
      "features": {...},
      "market_regime": "Bear",
      "similar_days": [...]
    }

Единый выход:
    {
      "model": "...",
      "prediction": "UP|DOWN|HOLD",
      "confidence": 0.0..1.0,
      "probability_up": 0.0..1.0,
      "explanation": "..."
    }

Providers:
- MockLLMProvider — детерминированный офлайн-модель (для тестов и сквозных
  прогонов без API-ключа; результаты являются заглушкой).
- OpenAIChatProvider — OpenAI-совместимый API (ключ берётся из переменной
  окружения OPENAI_API_KEY; секреты в репозиторий не попадают).
- AnymodelProvider — AnyModel (https://anymodel.org/v1), один ключ для
  бесплатных моделей am/*; ключ в переменной окружения ANYMODEL_API_KEY.

Невалидный ответ никогда не превращается в случайный сигнал: поднимается
ошибка, в лог пишется raw response.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

# Символы, недопустимые в именах файлов/папок на Windows
_UNSAFE_NAME_RE = re.compile(r"[:/\\*?\"<>|]+")


def safe_name(name):
    """Делает имя модели безопасным для использования в путях кэша."""
    return _UNSAFE_NAME_RE.sub("_", name)

DEFAULT_CACHE_DIR = Path(__file__).resolve().with_name("llm_cache")

DEFAULT_PROMPT_TEMPLATE = (
    "You are a financial forecast assistant.\n"
    "Given the market state and similar historical days, predict whether the "
    "ticker will move UP or DOWN over the next trading day (or HOLD if the "
    "signal is weak).\n\n"
    "Ticker: {ticker}\n"
    "Date: {date}\n"
    "Market regime: {market_regime}\n"
    "Features: {features}\n"
    "Similar historical days: {similar_days}\n\n"
    "Answer ONLY with a JSON object:\n"
    '{{"prediction": "UP|DOWN|HOLD", "confidence": <0..1>, '
    '"probability_up": <0..1>, "explanation": "<short reason>"}}\n'
)

VALID_PREDICTIONS = ("UP", "DOWN", "HOLD")


class LLMOutputError(ValueError):
    """Ответ модели не соответствует единому выходному формату."""


def build_llm_payload(
    ticker,
    date,
    features,
    market_regime=None,
    similar_days=None,
):
    """Собирает единый входной формат для LLM."""
    payload = {
        "ticker": str(ticker),
        "date": str(pd_str(date)),
        "features": {k: float(v) for k, v in features.items()},
        "market_regime": None if market_regime is None else str(market_regime),
        "similar_days": list(similar_days or []),
    }
    return payload


def pd_str(value):
    import pandas as pd
    stamp = pd.to_datetime(value)
    return stamp.strftime("%Y-%m-%d")


def build_prompt(payload, template=DEFAULT_PROMPT_TEMPLATE):
    """Формирует полный prompt из входного payload."""
    return template.format(
        ticker=payload.get("ticker"),
        date=payload.get("date"),
        market_regime=payload.get("market_regime") or "unknown",
        features=json.dumps(payload.get("features", {}), ensure_ascii=False),
        similar_days=json.dumps(
            payload.get("similar_days", []), ensure_ascii=False
        ),
    )


def parse_llm_output(raw):
    """Извлекает и валидирует JSON-ответ. Ошибка -> LLMOutputError."""
    if not isinstance(raw, str) or not raw.strip():
        raise LLMOutputError("Пустой ответ модели")

    text = raw.strip()
    # Gemma (через Gemini API), MiniMax и некоторые reasoning-модели
    # оборачивают ответ в служебный блок — убираем его целиком
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
    text = re.sub(r"<mm:think>.*?</mm:think>", "", text, flags=re.DOTALL)
    text = re.sub(r" openid", "", text, flags=re.DOTALL)
    # Убираем markdown-обёртку ```...``` при её наличии
    if text.startswith("```"):
        stripped = text.strip("`")
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first == -1 or last == -1:
            raise LLMOutputError("В ответе модели нет JSON-объекта")
        stripped = stripped[first:last + 1]
        text = stripped

    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1:
        raise LLMOutputError("В ответе модели нет JSON-объекта")

    obj = json.loads(text[first:last + 1])

    if not isinstance(obj, dict):
        raise LLMOutputError("Ответ модели должен быть JSON-объектом")

    required = ("prediction", "confidence", "probability_up")
    missing = [k for k in required if k not in obj]
    if missing:
        raise LLMOutputError(f"Отсутствуют поля: {missing}")

    prediction = str(obj["prediction"]).upper().strip()
    if prediction not in VALID_PREDICTIONS:
        raise LLMOutputError(
            f"Недопустимое prediction: {obj['prediction']!r}"
        )

    def _as_float(key):
        value = obj.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LLMOutputError(f"Поле {key} должно быть числом")
        value = float(value)
        if not (0.0 <= value <= 1.0):
            raise LLMOutputError(f"Поле {key} вне диапазона [0, 1]")
        return value

    confidence = _as_float("confidence")
    probability_up = _as_float("probability_up")

    return {
        "model": str(obj.get("model", "")),
        "prediction": prediction,
        "confidence": confidence,
        "probability_up": probability_up,
        "explanation": str(obj.get("explanation", ""))[:500],
    }


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

class LLMProvider:
    """Интерфейс провайдера: вызывает модель и возвращает сырой текст."""

    def name(self):
        raise NotImplementedError

    def complete(self, prompt, temperature=0.2, max_retries=2):
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    """Детерминированная офлайн-заглушка (только для тестов/интеграции).

    Правила style="momentum" | "reversion" | "rsi":
    - momentum:   UP при заметном положительном return_5d, иначе DOWN,
                  HOLD при слабом сигнале;
    - reversion:  зеркальное правило к momentum;
    - rsi:        UP при RSI ниже нейтральной зоны перепроданности,
                  DOWN при RSI выше зоны перекупленности, HOLD внутри зоны.

    Провайдер использует ТОЛЬКО признаки текущего дня. Аналоги (similar_days)
    передаются в prompt, но их future outcomes в заглушку не заходят —
    это соблюдает запрет утечки будущих данных.
    """

    # Пороги HOLD по умолчанию подобраны так, чтобы разные mock-модели
    # давали разные сигналы (reversion реагирует на более сильные движения)
    STYLE_DEFAULT_HOLD_DELTA = {
        "momentum": 0.005,
        "reversion": 0.02,
        "rsi": 0.0,
    }

    RSI_ENTRY = 40.0

    def __init__(
        self,
        model_name="mock-rule-based",
        style="momentum",
        hold_delta=None,
    ):
        self.model_name = model_name
        self.style = style
        if hold_delta is None:
            hold_delta = self.STYLE_DEFAULT_HOLD_DELTA.get(style, 0.005)
        self.hold_delta = hold_delta

    def name(self):
        return self.model_name

    def complete(self, prompt, temperature=0.0, max_retries=2):
        features_line = next(
            (ln for ln in prompt.splitlines() if ln.startswith("Features: ")),
            "{}",
        )
        try:
            features = json.loads(
                features_line.split(": ", 1)[1]
            )
        except Exception:
            raise LLMOutputError(
                "Mock-провайдер не смог разобрать признаки в prompt"
            )

        return_5d = float(features.get("return_5d", 0.0) or 0.0)
        rsi_raw = features.get("rsi", features.get("RSI"))
        rsi = float(rsi_raw) if rsi_raw is not None else None

        if self.style == "momentum":
            vote = return_5d
        elif self.style == "reversion":
            vote = -return_5d
        elif self.style == "rsi":
            if rsi is None:
                vote = 0.0
            elif rsi < 45.0:
                vote = 0.05
            elif rsi > 55.0:
                vote = -0.05
            else:
                vote = 0.0
        else:
            raise ValueError(f"Неизвестный стиль mock-провайдера: {self.style}")

        if vote > self.hold_delta:
            prediction, probability_up = "UP", 0.5 + min(vote * 10, 0.25)
            confidence = 0.5 + min(vote * 10, 0.2)
        elif vote < -self.hold_delta:
            prediction, probability_up = "DOWN", 0.5 + min(-vote * 10, 0.25)
            confidence = 0.5 + min(-vote * 10, 0.2)
        else:
            prediction, probability_up = "HOLD", 0.5
            confidence = 0.5

        return json.dumps({
            "model": self.model_name,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probability_up": round(probability_up, 4),
            "explanation": f"mock {self.style} rule: signal {vote:+.4f}",
        })


class OpenAIChatProvider(LLMProvider):
    """OpenAI-совместимый провайдер (ключ из переменной окружения)."""

    def __init__(
        self,
        model_name="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        extra_headers=None,
        max_tokens=None,
    ):
        self.model_name = model_name
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.extra_headers = extra_headers or {}
        self.max_tokens = max_tokens

    def name(self):
        return f"{type(self).__name__.lower().replace('provider','')}:{self.model_name}"

    def complete(self, prompt, temperature=0.2, max_retries=2):
        import requests

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Задайте переменную окружения {self.api_key_env}; "
                "секреты нельзя хранить в репозитории"
            )

        request_body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You output only JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        if self.max_tokens is not None:
            request_body["max_tokens"] = self.max_tokens

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                **self.extra_headers,
            },
            json=request_body,
            timeout=120,
        )

        if response.status_code != 200:
            detail = response.text[:200]
            raise RuntimeError(
                f"{self.base_url} вернул {response.status_code}: {detail}"
            )

        return response.json()["choices"][0]["message"]["content"]


class OpenRouterProvider(OpenAIChatProvider):
    """OpenRouter (https://openrouter.ai) — бесплатные модели с суффиксом ':free'.

    Ключ в переменной окружения OPENROUTER_API_KEY."""

    def __init__(self, model_name, api_key_env="OPENROUTER_API_KEY", max_tokens=None):
        super().__init__(
            model_name=model_name,
            api_key_env=api_key_env,
            base_url="https://openrouter.ai/api/v1",
            extra_headers={
                "HTTP-Referer": "https://github.com/article-12",
                "X-Title": "Market Forecast Research",
            },
            max_tokens=max_tokens,
        )


class OrcaRouterProvider(OpenAIChatProvider):
    """OrcaRouter (https://api.orcarouter.ai/v1) — OpenAI-совместимый роутер.

    Бесплатные модели заканчиваются на '-free' (требуется привязанный
    GitHub-аккаунт в консоли OrcaRouter). Ключ: ORCA_API_KEY."""

    def __init__(self, model_name, api_key_env="ORCA_API_KEY"):
        super().__init__(
            model_name=model_name,
            api_key_env=api_key_env,
            base_url="https://api.orcarouter.ai/v1",
        )


class GeminiProvider(OpenAIChatProvider):
    """Google Gemini через OpenAI-совместимый эндпоинт.

    Ключ из Google AI Studio в переменной окружения GEMINI_API_KEY.
    Работающие модели (октябрь 2026): gemini-3-flash-preview,
    gemini-3.1-flash-lite, gemini-3.5-flash-lite."""

    def __init__(self, model_name, api_key_env="GEMINI_API_KEY"):
        super().__init__(
            model_name=model_name,
            api_key_env=api_key_env,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )


class AnymodelProvider(OpenAIChatProvider):
    """AnyModel (https://anymodel.org/v1) — OpenAI-совместимый шлюз.

    Бесплатные text-модели имеют id вида ``am/...`` (coefficient 0).
    Ключ берётся из переменной окружения ANYMODEL_API_KEY."""

    def __init__(self, model_name, api_key_env="ANYMODEL_API_KEY",
                 max_tokens=None):
        super().__init__(
            model_name=model_name,
            api_key_env=api_key_env,
            base_url="https://anymodel.org/v1",
            max_tokens=max_tokens,
        )


# --------------------------------------------------------------------------
# Прогноз одного дня
# --------------------------------------------------------------------------

def _cache_path(cache_dir, payload, model_name):
    key = json.dumps(
        {
            "model": model_name,
            "ticker": payload.get("ticker"),
            "date": payload.get("date"),
            "features": payload.get("features"),
            "market_regime": payload.get("market_regime"),
            "similar_days": payload.get("similar_days"),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{safe_name(model_name)}_{digest}.json"


def _record_to_jsonl(record, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "llm_prompts_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def predict_for_day(
    payload,
    provider,
    *,
    retrieval_mode="standard",
    temperature=0.2,
    cache_dir=DEFAULT_CACHE_DIR,
    log_dir=Path("results/llm"),
    generation_params=None,
    max_retries=2,
    use_cache=True,
):
    """Возвращает единый JSON-ответ для одного дня.

    - кэширует ответы по (модель, входной payload);
    - сохраняет prompt, ответ и время вызова в лог;
    - при невалидном JSON повторяет запрос до `max_retries` раз;
    - после исчерпания повторов поднимает LLMOutputError.
    """
    cache_dir = Path(cache_dir)
    cache_file = _cache_path(
        cache_dir, payload, provider.name(),
    )

    prompt = build_prompt(payload)
    record = {
        "model": provider.name(),
        "ticker": payload.get("ticker"),
        "date": payload.get("date"),
        "retrieval_mode": retrieval_mode,
        "market_regime": payload.get("market_regime"),
        "n_similar_days": len(payload.get("similar_days", [])),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "generation_params": {
            "temperature": temperature,
            **(generation_params or {}),
        },
        "called_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cached": False,
    }

    if use_cache and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if "prediction" in cached:
                record["cached"] = True
                _record_to_jsonl(record, Path(log_dir))
                return cached
        except Exception:
            pass  # повреждённый кэш игнорируется, делаем обычный вызов

    last_error = None
    for attempt in range(1 + int(max_retries)):
        try:
            raw = provider.complete(prompt, temperature=temperature)
        except Exception as exc:  # сетевые/API ошибки логируются
            last_error = f"provider_error: {exc}"
            time.sleep(1)
            continue

        record["raw_response"] = raw
        try:
            parsed = parse_llm_output(raw)
            parsed["model"] = parsed["model"] or provider.name()
            record["status"] = "ok"
            last_error = None
            break
        except Exception as exc:
            last_error = f"invalid_output: {exc}"

    if last_error is not None:
        record["status"] = "error"
        record["error"] = last_error
        _record_to_jsonl(record, Path(log_dir))
        raise LLMOutputError(
            f"{provider.name()} {payload.get('date')}: {last_error}"
        )

    if not use_cache:
        _record_to_jsonl(record, Path(log_dir))
        return parsed

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(parsed, ensure_ascii=False), encoding="utf-8"
    )
    _record_to_jsonl(record, Path(log_dir))
    return parsed


if __name__ == "__main__":
    provider = MockLLMProvider()
    payload = build_llm_payload(
        ticker="SPX",
        date="2024-03-10",
        features={"return_5d": 0.02, "volatility_20d": 0.15, "rsi": 58.4},
        market_regime="Bear",
        similar_days=[
            {"date": "2020-03-12", "similarity": 0.94,
             "forward_return": -0.021},
            {"date": "2022-06-13", "similarity": 0.92,
             "forward_return": -0.018},
        ],
    )
    print(json.dumps(
        predict_for_day(payload, provider, log_dir=Path("results/llm")),
        ensure_ascii=False, indent=2,
    ))
