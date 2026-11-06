"""ИИ-анализ через OpenRouter: модель получает всё, что посчитал DEN, и пишет разбор с выводом лонг/шорт.

Ключ берётся из DEN_OPENROUTER_KEY (или OPENROUTER_API_KEY). Модель задаётся в DEN_AI_MODEL;
по умолчанию `openrouter/free`: OpenRouter сам выбирает доступную бесплатную модель.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"

SYSTEM_PROMPT = """Ты аналитик краткосрочной торговли фьючерсами и токенизированными активами.
Тебе дают сводку по одному активу: цену, голоса индикаторов и консенсус по таймфреймам, уровни,
ожидаемые диапазоны цены, проверку сигналов на истории и свежие заголовки новостей.
Оцени, что вероятнее в ближайшие часы и сутки: рост, падение или боковик. Опирайся только на данные
из сводки, не выдумывай цифры и события. Если факторы противоречат друг другу или сигналы на истории
работали хуже монетки, честно снижай уверенность.

Ответь строго одним JSON-объектом без пояснений вокруг, на русском языке:
{"direction": "лонг" | "шорт" | "нейтрально",
 "confidence": "низкая" | "средняя" | "высокая",
 "summary": "3-6 предложений: главный вывод и почему",
 "key_factors": ["до 5 коротких пунктов"],
 "risks": ["до 4 коротких пунктов: что сломает сценарий"],
 "levels": "ключевые уровни для входа, стопа и целей одной строкой"}"""


class AIError(RuntimeError):
    pass


@dataclass
class AIResult:
    direction: str
    confidence: str
    summary: str
    key_factors: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    levels: str = ""
    model: str = ""
    raw: str = ""


def api_key() -> str | None:
    return (os.environ.get("DEN_OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY") or "").strip() or None


def model_name() -> str:
    return (os.environ.get("DEN_AI_MODEL") or DEFAULT_MODEL).strip()


def _f(x: float) -> str:
    return f"{x:,.4f}" if abs(x) < 10 else f"{x:,.2f}"


def build_context(asset: str, contract: str | None, results: dict, total_score: float, total_label: str,
                  verdict=None, ranges: list[dict] | None = None, backtests: dict | None = None,
                  news: list | None = None, news_sentiment: dict | None = None, max_news: int = 20,
                  signal=None, backdrop=None) -> str:
    """Собрать компактную текстовую сводку для модели из того, что уже посчитал дашборд."""
    lines = [f"Актив: {asset}" + (f" (контракт {contract})" if contract else "")]
    any_res = next(iter(results.values()), None)
    if any_res is not None:
        lines.append(f"Цена: {_f(any_res.price)}")
    lines.append(f"Общий консенсус по всем таймфреймам: {total_label} (балл {total_score:+.2f}, от -1 до +1)")
    for tf, r in results.items():
        lines.append(f"\n## Таймфрейм {tf}: {r.label}, за лонг {r.long_pct:.0f}%, "
                     f"изменение за свечу {r.change_pct:+.2f}%, ATR {_f(r.atr)}"
                     + (f", режим рынка: {r.regime.label.lower()} ({r.regime.reason})"
                        if getattr(r, "regime", None) else ""))
        for v in r.votes:
            lines.append(f"- {v.name}: {v.score:+.2f} (вес {v.weight}), {v.reason}")
        if r.levels:
            lines.append("Сопротивления: " + (", ".join(_f(x) for x in r.levels.get("resistance", [])) or "нет"))
            lines.append("Поддержки: " + (", ".join(_f(x) for x in r.levels.get("support", [])) or "нет"))
    if signal is not None:
        lines.append(f"\n## Итоговый сигнал DEN: {signal.label}, уверенность {signal.confidence:.0f}% "
                     f"(порог {signal.threshold:.0f}%)")
        lines += [f"- {x}" for x in signal.reasons]
    if backdrop is not None:
        lines.append(f"\n## Геополитический фон: {backdrop.label} (не голос в консенсусе, влияет на размер позиции)")
        lines += [f"- {x}" for x in backdrop.reasons]
    if verdict is not None:
        lines.append(f"\n## Вердикт DEN: {verdict.label}, уверенность {verdict.confidence}")
        lines += [f"- {x}" for x in verdict.reasons]
    if ranges:
        lines.append("\n## Ожидаемый диапазон цены по волатильности")
        for r in ranges:
            lines.append(f"- {r['Горизонт']}: 68% {_f(r['68%: от'])}–{_f(r['68%: до'])}, "
                         f"95% {_f(r['95%: от'])}–{_f(r['95%: до'])}, ход ±{r['Ход ±%']:.2f}%")
    if backtests:
        lines.append("\n## Как сигналы срабатывали на истории")
        for name, b in backtests.items():
            lo, sh = b["long"], b["short"]
            lines.append(f"- через {name}: лонг {lo['n']} сигналов, угадал "
                         + (f"{lo['hit']:.0%}" if lo["n"] else "–")
                         + f"; шорт {sh['n']} сигналов, угадал " + (f"{sh['hit']:.0%}" if sh["n"] else "–")
                         + f"; рост без сигнала {b['base_up']:.0%}")
    if news:
        head = "\n## Заголовки новостей за 3 дня (новые сверху)"
        if news_sentiment:
            head += f", тон {news_sentiment['score']:+.2f}"
        lines.append(head)
        for n in news[:max_news]:
            lines.append(f"- {n.published:%d.%m %H:%M} {n.source}: {n.title}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    candidate = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    if not candidate:
        raise ValueError("в ответе нет JSON")
    return json.loads(candidate)


def parse_answer(text: str, model: str = "") -> AIResult:
    """Разобрать ответ модели. Если JSON не получился, показать текст как есть."""
    try:
        d = _extract_json(text)
    except (ValueError, json.JSONDecodeError):
        return AIResult("нейтрально", "низкая", text.strip() or "Модель вернула пустой ответ.", model=model, raw=text)

    def as_list(x) -> list[str]:
        if isinstance(x, str):
            return [x] if x.strip() else []
        return [str(i) for i in (x or [])]

    direction = str(d.get("direction", "нейтрально")).lower().strip()
    if direction not in ("лонг", "шорт", "нейтрально"):
        direction = "лонг" if "long" in direction or "лонг" in direction else \
                    "шорт" if "short" in direction or "шорт" in direction else "нейтрально"
    return AIResult(direction=direction, confidence=str(d.get("confidence", "низкая")).lower().strip(),
                    summary=str(d.get("summary", "")).strip(), key_factors=as_list(d.get("key_factors")),
                    risks=as_list(d.get("risks")), levels=str(d.get("levels", "") or "").strip(),
                    model=model, raw=text)


def _chat(messages: list[dict], key: str | None = None, model: str | None = None, timeout: float = 120,
          max_tokens: int = 4000, temperature: float = 0.3) -> tuple[str, str]:
    """Один запрос к OpenRouter. Возвращает (текст ответа, имя модели, которая ответила)."""
    import requests

    key = key or api_key()
    if not key:
        raise AIError("Нет ключа OpenRouter. Задайте DEN_OPENROUTER_KEY или введите ключ в боковой панели.")
    model = model or model_name()
    proxies = {"https": os.environ["DEN_PROXY"]} if os.environ.get("DEN_PROXY") else None
    try:
        r = requests.post(API_URL, timeout=timeout, proxies=proxies,
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                   "X-Title": "DEN"},
                          json={"model": model, "temperature": temperature, "max_tokens": max_tokens,
                                "messages": messages})
    except requests.RequestException as e:
        raise AIError(f"OpenRouter недоступен: {e}") from e
    if r.status_code == 401:
        raise AIError("OpenRouter не принял ключ (401). Проверьте DEN_OPENROUTER_KEY.")
    if r.status_code == 429:
        raise AIError("Лимит бесплатных запросов OpenRouter исчерпан (429). Подождите минуту "
                      "или до завтра; после разовой покупки кредитов на $10 лимит вырастет до 1000 в день.")
    try:
        data = r.json()
    except ValueError:
        raise AIError(f"OpenRouter вернул {r.status_code}: {r.text[:300]}") from None
    if r.status_code >= 400 or "error" in data:
        msg = (data.get("error") or {}).get("message") if isinstance(data.get("error"), dict) else data.get("error")
        raise AIError(f"OpenRouter вернул ошибку {r.status_code}: {msg or r.text[:300]}")
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise AIError(f"Неожиданный ответ OpenRouter: {str(data)[:300]}") from None
    return msg.get("content") or msg.get("reasoning") or "", data.get("model", model)


def analyze_with_ai(context: str, key: str | None = None, model: str | None = None,
                    timeout: float = 120) -> AIResult:
    text, used = _chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": context}],
                       key, model, timeout)
    return parse_answer(text, used)


ASSET_TITLES = {"GOLD": "золото (XAU)", "SILVER": "серебро (XAG)", "AAPL": "акции Apple (AAPL)"}

NEWS_PROMPT = """Ты оцениваешь, как новостные заголовки повлияют на цену актива в ближайшие часы и сутки.
Для каждого заголовка поставь число от -1 до 1: -1 — сильно давит на цену вниз, 0 — не влияет или
не про этот актив, 1 — сильно толкает вверх. Учитывай смысл, а не отдельные слова: для металлов рост
доходностей и сильный доллар обычно давят на цену, снижение ставок и геополитические риски поддерживают;
для акций важны отчёты, прогнозы, рейтинги аналитиков, суды и пошлины.
Ответь строго одним JSON-объектом без пояснений: {"scores": [числа в том же порядке, ровно N штук]}"""


def parse_scores(text: str, n: int) -> list[float]:
    d = _extract_json(text)
    vals = d.get("scores") if isinstance(d, dict) else None
    if not isinstance(vals, list) or len(vals) != n:
        raise ValueError(f"ожидалось {n} оценок, пришло {len(vals) if isinstance(vals, list) else 0}")
    return [max(-1.0, min(1.0, float(v))) for v in vals]


def score_headlines(titles: list[str], asset: str, key: str | None = None, model: str | None = None,
                    timeout: float = 90) -> tuple[list[float], str]:
    """Тон заголовков через модель одним запросом. Возвращает (оценки по порядку, модель)."""
    if not titles:
        return [], model or model_name()
    body = f"Актив: {ASSET_TITLES.get(asset, asset)}. Заголовков: {len(titles)}.\n" + \
        "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    text, used = _chat([{"role": "system", "content": NEWS_PROMPT.replace("N штук", f"{len(titles)} штук")},
                        {"role": "user", "content": body}], key, model, timeout,
                       max_tokens=2000, temperature=0.0)
    try:
        return parse_scores(text, len(titles)), used
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        raise AIError(f"Модель вернула оценки новостей в неожиданном виде: {e}") from None


def rescore_news(items: list, asset: str, key: str | None = None, model: str | None = None) -> tuple[list, str]:
    """Новости с тоном от модели вместо словаря (NewsItem с заменённым score)."""
    from dataclasses import replace

    scores, used = score_headlines([n.title for n in items], asset, key, model)
    return [replace(n, score=s) for n, s in zip(items, scores)], used


def rescore_news_throttled(items: list, asset: str, store: dict, model: str | None = None,
                           min_interval: float = 1800, now: float | None = None) -> tuple[list, str]:
    """Как rescore_news, но не чаще раза в min_interval секунд на актив и модель.
    Между запросами уже оценённые заголовки берут оценку модели, новые — оценку словаря."""
    import time
    from dataclasses import replace

    now = time.time() if now is None else now
    slot = (asset, model or model_name())
    entry = store.get(slot)
    if entry and now - entry[0] < min_interval:
        known = entry[1]
        return [replace(n, score=known.get(n.title, n.score)) for n in items], entry[2]
    out, used = rescore_news(items, asset, model=model)
    store[slot] = (now, {n.title: n.score for n in out}, used)
    return out, used
