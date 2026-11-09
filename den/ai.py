"""ИИ-анализ: модель получает всё, что посчитал DEN, и оценивает вероятность роста на заданном горизонте.

Провайдер по умолчанию Tooken Club (платный посредник с OpenAI-совместимым API), модель GPT-6 Astra.
Ключ в DEN_TOOKEN_KEY, имя модели в DEN_TOOKEN_MODEL (по умолчанию `gpt-6-astra`; если в личном кабинете
tooken.club она называется иначе, укажите точное имя). Модель закреплена намеренно: ответы разных моделей
нельзя сравнивать в журнале прогнозов.

OpenRouter остался только для запуска вручную: DEN_AI_PROVIDER=openrouter, ключ в DEN_OPENROUTER_KEY.

Как устроен запрос:
- сводка начинается со времени, сессии и возраста данных, голоса индикаторов сведены по семействам,
  новости только за сутки и самые значимые, сигнал DEN стоит в самом конце и только для сравнения;
- модель отвечает вероятностью роста prob_up, направление код выводит из неё сам;
- уровни входа, стопа и цели проверяются: они должны совпадать с уровнями из сводки или с ценой ± k·ATR;
- делается несколько независимых запросов (DEN_AI_SAMPLES, по умолчанию 3), итог по медиане prob_up.
"""
from __future__ import annotations

import json
import os
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"
TOOKEN_MODEL = "gpt-6-astra"
PROMPT_VERSION = "v2"
HORIZON_H = int(os.environ.get("DEN_AI_HORIZON_H", "4") or 4)
LONG_FROM, SHORT_TO = 60, 40  # prob_up: от 60 лонг, до 40 шорт, между ними нейтрально


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    url: str
    key_envs: tuple[str, ...]
    model_env: str
    default_model: str
    key_help: str


PROVIDERS = {
    "tooken": Provider("tooken", "Tooken Club", "https://tooken.club/v1/chat/completions", ("DEN_TOOKEN_KEY",),
                       "DEN_TOOKEN_MODEL", TOOKEN_MODEL, "Ключ из личного кабинета tooken.club."),
    "openrouter": Provider("openrouter", "OpenRouter", API_URL, ("DEN_OPENROUTER_KEY", "OPENROUTER_API_KEY"),
                           "DEN_AI_MODEL", DEFAULT_MODEL, "Ключ с openrouter.ai/keys."),
}
DEFAULT_PROVIDER = "tooken"

SYSTEM_PROMPT = """Ты риск-аналитик краткосрочной торговли. Задача: независимо проверить сигнал системы DEN по одному активу и оценить вероятность, что через {H} ч цена будет выше текущей.

Правила, обязательные для каждого ответа:
1. Используй только числа и факты из сводки. Каждая цена в ответе MUST быть уровнем из сводки (поддержка, сопротивление, граница диапазона, план DEN) или ценой либо таким уровнем ± k·ATR указанного таймфрейма, k кратно 0.25.
2. Сначала оцени сам по таймфреймам, режиму рынка, бэктесту и новостям. Сигнал DEN в конце сводки используй только для сравнения, не копируй его.
3. Индикатор или сигнал, у которого на истории доля попаданий не выше базовой частоты роста + 5 п.п., считай шумом.
4. Если 1d или 1w против младших таймфреймов, или режим рынка «боковик», prob_up MUST быть от 40 до 60.
5. Новости учитывай только за последние 24 ч и только прямо про актив, ставки ФРС, доллар, пошлины или войну. Если цена уже сдвинулась после новости, она отыграна и веса не имеет.
6. Вне основной сессии и при старых данных снижай уверенность: prob_up ближе к 50.
7. При противоречивых или скудных данных выбирай «нейтрально». Слабый сигнал хуже честного отказа.

Ответь одним JSON-объектом без текста до и после, на русском:
{"prob_up": целое 0-100,
 "direction": "лонг" если prob_up >= 60, "шорт" если prob_up <= 40, иначе "нейтрально",
 "agree_with_den": true или false,
 "summary": "2-4 предложения: вывод и главная причина",
 "key_factors": ["до 4 пунктов, в каждом число из сводки"],
 "invalidation": "цена, при которой сценарий сломан, и почему",
 "entry": число или null, "stop": число или null, "target": число или null,
 "data_gaps": ["чего не хватило для уверенного вывода"]}
Для «нейтрально» entry, stop и target равны null.

Пример формата:
{"prob_up": 62, "direction": "лонг", "agree_with_den": true, "summary": "1h и 4h за рост, 1d нейтрален, режим трендовый. Лонг-сигналы на 4h угадывали 58% при базе 51%.", "key_factors": ["4h: за лонг 72%", "отскок от поддержки 2318.40", "бэктест 4h: 58% против базы 51%"], "invalidation": "закрытие 4h ниже 2305.00 (поддержка 2318.40 минус 0.5 ATR 4h)", "entry": 2321.00, "stop": 2305.00, "target": 2352.00, "data_gaps": ["нет календаря событий"]}"""


def system_prompt(horizon_h: int = HORIZON_H) -> str:
    return SYSTEM_PROMPT.replace("{H}", str(horizon_h))


class AIError(RuntimeError):
    pass


@dataclass
class AIResult:
    direction: str
    confidence: str
    summary: str
    key_factors: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)   # что сломает сценарий (invalidation)
    levels: str = ""
    model: str = ""
    raw: str = ""
    prob_up: float | None = None
    agree_with_den: bool | None = None
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    data_gaps: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)   # что не прошло проверку
    samples: list[float] = field(default_factory=list)  # prob_up каждого запроса
    horizon_h: int = HORIZON_H

    @property
    def reliable(self) -> bool:
        return self.prob_up is not None and not self.issues


def provider(name: str | None = None) -> Provider:
    """Текущий провайдер ИИ: аргумент, иначе DEN_AI_PROVIDER, иначе Tooken Club."""
    name = (name or os.environ.get("DEN_AI_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise AIError(f"Неизвестный провайдер ИИ «{name}». Возможные: {', '.join(PROVIDERS)}.")
    return PROVIDERS[name]


def api_key(prov: Provider | None = None) -> str | None:
    prov = prov or provider()
    for env in prov.key_envs:
        v = (os.environ.get(env) or "").strip()
        if v:
            return v
    return None


def model_name(prov: Provider | None = None) -> str:
    prov = prov or provider()
    return (os.environ.get(prov.model_env) or prov.default_model).strip()


def samples_count() -> int:
    try:
        return max(1, min(5, int(os.environ.get("DEN_AI_SAMPLES", "3"))))
    except ValueError:
        return 3


def _f(x: float) -> str:
    return f"{x:,.4f}" if abs(x) < 10 else f"{x:,.2f}"


def _norm_title(t: str) -> str:
    return re.sub(r"[^\w ]+", "", t.lower()).strip()


def dedupe_news(items: list) -> list:
    """Один заголовок из нескольких источников оставляем один раз (самый свежий)."""
    seen, out = set(), []
    for n in sorted(items, key=lambda n: n.published, reverse=True):
        k = _norm_title(n.title)
        if k and k not in seen:
            seen.add(k)
            out.append(n)
    return out


def pick_news(items: list, now: datetime, max_news: int = 10, max_age_h: float = 24) -> list:
    """Новости за сутки без дублей, самые значимые (релевантность × сила тона), потом по времени."""
    fresh = [n for n in dedupe_news(items) if (now - n.published).total_seconds() <= max_age_h * 3600]
    fresh.sort(key=lambda n: getattr(n, "relevance", 1.0) * abs(n.score), reverse=True)
    return sorted(fresh[:max_news], key=lambda n: n.published, reverse=True)


def _family_lines(votes) -> list[str]:
    from .analysis import FAMILY_WORD, family_of

    fams: dict[str, list] = {}
    for v in votes:
        fams.setdefault(family_of(v), []).append(v)
    out = []
    for fam, vs in fams.items():
        w = sum(v.weight for v in vs) or 1.0
        s = sum(v.score * v.weight for v in vs) / w
        top = max(vs, key=lambda v: abs(v.score) * v.weight)
        out.append(f"- {FAMILY_WORD.get(fam, fam)}: {s:+.2f} ({len(vs)} инд.), сильнейший {top.name} "
                   f"{top.score:+.2f}: {top.reason}")
    return out


def allowed_levels(results: dict, ranges: list[dict] | None = None) -> list[float]:
    """Цены, на которые модели разрешено ссылаться: уровни, план DEN и границы диапазонов."""
    out = []
    for r in results.values():
        out.append(r.price)
        for xs in (r.levels or {}).values():
            out += list(xs)
        out += [x for x in (r.trade or {}).values() if isinstance(x, (int, float))]
    for g in ranges or []:
        out += [g.get(k) for k in ("68%: от", "68%: до", "95%: от", "95%: до") if g.get(k) is not None]
    return [float(x) for x in out]


def build_context(asset: str, contract: str | None, results: dict, total_score: float, total_label: str,
                  verdict=None, ranges: list[dict] | None = None, backtests: dict | None = None,
                  news: list | None = None, news_sentiment: dict | None = None, max_news: int = 10,
                  signal=None, backdrop=None, now: datetime | None = None, asset_code: str | None = None,
                  last_bar: datetime | None = None, horizon_h: int = HORIZON_H) -> str:
    """Компактная сводка для модели. Порядок важен: время и данные сначала, сигнал DEN в конце,
    чтобы модель сначала оценила сама и не повторяла готовый вывод."""
    now = now or datetime.now(timezone.utc)
    lines = [f"Актив: {asset}" + (f" (контракт {contract})" if contract else ""),
             f"Сейчас: {now:%Y-%m-%d %H:%M} UTC, {['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс'][now.weekday()]}. "
             f"Горизонт прогноза: {horizon_h} ч."]
    if asset_code:
        from .sessions import SESSION_WORD, in_session

        if asset_code in SESSION_WORD:
            lines.append(f"Основная сессия ({SESSION_WORD[asset_code]}): "
                         + ("открыта" if in_session(now, asset_code) else "закрыта, рынок тонкий"))
    if last_bar is not None:
        lb = last_bar if last_bar.tzinfo else last_bar.replace(tzinfo=timezone.utc)
        lines.append(f"Последняя свеча: {lb:%H:%M} UTC ({max(0, (now - lb).total_seconds()) / 60:.0f} мин назад)")
    any_res = next(iter(results.values()), None)
    if any_res is not None:
        lines.append(f"Цена: {_f(any_res.price)}")
    lines.append(f"Общий консенсус индикаторов по всем таймфреймам: {total_label} "
                 f"(балл {total_score:+.2f}, от -1 до +1)")
    for tf, r in results.items():
        lines.append(f"\n## Таймфрейм {tf}: {r.label}, за лонг {r.long_pct:.0f}%, "
                     f"изменение за свечу {r.change_pct:+.2f}%, ATR {_f(r.atr)}"
                     + (f", режим рынка: {r.regime.label.lower()} ({r.regime.reason})"
                        if getattr(r, "regime", None) else ""))
        lines += _family_lines(r.votes)
        if r.levels:
            lines.append("Сопротивления: " + (", ".join(_f(x) for x in r.levels.get("resistance", [])) or "нет"))
            lines.append("Поддержки: " + (", ".join(_f(x) for x in r.levels.get("support", [])) or "нет"))
    if ranges:
        lines.append("\n## Ожидаемый диапазон цены по волатильности")
        for r in ranges:
            lines.append(f"- {r['Горизонт']}: 68% {_f(r['68%: от'])}–{_f(r['68%: до'])}, "
                         f"95% {_f(r['95%: от'])}–{_f(r['95%: до'])}, ход ±{r['Ход ±%']:.2f}%")
    if backtests:
        lines.append("\n## Как сигналы индикаторов срабатывали на истории")
        for name, b in backtests.items():
            lo, sh = b["long"], b["short"]
            lines.append(f"- через {name}: лонг {lo['n']} сигналов, угадал "
                         + (f"{lo['hit']:.0%}" if lo["n"] else "–")
                         + f"; шорт {sh['n']} сигналов, угадал " + (f"{sh['hit']:.0%}" if sh["n"] else "–")
                         + f"; рост без сигнала (база) {b['base_up']:.0%}")
    if backdrop is not None:
        lines.append(f"\n## Геополитический фон: {backdrop.label}")
        lines += [f"- {x}" for x in backdrop.reasons]
    picked = pick_news(news or [], now, max_news)
    if picked:
        head = f"\n## Главные новости за 24 ч ({len(picked)} из {len(news)}, новые сверху)"
        if news_sentiment:
            head += f", общий тон {news_sentiment['score']:+.2f}"
        lines.append(head)
        for n in picked:
            age = (now - n.published).total_seconds() / 3600
            lines.append(f"- {age:.0f} ч назад, {n.source}, тон {n.score:+.1f}: {n.title}")
    if signal is not None or verdict is not None:
        lines.append("\n## Для сравнения (не копировать): вывод системы DEN")
    if signal is not None:
        lines.append(f"Итоговый сигнал: {signal.label}, уверенность {signal.confidence:.0f}% "
                     f"(порог {signal.threshold:.0f}%)")
        lines += [f"- {x}" for x in signal.reasons]
    if verdict is not None:
        lines.append(f"Вердикт по индикаторам и новостям: {verdict.label}, уверенность {verdict.confidence}")
    if any_res is not None and any_res.trade:
        lines.append("План DEN: " + ", ".join(f"{k} {_f(x)}" for k, x in any_res.trade.items()
                                              if isinstance(x, (int, float))))
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    candidate = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    if not candidate:
        raise ValueError("в ответе нет JSON")
    return json.loads(candidate)


def _num(x) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(str(x).replace(",", "").replace(" ", ""))
    except ValueError:
        return None
    return v if v == v else None


def direction_of(prob_up: float | None) -> str:
    if prob_up is None:
        return "нейтрально"
    return "лонг" if prob_up >= LONG_FROM else "шорт" if prob_up <= SHORT_TO else "нейтрально"


def confidence_of(prob_up: float | None) -> str:
    if prob_up is None:
        return "низкая"
    d = abs(prob_up - 50)
    return "высокая" if d >= 20 else "средняя" if d >= 10 else "низкая"


def parse_answer(text: str, model: str = "") -> AIResult:
    """Разобрать ответ модели. Направление и уверенность выводятся из prob_up; если его нет,
    берётся direction модели. Если JSON не получился, текст показывается как есть."""
    try:
        d = _extract_json(text)
    except (ValueError, json.JSONDecodeError):
        return AIResult("нейтрально", "низкая", text.strip() or "Модель вернула пустой ответ.", model=model, raw=text,
                        issues=["ответ не в формате JSON"])

    def as_list(x) -> list[str]:
        if isinstance(x, str):
            return [x] if x.strip() else []
        return [str(i) for i in (x or [])]

    issues = []
    prob = _num(d.get("prob_up"))
    if prob is not None:
        if 0 < prob <= 1 and not float(prob).is_integer():
            prob = round(prob * 100, 1)  # модель ответила долей вместо процентов
        if not 0 <= prob <= 100:
            issues.append(f"prob_up вне 0–100: {prob:g}")
            prob = max(0.0, min(100.0, prob))
    said = str(d.get("direction", "")).lower().strip()
    said = "лонг" if "long" in said or "лонг" in said else "шорт" if "short" in said or "шорт" in said else \
        "нейтрально" if said else ""
    if prob is None:
        direction = said or "нейтрально"
        confidence = str(d.get("confidence", "низкая")).lower().strip()
        issues.append("нет prob_up")
    else:
        direction, confidence = direction_of(prob), confidence_of(prob)
        if said and said != direction:
            issues.append(f"направление «{said}» не совпадает с prob_up {prob:.0f}")
    agree = d.get("agree_with_den")
    entry, stop, target = (_num(d.get(k)) for k in ("entry", "stop", "target"))
    levels = str(d.get("levels", "") or "").strip()
    if not levels and any(x is not None for x in (entry, stop, target)):
        levels = ", ".join(f"{w} {_f(x)}" for w, x in (("вход", entry), ("стоп", stop), ("цель", target))
                           if x is not None)
    risks = as_list(d.get("risks")) + as_list(d.get("invalidation"))
    return AIResult(direction=direction, confidence=confidence, summary=str(d.get("summary", "")).strip(),
                    key_factors=as_list(d.get("key_factors")), risks=risks, levels=levels, model=model, raw=text,
                    prob_up=prob, agree_with_den=agree if isinstance(agree, bool) else None,
                    entry=entry, stop=stop, target=target, data_gaps=as_list(d.get("data_gaps")), issues=issues)


def validate(res: AIResult, price: float | None, levels: list[float], atrs: list[float]) -> list[str]:
    """Проверка ответа: уровни взяты из сводки или равны цене ± k·ATR (k кратно 0.25), стоп и цель
    по правильные стороны от входа. Возвращает список проблем, пустой если всё в порядке."""
    issues = list(res.issues)
    pts = {"вход": res.entry, "стоп": res.stop, "цель": res.target}
    if res.direction == "нейтрально":
        return issues
    if any(v is None for v in pts.values()):
        issues.append("для сделки нет входа, стопа или цели")
        return issues
    if not price:
        return issues
    atrs = [a for a in atrs if a and a > 0]
    tol = price * 0.0005
    anchors = [(price, 6.0)] + [(lv, 1.0) for lv in levels]  # (опорная цена, сколько ATR можно отступить)

    def explained(x: float) -> bool:
        if any(abs(x - lv) <= tol for lv in levels):
            return True
        for base, max_k in anchors:
            for a in atrs:
                k = (x - base) / a
                if abs(k) <= max_k and abs(k - round(k * 4) / 4) * a <= max(price * 0.0001, 0.02 * a):
                    return True
        return False

    for name, x in pts.items():
        if not explained(x):
            issues.append(f"{name} {_f(x)} не из сводки и не цена или уровень ± k·ATR")
    e, s, t = res.entry, res.stop, res.target
    if res.direction == "лонг" and not (s < e < t):
        issues.append("для лонга нужно стоп < вход < цель")
    if res.direction == "шорт" and not (t < e < s):
        issues.append("для шорта нужно цель < вход < стоп")
    return issues


def _is_reasoning_model(model: str) -> bool:
    """GPT-5/6 и o-серия принимают max_completion_tokens и не принимают temperature."""
    m = model.lower().split("/")[-1]
    return bool(re.match(r"(gpt-[5-9]|o\d)", m))


def _chat(messages: list[dict], key: str | None = None, model: str | None = None, timeout: float = 180,
          max_tokens: int = 4000, temperature: float | None = 0.3) -> tuple[str, str]:
    """Один запрос к текущему провайдеру. Возвращает (текст ответа, имя модели, которая ответила)."""
    import requests

    prov = provider()
    name = prov.label
    key = key or api_key(prov)
    if not key:
        raise AIError(f"Нет ключа {name}. Задайте {prov.key_envs[0]} или введите ключ в боковой панели.")
    model = model or model_name(prov)
    if not model:
        raise AIError(f"Не указана модель {name}. Задайте {prov.model_env} или введите имя модели "
                      "в боковой панели.")
    proxies = {"https": os.environ["DEN_PROXY"]} if os.environ.get("DEN_PROXY") else None
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if prov.name == "openrouter":
        headers["X-Title"] = "DEN"
    body = {"model": model, "messages": messages}
    if _is_reasoning_model(model):
        body["max_completion_tokens"] = max_tokens * 2  # часть токенов уходит на рассуждение модели
    else:
        body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
    try:
        r = requests.post(prov.url, timeout=timeout, proxies=proxies, headers=headers, json=body)
    except requests.RequestException as e:
        raise AIError(f"{name} недоступен: {e}") from e
    if r.status_code == 401:
        raise AIError(f"{name} не принял ключ (401). Проверьте {prov.key_envs[0]}.")
    if r.status_code == 402:
        raise AIError(f"{name}: на балансе не хватает средств (402). Пополните баланс.")
    if r.status_code == 429:
        if prov.name == "openrouter":
            raise AIError("Лимит бесплатных запросов OpenRouter исчерпан (429). Подождите минуту "
                          "или до завтра; после разовой покупки кредитов на $10 лимит вырастет до 1000 в день.")
        raise AIError(f"Лимит запросов {name} исчерпан (429). Подождите и повторите.")
    try:
        data = r.json()
    except ValueError:
        raise AIError(f"{name} вернул {r.status_code}: {r.text[:300]}") from None
    if r.status_code >= 400 or (isinstance(data, dict) and data.get("error")):
        err = data.get("error") if isinstance(data, dict) else None
        msg = err.get("message") if isinstance(err, dict) else err
        if r.status_code == 404 or "model" in str(msg).lower() and "not" in str(msg).lower():
            raise AIError(f"{name} не знает модель «{model}»: {msg}. Проверьте точное имя в личном кабинете "
                          f"и задайте его в {prov.model_env}.")
        raise AIError(f"{name} вернул ошибку {r.status_code}: {msg or r.text[:300]}")
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise AIError(f"Неожиданный ответ {name}: {str(data)[:300]}") from None
    return msg.get("content") or msg.get("reasoning") or "", data.get("model", model)


def _one_sample(messages, key, model, timeout, check) -> AIResult:
    """Запрос с проверкой; если ответ не прошёл проверку, один повтор с перечнем ошибок."""
    text, used = _chat(messages, key, model, timeout, temperature=0.2)
    res = parse_answer(text, used)
    res.issues = check(res)
    if res.issues:
        fix = messages + [{"role": "assistant", "content": text},
                          {"role": "user", "content": "Ответ не прошёл проверку: " + "; ".join(res.issues)
                           + ". Исправь и верни только JSON в том же формате."}]
        text2, used2 = _chat(fix, key, model, timeout, temperature=0.2)
        res2 = parse_answer(text2, used2)
        res2.issues = check(res2)
        if len(res2.issues) <= len(res.issues):
            res = res2
    return res


def analyze_with_ai(context: str, key: str | None = None, model: str | None = None, timeout: float = 180,
                    samples: int | None = None, price: float | None = None, levels: list[float] | None = None,
                    atrs: list[float] | None = None, horizon_h: int = HORIZON_H) -> AIResult:
    """Несколько независимых запросов (параллельно), итог по медиане prob_up среди прошедших проверку.
    Текст разбора берётся из ответа, чей prob_up ближе всего к медиане."""
    n = samples or samples_count()
    msgs = [{"role": "system", "content": system_prompt(horizon_h)}, {"role": "user", "content": context}]

    def check(r: AIResult) -> list[str]:
        return validate(r, price, levels or [], atrs or [])

    if n == 1:
        outs = [_one_sample(msgs, key, model, timeout, check)]
    else:
        with ThreadPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(_one_sample, msgs, key, model, timeout, check) for _ in range(n)]
            outs, errs = [], []
            for f in futs:
                try:
                    outs.append(f.result())
                except AIError as e:
                    errs.append(e)
            if not outs:
                raise errs[0]
    with_prob = [r for r in outs if r.prob_up is not None]
    good = [r for r in with_prob if not r.issues] or with_prob
    if not good:
        best = outs[0]
        best.samples = []
        return best
    med = statistics.median(r.prob_up for r in good)
    best = min(good, key=lambda r: abs(r.prob_up - med))
    best.samples = [r.prob_up for r in with_prob]
    best.prob_up = float(med)
    best.direction, best.confidence = direction_of(med), confidence_of(med)
    if best.direction == "нейтрально":
        best.entry = best.stop = best.target = None
        best.levels = ""
    if len(good) < len(outs):
        best.issues = best.issues + [f"{len(outs) - len(good)} из {len(outs)} ответов не прошли проверку"]
    best.horizon_h = horizon_h
    return best


ASSET_TITLES = {"GOLD": "золото (XAU)", "SILVER": "серебро (XAG)", "AAPL": "акции Apple (AAPL)"}

NEWS_PROMPT = """Ты оцениваешь, как новостные заголовки повлияют на цену актива в ближайшие часы и сутки.
Для каждого заголовка дай два числа:
- relevance от 0 до 1: насколько новость прямо касается этого актива или его главных драйверов
  (для металлов: ставки ФРС, доходности, доллар, геополитика, спрос центробанков; для Apple: отчёты,
  продажи, прогнозы, рейтинги аналитиков, суды, регуляторы, пошлины);
- impact от -1 до 1: -1 сильно давит на цену вниз, 0 не влияет, 1 сильно толкает вверх.
Учитывай смысл, а не отдельные слова. Новость старше 12 ч, скорее всего, уже в цене: уменьшай impact.
Если relevance ниже 0.3, impact MUST быть 0.

Примеры:
- «Fed signals more rate cuts as inflation cools», 2 ч назад, золото → relevance 0.9, impact 0.6
- «Treasury yields jump after strong jobs report», 1 ч назад, серебро → relevance 0.8, impact -0.5
- «Gold jewelry demand in India slips during festival», 20 ч назад, золото → relevance 0.4, impact -0.1
- «Apple hit with EU antitrust fine», 3 ч назад, Apple → relevance 0.9, impact -0.4
- «Samsung unveils new foldable phone», 5 ч назад, Apple → relevance 0.3, impact 0

Ответь строго одним JSON-объектом без пояснений:
{"items": [{"n": номер, "relevance": число, "impact": число}, ...]} ровно N элементов в том же порядке."""


def parse_scores(text: str, n: int) -> list[tuple[float, float]]:
    """Список (impact, relevance) по порядку. Понимает и старый формат {"scores": [...]}."""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    try:
        d = _extract_json(clean)
    except (ValueError, json.JSONDecodeError):
        d = None
    pairs = None
    if isinstance(d, dict) and isinstance(d.get("items"), list):
        rows = {}
        for pos, row in enumerate(d["items"]):
            if not isinstance(row, dict):
                continue
            try:
                i = int(row.get("n", pos + 1)) - 1
            except (TypeError, ValueError):
                i = pos
            imp, rel = _num(row.get("impact")), _num(row.get("relevance"))
            if 0 <= i < n and imp is not None:
                rows[i] = (imp, 1.0 if rel is None else rel)
        if len(rows) == n:
            pairs = [rows[i] for i in range(n)]
        else:
            raise ValueError(f"ожидалось {n} оценок, пришло {len(rows)}")
    elif isinstance(d, dict) and isinstance(d.get("scores"), list):
        pairs = [(_num(v) or 0.0, 1.0) for v in d["scores"]]
    else:
        # Сломанный JSON: берём числа из первого списка [...] после "scores".
        m = re.search(r"scores\W*\[([^\]]*)", clean)
        if m:
            pairs = [(float(x), 1.0) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))]
    if not isinstance(pairs, list) or len(pairs) != n:
        raise ValueError(f"ожидалось {n} оценок, пришло {len(pairs) if isinstance(pairs, list) else 0}")
    out = []
    for imp, rel in pairs:
        rel = max(0.0, min(1.0, float(rel)))
        imp = 0.0 if rel < 0.3 else max(-1.0, min(1.0, float(imp)))
        out.append((imp, rel))
    return out


def score_headlines(titles: list[str], asset: str, key: str | None = None, model: str | None = None,
                    timeout: float = 120, ages_h: list[float] | None = None) -> tuple[list[tuple[float, float]], str]:
    """Тон и релевантность заголовков одним запросом. Возвращает ([(impact, relevance)], модель)."""
    if not titles:
        return [], model or model_name()
    ages = ages_h or [None] * len(titles)
    body = f"Актив: {ASSET_TITLES.get(asset, asset)}. Заголовков: {len(titles)}.\n" + \
        "\n".join(f"{i + 1}. {t}" + (f" ({a:.0f} ч назад)" if a is not None else "")
                  for i, (t, a) in enumerate(zip(titles, ages)))
    text, used = _chat([{"role": "system", "content": NEWS_PROMPT.replace("ровно N", f"ровно {len(titles)}")},
                        {"role": "user", "content": body}], key, model, timeout,
                       max_tokens=2000, temperature=0.0)
    try:
        return parse_scores(text, len(titles)), used
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        raise AIError(f"Модель вернула оценки новостей в неожиданном виде: {e}") from None


def rescore_news(items: list, asset: str, key: str | None = None, model: str | None = None,
                 now: datetime | None = None) -> tuple[list, str]:
    """Новости с тоном и релевантностью от модели. Дубли заголовков уходят в модель один раз."""
    from dataclasses import replace

    now = now or datetime.now(timezone.utc)
    uniq: dict[str, object] = {}
    for n in items:
        uniq.setdefault(_norm_title(n.title), n)
    keys = list(uniq)
    ages = [max(0.0, (now - uniq[k].published).total_seconds() / 3600) for k in keys]
    pairs, used = score_headlines([uniq[k].title for k in keys], asset, key, model, ages_h=ages)
    by_key = dict(zip(keys, pairs))
    return [replace(n, score=by_key[_norm_title(n.title)][0], relevance=by_key[_norm_title(n.title)][1])
            for n in items], used


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
        out = []
        for n in items:
            k = known.get(n.title)
            out.append(replace(n, score=k[0], relevance=k[1]) if k else n)
        return out, entry[2]
    out, used = rescore_news(items, asset, model=model)
    store[slot] = (now, {n.title: (n.score, n.relevance) for n in out}, used)
    return out, used
