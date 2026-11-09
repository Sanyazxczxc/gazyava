from datetime import datetime, timezone

import pytest

from den import ai
from den.analysis import analyze, multi_timeframe
from den.forecast import price_ranges, verdict
from den.news import NewsItem
from tests.test_analysis import make


def test_context_contains_key_data():
    results = {"15m": analyze(make(0.8), "GC=F", "15m"), "1h": analyze(make(0.5, seed=1), "GC=F", "1h")}
    score, label = multi_timeframe(results)
    news = [NewsItem("Gold surges on rate cut bets", "http://a", "Reuters",
                     datetime(2026, 11, 5, 10, tzinfo=timezone.utc), 0.8)]
    now = datetime(2026, 11, 5, 14, tzinfo=timezone.utc)
    old = NewsItem("Gold slips", "http://b", "FT", datetime(2026, 11, 3, tzinfo=timezone.utc), -0.5)
    ctx = ai.build_context("Золото", "XAUTUSDT", results, score, label, verdict=verdict(score, 0.8, None),
                           ranges=price_ranges(make(0.0)), news=news + [old], news_sentiment={"score": 0.8},
                           now=now, asset_code="GOLD", last_bar=datetime(2026, 11, 5, 13, 45))
    for part in ("Золото", "XAUTUSDT", "Таймфрейм 15m", "Таймфрейм 1h", "тренд и импульс", "Для сравнения",
                 "Ожидаемый диапазон", "Gold surges on rate cut bets", "14:00 UTC", "сессия", "открыта",
                 "15 мин назад", "4 ч назад"):
        assert part in ctx, part
    assert "Gold slips" not in ctx  # старше суток
    assert ctx.index("Ожидаемый диапазон") < ctx.index("Для сравнения")  # сигнал DEN в конце


def test_parse_json_variants():
    body = '{"direction": "лонг", "confidence": "средняя", "summary": "Тренд вверх.", ' \
           '"key_factors": ["EMA"], "risks": "ФРС", "levels": "стоп 2000"}'
    for text in (body, f"<think>хм</think>\n```json\n{body}\n```", f"Вот ответ: {body} Удачи"):
        r = ai.parse_answer(text, "m")
        assert r.direction == "лонг" and r.confidence == "средняя" and r.risks == ["ФРС"] and r.model == "m"
    assert ai.parse_answer('{"direction": "Short"}').direction == "шорт"


def test_parse_prob_sets_direction():
    r = ai.parse_answer('{"prob_up": 72, "direction": "шорт", "entry": "2,320.5", "stop": 2300, "target": 2360,'
                        ' "agree_with_den": false, "invalidation": "ниже 2300", "data_gaps": ["календарь"]}')
    assert r.prob_up == 72 and r.direction == "лонг" and r.confidence == "высокая"
    assert r.entry == 2320.5 and r.agree_with_den is False and r.risks == ["ниже 2300"]
    assert any("не совпадает" in x for x in r.issues)
    assert ai.parse_answer('{"prob_up": 0.55}').prob_up == 55
    assert ai.parse_answer('{"prob_up": 50}').direction == "нейтрально"


def test_validate_levels():
    ok = ai.AIResult("лонг", "средняя", "", prob_up=65, entry=2000.0, stop=1990.0, target=2025.0)
    assert ai.validate(ok, 2000.0, [1990.0], [10.0]) == []  # стоп уровень, цель цена + 2.5 ATR
    bad = ai.AIResult("лонг", "средняя", "", prob_up=65, entry=2000.0, stop=2013.0, target=2007.0)
    issues = ai.validate(bad, 2000.0, [1990.0], [10.0])
    assert any("цель" in x and "не из сводки" in x for x in issues) and any("стоп < вход" in x for x in issues)
    assert ai.validate(ai.AIResult("нейтрально", "низкая", "", prob_up=50), 2000.0, [], [10.0]) == []


def test_samples_median_and_retry(monkeypatch):
    answers = iter(['{"prob_up": 70, "entry": 2000, "stop": 1990, "target": 2020}',
                    '{"prob_up": 64, "entry": 2000, "stop": 1990, "target": 2020}',
                    '{"prob_up": 90, "entry": 2000, "stop": 2050, "target": 2020}',  # стоп не с той стороны
                    '{"prob_up": 61, "entry": 2000, "stop": 1990, "target": 2020}'])  # исправленный повтор
    calls = []

    def fake_chat(messages, key=None, model=None, timeout=0, max_tokens=0, temperature=0):
        calls.append(len(messages))
        return next(answers), "gpt-6-astra"
    monkeypatch.setattr(ai, "_chat", fake_chat)
    monkeypatch.setenv("DEN_AI_SAMPLES", "3")
    r = ai.analyze_with_ai("ctx", samples=3, price=2000.0, levels=[1990.0, 2020.0], atrs=[10.0])
    assert len(calls) == 4 and r.prob_up == 64 and r.direction == "лонг" and sorted(r.samples) == [61, 64, 70]
    assert not r.issues and r.reliable


def test_reasoning_model_payload(monkeypatch):
    import requests

    sent = {}

    def ok(url, **kw):
        sent.update(kw)
        return _Resp(200, {"model": "gpt-6-astra", "choices": [{"message": {"content": '{"prob_up": 50}'}}]})
    monkeypatch.setattr(requests, "post", ok)
    monkeypatch.setenv("DEN_AI_PROVIDER", "tooken")
    monkeypatch.delenv("DEN_TOOKEN_MODEL", raising=False)
    ai._chat([{"role": "user", "content": "x"}], key="k")
    assert sent["json"]["model"] == "gpt-6-astra" and "temperature" not in sent["json"]
    assert "max_completion_tokens" in sent["json"] and "max_tokens" not in sent["json"]


def test_parse_plain_text_fallback():
    r = ai.parse_answer("Скорее рост, но неуверенно.")
    assert r.direction == "нейтрально" and "рост" in r.summary


class _Resp:
    def __init__(self, status, data):
        self.status_code, self._data, self.text = status, data, str(data)

    def json(self):
        return self._data


def test_request_and_errors(monkeypatch):
    import requests

    monkeypatch.setenv("DEN_AI_PROVIDER", "openrouter")

    sent = {}

    def ok(url, **kw):
        sent.update(kw)
        return _Resp(200, {"model": "x/y:free",
                           "choices": [{"message": {"content": '{"direction": "шорт", "confidence": "низкая"}'}}]})

    monkeypatch.setattr(requests, "post", ok)
    r = ai.analyze_with_ai("ctx", key="k", model="openrouter/free", samples=1)
    assert r.direction == "шорт" and r.model == "x/y:free"
    assert sent["headers"]["Authorization"] == "Bearer k" and sent["json"]["model"] == "openrouter/free"

    monkeypatch.setattr(requests, "post", lambda url, **kw: _Resp(429, {"error": {"message": "rate"}}))
    with pytest.raises(ai.AIError, match="Лимит"):
        ai.analyze_with_ai("ctx", key="k", samples=1)

    monkeypatch.delenv("DEN_OPENROUTER_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ai.AIError, match="Нет ключа"):
        ai.analyze_with_ai("ctx", samples=1)


def test_score_headlines_parses_and_clips(monkeypatch):
    sent = {}

    def fake_chat(messages, key=None, model=None, timeout=0, max_tokens=0, temperature=0):
        sent["user"] = messages[1]["content"]
        return '```json\n{"scores": [0.8, -2, 0]}\n```', "m:free"
    monkeypatch.setattr(ai, "_chat", fake_chat)
    scores, used = ai.score_headlines(["Gold jumps", "Yields soar", "Weather"], "GOLD")
    assert scores == [(0.8, 1.0), (-1.0, 1.0), (0.0, 1.0)] and used == "m:free"  # старый формат тоже понимаем
    assert "золото" in sent["user"] and "3. Weather" in sent["user"]


def test_score_headlines_wrong_length(monkeypatch):
    monkeypatch.setattr(ai, "_chat", lambda *a, **k: ('{"scores": [0.1]}', "m"))
    with pytest.raises(ai.AIError, match="ожидалось 2"):
        ai.score_headlines(["a", "b"], "AAPL")


def test_score_headlines_items_format(monkeypatch):
    sent = {}

    def fake_chat(messages, key=None, model=None, timeout=0, max_tokens=0, temperature=0):
        sent["sys"], sent["user"] = messages[0]["content"], messages[1]["content"]
        return ('{"items": [{"n": 2, "relevance": 0.1, "impact": 0.9}, {"n": 1, "relevance": 0.8, "impact": 0.5}]}',
                "m")
    monkeypatch.setattr(ai, "_chat", fake_chat)
    pairs, _ = ai.score_headlines(["Fed cuts", "Phone launch"], "GOLD", ages_h=[2, 30])
    assert pairs == [(0.5, 0.8), (0.0, 0.1)]  # низкая релевантность обнуляет влияние
    assert "ровно 2" in sent["sys"] and "(2 ч назад)" in sent["user"]


def test_rescore_news_dedupes(monkeypatch):
    seen = []

    def fake(titles, asset, key=None, model=None, timeout=0, ages_h=None):
        seen.append(titles)
        return [(0.4, 0.9)] * len(titles), "m"
    monkeypatch.setattr(ai, "score_headlines", fake)
    t = datetime(2026, 11, 5, tzinfo=timezone.utc)
    items = [NewsItem("Gold rises!", "", "A", t, 0.0), NewsItem("gold rises", "", "B", t, 0.0)]
    out, _ = ai.rescore_news(items, "GOLD", now=t)
    assert len(seen[0]) == 1 and [n.score for n in out] == [0.4, 0.4] and out[1].relevance == 0.9


def test_rescore_news_replaces_scores(monkeypatch):
    monkeypatch.setattr(ai, "_chat", lambda *a, **k: ('{"scores": [-0.5]}', "m"))
    item = NewsItem("Apple faces EU fine", "http://a", "FT", datetime(2026, 11, 5, tzinfo=timezone.utc), 0.0)
    out, _ = ai.rescore_news([item], "AAPL")
    assert out[0].score == -0.5 and out[0].title == item.title and item.score == 0.0


def test_news_throttled(monkeypatch):
    calls = []

    def fake(items, asset, key=None, model=None):
        calls.append(1)
        from dataclasses import replace
        return [replace(n, score=0.9) for n in items], "m"
    monkeypatch.setattr(ai, "rescore_news", fake)
    t0 = datetime(2026, 11, 5, tzinfo=timezone.utc)
    a = NewsItem("Old", "", "", t0, 0.1)
    b = NewsItem("New", "", "", t0, -0.3)
    store = {}
    out, _ = ai.rescore_news_throttled([a], "GOLD", store, "m", now=0)
    out, _ = ai.rescore_news_throttled([a, b], "GOLD", store, "m", now=600)
    assert len(calls) == 1 and [n.score for n in out] == [0.9, -0.3]  # новый заголовок: словарь
    ai.rescore_news_throttled([a, b], "GOLD", store, "m", now=1900)
    assert len(calls) == 2


def test_tooken_provider(monkeypatch):
    import requests

    sent = {}

    def ok(url, **kw):
        sent.update(kw, url=url)
        return _Resp(200, {"model": "deepseek-chat",
                           "choices": [{"message": {"content": '{"direction": "лонг", "confidence": "средняя"}'}}]})

    monkeypatch.setenv("DEN_AI_PROVIDER", "tooken")
    monkeypatch.setenv("DEN_OPENROUTER_KEY", "or-key")
    monkeypatch.delenv("DEN_TOOKEN_KEY", raising=False)
    monkeypatch.delenv("DEN_TOOKEN_MODEL", raising=False)
    monkeypatch.setattr(requests, "post", ok)
    assert ai.api_key() is None  # ключ OpenRouter не уходит в Tooken Club
    with pytest.raises(ai.AIError, match="Нет ключа Tooken Club"):
        ai.analyze_with_ai("ctx", samples=1)
    monkeypatch.setenv("DEN_TOOKEN_KEY", "tk")
    assert ai.model_name() == "gpt-6-astra"  # модель по умолчанию закреплена
    monkeypatch.setenv("DEN_TOOKEN_MODEL", "deepseek-chat")
    r = ai.analyze_with_ai("ctx", samples=1)
    assert r.direction == "лонг" and r.model == "deepseek-chat"
    assert sent["url"] == "https://tooken.club/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer tk" and "X-Title" not in sent["headers"]
    monkeypatch.setattr(requests, "post", lambda url, **kw: _Resp(402, {"error": {"message": "balance"}}))
    with pytest.raises(ai.AIError, match="баланс"):
        ai.analyze_with_ai("ctx", samples=1)
    monkeypatch.setenv("DEN_AI_PROVIDER", "nope")
    with pytest.raises(ai.AIError, match="Неизвестный провайдер"):
        ai.provider()
