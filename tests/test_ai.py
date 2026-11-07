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
    ctx = ai.build_context("Золото", "XAUTUSDT", results, score, label, verdict=verdict(score, 0.8, None),
                           ranges=price_ranges(make(0.0)), news=news, news_sentiment={"score": 0.8})
    for part in ("Золото", "XAUTUSDT", "Таймфрейм 15m", "Таймфрейм 1h", "MACD", "Вердикт DEN",
                 "Ожидаемый диапазон", "Gold surges on rate cut bets"):
        assert part in ctx


def test_parse_json_variants():
    body = '{"direction": "лонг", "confidence": "средняя", "summary": "Тренд вверх.", ' \
           '"key_factors": ["EMA"], "risks": "ФРС", "levels": "стоп 2000"}'
    for text in (body, f"<think>хм</think>\n```json\n{body}\n```", f"Вот ответ: {body} Удачи"):
        r = ai.parse_answer(text, "m")
        assert r.direction == "лонг" and r.confidence == "средняя" and r.risks == ["ФРС"] and r.model == "m"
    assert ai.parse_answer('{"direction": "Short"}').direction == "шорт"


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

    sent = {}

    def ok(url, **kw):
        sent.update(kw)
        return _Resp(200, {"model": "x/y:free",
                           "choices": [{"message": {"content": '{"direction": "шорт", "confidence": "низкая"}'}}]})

    monkeypatch.setattr(requests, "post", ok)
    r = ai.analyze_with_ai("ctx", key="k", model="openrouter/free")
    assert r.direction == "шорт" and r.model == "x/y:free"
    assert sent["headers"]["Authorization"] == "Bearer k" and sent["json"]["model"] == "openrouter/free"

    monkeypatch.setattr(requests, "post", lambda url, **kw: _Resp(429, {"error": {"message": "rate"}}))
    with pytest.raises(ai.AIError, match="Лимит"):
        ai.analyze_with_ai("ctx", key="k")

    monkeypatch.delenv("DEN_OPENROUTER_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ai.AIError, match="Нет ключа"):
        ai.analyze_with_ai("ctx")


def test_score_headlines_parses_and_clips(monkeypatch):
    sent = {}

    def fake_chat(messages, key=None, model=None, timeout=0, max_tokens=0, temperature=0):
        sent["user"] = messages[1]["content"]
        return '```json\n{"scores": [0.8, -2, 0]}\n```', "m:free"
    monkeypatch.setattr(ai, "_chat", fake_chat)
    scores, used = ai.score_headlines(["Gold jumps", "Yields soar", "Weather"], "GOLD")
    assert scores == [0.8, -1.0, 0.0] and used == "m:free"
    assert "золото" in sent["user"] and "3. Weather" in sent["user"]


def test_score_headlines_wrong_length(monkeypatch):
    monkeypatch.setattr(ai, "_chat", lambda *a, **k: ('{"scores": [0.1]}', "m"))
    with pytest.raises(ai.AIError, match="ожидалось 2"):
        ai.score_headlines(["a", "b"], "AAPL")


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
        ai.analyze_with_ai("ctx")
    monkeypatch.setenv("DEN_TOOKEN_KEY", "tk")
    with pytest.raises(ai.AIError, match="Не указана модель"):
        ai.analyze_with_ai("ctx")
    monkeypatch.setenv("DEN_TOOKEN_MODEL", "deepseek-chat")
    r = ai.analyze_with_ai("ctx")
    assert r.direction == "лонг" and r.model == "deepseek-chat"
    assert sent["url"] == "https://tooken.club/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer tk" and "X-Title" not in sent["headers"]
    monkeypatch.setattr(requests, "post", lambda url, **kw: _Resp(402, {"error": {"message": "balance"}}))
    with pytest.raises(ai.AIError, match="баланс"):
        ai.analyze_with_ai("ctx")
    monkeypatch.setenv("DEN_AI_PROVIDER", "nope")
    with pytest.raises(ai.AIError, match="Неизвестный провайдер"):
        ai.provider()
