from den import alerts
from den.analysis import analyze
from tests.test_analysis import make


def _snap(trends, agree=("5m", "15m", "1h")):
    results = {tf: analyze(make(t, seed=i), "BYBIT:GOLD", tf) for i, (tf, t) in enumerate(trends.items())}
    from den.analysis import multi_timeframe
    score, label = multi_timeframe(results)
    return alerts.snapshot(results, score, label, list(agree)), results


def test_first_check_is_silent_and_change_alerts():
    up, _ = _snap({"5m": 0.8, "15m": 0.8, "1h": 0.8, "1d": 0.8})
    down, _ = _snap({"5m": -0.8, "15m": -0.8, "1h": -0.8, "1d": -0.8})
    assert up["dir"] == 1 and up["agree"] == 1 and down["agree"] == -1
    assert alerts.detect_events(None, up) == []
    assert alerts.detect_events(up, up) == []
    assert alerts.detect_events(up, down) == ["consensus", "agree"]


def test_agreement_needs_all_timeframes():
    mixed, _ = _snap({"5m": 0.8, "15m": -0.8, "1h": 0.8, "1d": 0.8})
    assert mixed["agree"] == 0
    missing, _ = _snap({"15m": 0.8, "1h": 0.8})
    assert missing["agree"] == 0  # 5m не загрузился: совпадением не считаем


def test_message_text():
    prev, _ = _snap({"5m": -0.8, "15m": -0.8, "1h": -0.8, "1d": -0.8})
    cur, res = _snap({"5m": 0.8, "15m": 0.8, "1h": 0.8, "1d": 0.8})
    msg = alerts.format_message("BYBIT:GOLD", cur, prev, ["consensus", "agree"], ["5m", "15m", "1h"], res)
    assert "Золото" in msg and "консенсус сменился" in msg and "за лонг" in msg and "15m:" in msg


def test_check_once_sends_on_change(monkeypatch, tmp_path):
    monkeypatch.setenv("DEN_ALERT_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("DEN_ALERT_ASSETS", "BYBIT:GOLD")
    trend = {"t": 0.8}
    monkeypatch.setattr(alerts, "load_ohlcv", lambda sym, tf: make(trend["t"], seed=hash(tf) % 100))
    monkeypatch.setattr(alerts, "load_derivatives", lambda asset, tf: None)
    sent = []
    assert alerts.check_once(send=sent.append, log=lambda *_: None) == 0  # первая проверка: только запомнить
    assert alerts.check_once(send=sent.append, log=lambda *_: None) == 0  # без изменений
    trend["t"] = -0.8
    assert alerts.check_once(send=sent.append, log=lambda *_: None) == 1
    assert "консенсус сменился" in sent[0]


def test_failed_send_keeps_old_state(monkeypatch, tmp_path):
    monkeypatch.setenv("DEN_ALERT_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("DEN_ALERT_ASSETS", "BYBIT:GOLD")
    trend = {"t": 0.8}
    monkeypatch.setattr(alerts, "load_ohlcv", lambda sym, tf: make(trend["t"], seed=hash(tf) % 100))
    monkeypatch.setattr(alerts, "load_derivatives", lambda asset, tf: None)
    alerts.check_once(send=lambda m: None, log=lambda *_: None)
    trend["t"] = -0.8

    def fail(_):
        raise alerts.TelegramError("нет сети")
    alerts.check_once(send=fail, log=lambda *_: None)
    sent = []
    assert alerts.check_once(send=sent.append, log=lambda *_: None) == 1  # повтор после сбоя


def test_send_telegram_request(monkeypatch):
    import requests

    calls = {}

    class R:
        status_code = 200

        def json(self):
            return {"ok": True, "result": {}}

    def post(url, json, timeout, proxies):
        calls.update(url=url, body=json)
        return R()
    monkeypatch.setattr(requests, "post", post)
    alerts.send_telegram("привет", token="T", chat_id="42")
    assert calls["url"].endswith("/botT/sendMessage") and calls["body"]["chat_id"] == "42"
