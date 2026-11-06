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


def _setup(monkeypatch, tmp_path):
    monkeypatch.setenv("DEN_HOME", str(tmp_path))
    for k in ("DEN_ALERT_STATE", "DEN_ALERT_CONFIG", "DEN_TG_TOKEN", "DEN_TG_CHAT"):
        monkeypatch.delenv(k, raising=False)
    alerts.save_config(alerts.AlertConfig(enabled=True, assets=["BYBIT:GOLD"]))
    monkeypatch.setattr(alerts, "geo_risk", lambda sym: None)  # без сети


def test_disabled_by_default_and_toggle(monkeypatch, tmp_path):
    monkeypatch.setenv("DEN_HOME", str(tmp_path))
    monkeypatch.delenv("DEN_ALERT_CONFIG", raising=False)
    assert alerts.load_config().enabled is False
    assert alerts.check_once(send=lambda m: 1 / 0, log=lambda *_: None) == 0  # выключено: ничего не делает
    alerts.main(["--on"])
    assert alerts.load_config().enabled is True
    alerts.main(["--off"])
    assert alerts.load_config().enabled is False


def test_config_roundtrip_and_env_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("DEN_HOME", str(tmp_path))
    monkeypatch.setenv("DEN_TG_TOKEN", "from-env")
    alerts.save_config(alerts.AlertConfig(enabled=True, chat_id="7", assets=["BYBIT:AAPL"], on_agree=False,
                                          agree_tfs=["15m", "1h"], interval=900))
    c = alerts.load_config()
    assert (c.enabled, c.chat_id, c.assets, c.on_agree, c.agree_tfs, c.interval) == \
        (True, "7", ["BYBIT:AAPL"], False, ["15m", "1h"], 900)
    assert c.token == "from-env"


def test_events_can_be_turned_off():
    prev = {"dir": 1, "agree": 0}
    cur = {"dir": -1, "agree": -1}
    assert alerts.detect_events(prev, cur, on_consensus=False) == ["agree"]
    assert alerts.detect_events(prev, cur, on_agree=False) == ["consensus"]


def test_lock_prevents_double_sending(monkeypatch, tmp_path):
    monkeypatch.setenv("DEN_HOME", str(tmp_path))
    assert alerts.claim_lock("dashboard", ttl=600)
    assert not alerts.claim_lock("script", ttl=600)
    assert alerts.claim_lock("dashboard", ttl=600)
    assert alerts.claim_lock("script", ttl=0)  # замок устарел


def test_check_once_sends_on_change(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
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
    _setup(monkeypatch, tmp_path)
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


def test_signal_event_and_message():
    from den.signals import evaluate

    cur, res = _snap({"5m": 0.8, "15m": 0.8, "1h": 0.8, "1d": 0.8})
    sig = evaluate(res, make(0.8, seed=1))
    score, label = cur["score"], cur["label"]
    snap = alerts.snapshot(res, score, label, ["5m", "15m", "1h"], sig)
    assert snap["signal"] == 1 and "подтверждён" in snap["confirm"]
    prev = dict(snap, signal=0)
    assert "signal" in alerts.detect_events(prev, snap, on_signal=True)
    assert "signal" not in alerts.detect_events(prev, snap, on_signal=False)
    msg = alerts.format_message("BYBIT:GOLD", snap, prev, ["signal"], ["5m", "15m", "1h"], res)
    assert "уверенность" in msg and "Режим рынка" in msg


def test_token_not_leaked_in_errors(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("Max retries exceeded with url: /bot123:SECRET/sendMessage")
    monkeypatch.setattr(requests, "post", boom)
    try:
        alerts.tg_call("getMe", token="123:SECRET")
    except alerts.TelegramError as e:
        assert "SECRET" not in str(e) and e.__cause__ is None
    else:
        raise AssertionError("ожидалась ошибка")


def test_config_file_private(monkeypatch, tmp_path):
    import os
    import stat

    monkeypatch.setenv("DEN_HOME", str(tmp_path))
    monkeypatch.delenv("DEN_ALERT_CONFIG", raising=False)
    alerts.save_config(alerts.AlertConfig(token="T"))
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(alerts.config_path()).st_mode) == 0o600
