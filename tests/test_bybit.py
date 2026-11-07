import pytest

from den import data


def test_parse_bybit_drops_open_candle():
    res = {"symbol": "XAUTUSDT", "list": [  # Bybit отдаёт от новых к старым
        ["1762336800000", "2652", "2654", "2651", "2653", "10", "0"],
        ["1762336500000", "2651", "2653", "2650", "2652", "12", "0"],
        ["1762336200000", "2650", "2652", "2649", "2651", "8", "0"]]}
    df = data.parse_bybit(res)
    assert len(df) == 2 and df.index.is_monotonic_increasing
    assert df.iloc[-1]["close"] == 2652


def _fake(linear, spot):
    def get(path, p):
        syms = linear if p["category"] == "linear" else spot
        return {"list": [{"symbol": x, "status": "Trading"} for x in syms], "nextPageCursor": ""}
    return get


def test_symbol_prefers_futures_candidate(monkeypatch):
    data._bybit_symbol_cache.clear()
    monkeypatch.setattr(data, "_bybit_get", _fake(["BTCUSDT", "XAUTUSDT", "PAXGUSDT"], []))
    assert data.bybit_symbol("GOLD") == ("linear", "XAUTUSDT")


def test_symbol_falls_back_to_spot_token(monkeypatch):
    data._bybit_symbol_cache.clear()
    monkeypatch.setattr(data, "_bybit_get", _fake(["BTCUSDT"], ["AAPLXUSDT", "ETHUSDT"]))
    assert data.bybit_symbol("AAPL") == ("spot", "AAPLXUSDT")


def test_symbol_keyword_search_and_missing(monkeypatch):
    data._bybit_symbol_cache.clear()
    monkeypatch.setattr(data, "_bybit_get", _fake(["XAGNEWUSDT"], []))
    assert data.bybit_symbol("SILVER") == ("linear", "XAGNEWUSDT")
    with pytest.raises(ValueError, match="AAPL"):
        data.bybit_symbol("AAPL")


def _kline_server(start_ms: int, n: int, step_ms: int = 900_000):
    """Фейковый Bybit: n свечей с шагом step_ms, отдаёт от новых к старым, не больше limit, до end."""
    calls = []

    def get(path, p):
        assert path == "kline"
        calls.append(p)
        end = p.get("end", start_ms + (n - 1) * step_ms)
        ts = [t for t in range(start_ms, start_ms + n * step_ms, step_ms) if t <= end][::-1][: p["limit"]]
        return {"symbol": p["symbol"], "list": [[str(t), "1", "2", "0.5", str(1 + t / 1e12), "10"] for t in ts]}
    return get, calls


def test_bybit_paginates_and_caches(monkeypatch, tmp_path):
    import pandas as pd

    monkeypatch.setenv("DEN_HOME", str(tmp_path))
    data._bybit_symbol_cache["GOLD"] = ("linear", "XAUTUSDT")
    start = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000)
    get, calls = _kline_server(start, 2500)
    monkeypatch.setattr(data, "_bybit_get", get)
    df = data.load_bybit("GOLD", "15m", bars=5000)
    assert len(df) == 2500 and df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert len(calls) == 3  # 1000 + 1000 + 500, дальше начало торгов
    # Повторный запуск: одна страница новых свечей, история из кэша, начало торгов запомнено.
    get2, calls2 = _kline_server(start, 2510)
    monkeypatch.setattr(data, "_bybit_get", get2)
    df2 = data.load_bybit("GOLD", "15m", bars=5000)
    assert len(df2) == 2510 and len(calls2) == 1
    data._bybit_symbol_cache.clear()


def test_bybit_drops_unclosed_candle(monkeypatch, tmp_path):
    import pandas as pd

    monkeypatch.setenv("DEN_HOME", str(tmp_path))
    data._bybit_symbol_cache["GOLD"] = ("linear", "XAUTUSDT")
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    start = int((now - pd.Timedelta(minutes=15 * 99)).timestamp() * 1000)
    get, _ = _kline_server(start, 100)  # последняя свеча открылась только что
    monkeypatch.setattr(data, "_bybit_get", get)
    df = data.load_bybit("GOLD", "15m", bars=500, use_cache=False)
    assert len(df) == 99 and df.index[-1] == now - pd.Timedelta(minutes=15)
    data._bybit_symbol_cache.clear()
