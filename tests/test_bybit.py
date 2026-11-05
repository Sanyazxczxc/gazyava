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
