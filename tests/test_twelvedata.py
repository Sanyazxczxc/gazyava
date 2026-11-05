import pytest

from den.data import parse_twelvedata


def test_parse_metal_without_volume():
    js = {"status": "ok", "values": [
        {"datetime": "2026-11-05 10:05:00", "open": "2651", "high": "2653", "low": "2650", "close": "2652"},
        {"datetime": "2026-11-05 10:00:00", "open": "2650", "high": "2652", "low": "2649", "close": "2651"}]}
    df = parse_twelvedata(js, "XAU/USD")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing and df["volume"].sum() == 0
    assert df.iloc[-1]["close"] == 2652


def test_error_message():
    with pytest.raises(ValueError, match="apikey"):
        parse_twelvedata({"status": "error", "message": "Invalid apikey"}, "AAPL")
