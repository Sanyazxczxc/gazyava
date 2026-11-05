import pandas as pd

from den import data, derivatives as dv
from den.analysis import analyze
from tests.test_analysis import make


def _series(vals, freq="15min"):
    idx = pd.date_range("2026-11-05", periods=len(vals), freq=freq, tz="UTC")
    return pd.Series(vals, index=idx, dtype=float)


def _d(funding=0.0001, oi=None, br=None):
    return dv.Derivatives("XAUUSDT", "15m", funding, funding,
                          oi if oi is not None else pd.Series(dtype=float),
                          br if br is not None else pd.Series(dtype=float))


def by_name(votes):
    return {v.name: v for v in votes}


def test_funding_is_contrarian():
    price = _series([100] * 20)
    assert by_name(dv.derivative_votes(_d(0.0008), price))["Funding"].score < -0.5
    assert by_name(dv.derivative_votes(_d(-0.0004), price))["Funding"].score > 0.5
    assert abs(by_name(dv.derivative_votes(_d(0.0001), price))["Funding"].score) < 0.01


def test_open_interest_confirms_price():
    up, down = _series(range(100, 120)), _series(range(120, 100, -1))
    oi_up, oi_down = _series([1000 * 1.01 ** i for i in range(20)]), _series([1000 * 0.99 ** i for i in range(20)])
    assert by_name(dv.derivative_votes(_d(None, oi_up), up))["Open interest"].score > 0.5     # новые лонги
    assert by_name(dv.derivative_votes(_d(None, oi_up), down))["Open interest"].score < -0.5  # новые шорты
    s = by_name(dv.derivative_votes(_d(None, oi_down), up))["Open interest"].score             # закрытие шортов
    assert 0 < s <= 0.4


def test_long_short_ratio_contrarian():
    price = _series([100] * 5)
    assert by_name(dv.derivative_votes(_d(None, br=_series([0.75])), price))["Long/Short"].score < -0.5
    assert by_name(dv.derivative_votes(_d(None, br=_series([0.52])), price))["Long/Short"].score == 0


def test_no_derivatives_no_votes():
    assert dv.derivative_votes(None, _series([1, 2])) == []


def test_extra_votes_shift_consensus():
    df = make(0.0, seed=5)
    base = analyze(df, "BYBIT:GOLD", "15m")
    crowded = analyze(df, "BYBIT:GOLD", "15m", extra_votes=dv.derivative_votes(_d(0.002), df["close"]))
    assert crowded.score < base.score and any(v.name == "Funding" for v in crowded.votes)


def test_load_parses_bybit(monkeypatch):
    data._bybit_symbol_cache.clear()
    data._bybit_symbol_cache["GOLD"] = ("linear", "XAUUSDT")
    ts = [str(1762336800000 - i * 900000) for i in range(3)]  # Bybit отдаёт от новых к старым
    answers = {
        "tickers": {"list": [{"symbol": "XAUUSDT", "fundingRate": "0.0003"}]},
        "funding/history": {"list": [{"fundingRate": "0.0001"}, {"fundingRate": "0.0002"}]},
        "open-interest": {"list": [{"openInterest": str(100 + i), "timestamp": t} for i, t in enumerate(ts)]},
        "account-ratio": {"list": [{"buyRatio": "0.6", "sellRatio": "0.4", "timestamp": ts[0]}]},
    }
    monkeypatch.setattr(dv, "_get", lambda path, params: answers[path])
    d = dv.load_derivatives("GOLD", "15m")
    assert d.funding == 0.0003 and abs(d.funding_avg - 0.00015) < 1e-12
    assert list(d.oi) == [102, 101, 100] and d.oi.index.is_monotonic_increasing
    assert d.buy_ratio.iloc[-1] == 0.6


def test_spot_token_has_no_derivatives():
    data._bybit_symbol_cache.clear()
    data._bybit_symbol_cache["AAPL"] = ("spot", "AAPLXUSDT")
    assert dv.load_derivatives("AAPL") is None
