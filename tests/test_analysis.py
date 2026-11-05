import numpy as np
import pandas as pd

from den.analysis import analyze, multi_timeframe


def make(trend: float, n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-01", periods=n, freq="15min")
    close = 2000 + np.cumsum(trend + rng.normal(0, 1, n))
    high = close + rng.uniform(0.2, 2, n)
    low = close - rng.uniform(0.2, 2, n)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.integers(100, 1000, n)}, index=idx)


def test_uptrend_is_long():
    a = analyze(make(0.8), "GC=F", "15m")
    assert a.score > 0.15 and "лонг" in a.label.lower()
    assert a.trade["stop"] < a.price < a.trade["target1"]


def test_downtrend_is_short():
    a = analyze(make(-0.8), "GC=F", "15m")
    assert a.score < -0.15 and "шорт" in a.label.lower()
    assert a.trade["stop"] > a.price > a.trade["target1"]


def test_levels_bracket_price():
    a = analyze(make(0.0, seed=3), "SI=F", "15m")
    assert all(x > a.price for x in a.levels["resistance"])
    assert all(x < a.price for x in a.levels["support"])
    assert a.summary


def test_multi_timeframe():
    r = {"5m": analyze(make(0.8)), "1h": analyze(make(-0.8, seed=1))}
    s, _ = multi_timeframe(r)
    assert -1 <= s <= 1
