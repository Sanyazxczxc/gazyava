import numpy as np
import pandas as pd

from den.backtest import BTConfig, describe, simulate
from den.forecast import score_series
from tests.test_analysis import make


def _bars(closes, spread=0.5):
    c = np.asarray(closes, dtype=float)
    idx = pd.date_range("2026-11-01", periods=len(c), freq="15min")
    return pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c + spread, "low": c - spread, "close": c,
                         "volume": 100}, index=idx)


def test_trend_is_profitable_without_costs():
    df = make(1.0, n=500)
    s = score_series(df, last_n=300)
    r = simulate(df, s, BTConfig(fee=0, slippage=0))
    m = r.metrics
    assert m["trades"] > 0 and m["longs"] > 0 and m["total_return"] > 0 and m["profit_factor"] > 1
    assert m["max_drawdown"] <= 0
    assert len(r.equity) == len(s)


def test_costs_reduce_result():
    df = make(0.0, n=500, seed=2)
    s = score_series(df, last_n=300)
    free = simulate(df, s, BTConfig(fee=0, slippage=0)).metrics
    paid = simulate(df, s, BTConfig(fee=0.001, slippage=0)).metrics
    slipped = simulate(df, s, BTConfig(fee=0.001, slippage=0.001)).metrics
    assert free["trades"] == paid["trades"] > 0
    assert slipped["total_return"] < paid["total_return"] < free["total_return"]
    assert abs(paid["avg_trade"] - free["avg_trade"]) > 0.15  # около 0.2% комиссии за сделку


def test_entry_next_open_and_stop_first():
    # Сигнал лонг на свече 0, вход по открытию свечи 1 (=100). На свече 2 задеты и стоп, и цель: считаем стоп.
    df = _bars([100, 100, 100, 100])
    df["atr"] = 1.0
    df.loc[df.index[2], ["high", "low"]] = [110, 90]
    s = pd.Series([1.0, 0.0, 0.0, 0.0], index=df.index)
    r = simulate(df, s, BTConfig(fee=0, slippage=0, stop_atr=2, target_atr=4, exit_on_flip=False))
    t = r.trades.iloc[0]
    assert t["Сторона"] == "лонг" and t["Цена входа"] == 100 and t["Причина"] == "стоп"
    assert abs(t["Результат %"] - (-2.0)) < 1e-9


def test_short_target_and_fee_math():
    df = _bars([100, 100, 100, 95, 95])
    df["atr"] = 1.0
    s = pd.Series([-1.0, 0, 0, 0, 0], index=df.index)
    r = simulate(df, s, BTConfig(fee=0.001, slippage=0, stop_atr=2, target_atr=3, exit_on_flip=False))
    t = r.trades.iloc[0]
    assert t["Сторона"] == "шорт" and t["Причина"] == "цель" and t["Цена выхода"] == 97
    expected = (1 - 0.001) * (1 + 0.03) * (1 - 0.001) - 1
    assert abs(t["Результат %"] - expected * 100) < 1e-9


def test_no_signals():
    df = make(0.0, n=200)
    s = pd.Series(0.0, index=df.index[-100:])
    m = simulate(df, s).metrics
    assert m["trades"] == 0 and m["total_return"] == 0
    assert "Сделок не было" in describe(m)
