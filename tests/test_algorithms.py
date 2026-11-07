"""Проверки пакета улучшений: семейства, история сигнала без подглядывания, модель, фильтры."""
import numpy as np
import pandas as pd

from den import indicators as ind
from den.analysis import (Vote, analyze, apply_regime, collect_votes, market_regime, plan_trade, score_frame,
                          weighted_score)
from den.cross import cross_votes
from den.sessions import in_session
from den.signals import (barrier_outcomes, final_signal, fit_logit, signal_history)
from tests.test_analysis import make


def _frames(trend=0.3, n15=1500, seed=0):
    """Согласованные свечи всех ТФ из одной 5-минутной серии."""
    rng = np.random.default_rng(seed)
    n5 = n15 * 3
    idx = pd.date_range("2026-08-03", periods=n5, freq="5min", tz="UTC")
    close = 2000 + np.cumsum(trend / 3 + rng.normal(0, 0.6, n5))
    df5 = pd.DataFrame({"open": np.r_[close[0], close[:-1]], "close": close,
                        "high": close + rng.uniform(0.1, 1, n5), "low": close - rng.uniform(0.1, 1, n5),
                        "volume": rng.integers(100, 1000, n5)}, index=idx)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return {"5m": df5, "15m": df5.resample("15min").agg(agg).dropna(),
            "1h": df5.resample("1h").agg(agg).dropna(), "1d": df5.resample("1D").agg(agg).dropna(),
            "1w": df5.resample("W-MON", label="left", closed="left").agg(agg).dropna()}


def test_vectorized_score_matches_live_votes():
    full = ind.add_all(make(0.3, n=500, seed=4))
    sf = score_frame(full)
    for i in (80, 300, 499):
        part = full.iloc[: i + 1]
        live = weighted_score(apply_regime(collect_votes(part), market_regime(part)))
        assert abs(live - sf.iloc[i]) < 1e-9


def test_family_dedup_limits_correlated_votes():
    # Четыре одинаковых трендовых голоса против одного осциллятора: без семейств было бы 4:1.
    votes = [Vote(n, 1.0, 1.0, "") for n in ("EMA 9/21", "EMA 50/200", "MACD", "ADX/DMI")]
    votes.append(Vote("RSI 14", -1.0, 1.0, ""))
    s = weighted_score(votes)
    assert 0 < s < 0.3  # 1.5 против 1.0 по весам семейств


def test_barrier_outcomes():
    idx = pd.date_range("2026-11-02", periods=6, freq="15min")
    c = np.array([100, 100.5, 101, 102, 102, 102.0])
    df = pd.DataFrame({"open": c, "high": c + 0.2, "low": c - 0.2, "close": c, "volume": 1}, index=idx)
    df["atr"] = 1.0
    o = barrier_outcomes(df, horizon=3, k=1.5, fee=0)
    assert o["long"].iloc[0] == 1 and o["short"].iloc[0] == 0
    assert np.isnan(o["long"].iloc[-1])


def test_signal_history_has_no_lookahead():
    fr = _frames(0.3, n15=1200)
    h = signal_history(fr, asset=None)
    cut = fr["15m"].index[800]
    short = {tf: f[f.index < cut] for tf, f in fr.items()}
    h2 = signal_history(short, asset=None)
    a = h.frame.loc[h2.frame.index, ["s_5m", "s_15m", "s_1h", "s_1d", "s_1w", "dir"]]
    b = h2.frame[["s_5m", "s_15m", "s_1h", "s_1d", "s_1w", "dir"]]
    # Последние сутки: дневная свеча в укороченных данных не закрыта иначе, сравниваем до неё.
    keep = b.index < cut - pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(a[keep], b[keep])


def test_signal_history_trend_produces_long_signals_and_model():
    h = signal_history(_frames(1.0, n15=1500, seed=2), threshold=60)
    f = h.frame
    assert (f["side"] > 0).sum() > 20 and (f["side"] < 0).sum() < (f["side"] > 0).sum()
    assert h.model is not None and h.model.samples >= 200
    # Сигнал держится не меньше двух свечей направления: одиночные всплески не проходят.
    first = f.index[(f["side"] != 0) & (f["side"].shift(1) == 0)]
    assert all(f.loc[t, "persisted"] for t in first)


def test_fit_logit_recovers_direction():
    rng = np.random.default_rng(0)
    x = np.column_stack([np.ones(2000), rng.normal(size=2000)])
    y = (rng.random(2000) < 1 / (1 + np.exp(-(0.2 + 1.5 * x[:, 1])))).astype(float)
    w = fit_logit(x, y, l2=0.1)
    assert abs(w[1] - 1.5) < 0.3


def _res(t=0.8):
    return {tf: analyze(make(t, seed=i), "BYBIT:GOLD", tf) for i, tf in enumerate(("5m", "15m", "1h", "1d"))}


def test_filters_block_signal():
    res = _res()
    base = dict(hit=(0.9, 50), threshold=50, check_rr=False)
    assert final_signal(res, **base).passed
    assert not final_signal(res, **base, persisted=False).passed
    assert final_signal(res, **base, persisted=False, held=True).passed
    off = final_signal(res, **base, session_ok=False)
    assert not off.passed and any("сессия" in r for r in off.reasons)
    # Гистерезис: уже действующий сигнал держится при уверенности чуть ниже порога.
    p = 0.65
    assert not final_signal(res, hit=(0.9, 50), threshold=70, prob=p, check_rr=False).passed
    assert final_signal(res, hit=(0.9, 50), threshold=70, prob=p, held=True, check_rr=False).passed


def test_rr_filter():
    res = _res()
    res["15m"].trade = dict(res["15m"].trade, rr=0.8)
    sig = final_signal(res, (0.9, 50), threshold=50)
    assert not sig.passed and sig.rr == 0.8


def test_plan_trade_uses_levels():
    lv = {"support": [98.5, 95.0], "resistance": [104.0, 108.0]}
    t = plan_trade(100.0, 1.0, lv, 1)
    assert t["stop"] == 100 - 1.75 and t["stop_by_level"] == 1.0
    assert t["target1"] == 104.0 and t["target2"] == 108.0 and abs(t["rr"] - 4 / 1.75) < 1e-9
    t = plan_trade(100.0, 1.0, {"support": [], "resistance": []}, -1)
    assert t["stop"] == 101.5 and t["target1"] == 97.0 and t["rr"] == 2.0


def test_sessions():
    assert in_session(pd.Timestamp("2026-11-09 14:00", tz="UTC"), "GOLD")       # понедельник, Лондон+НЙ
    assert not in_session(pd.Timestamp("2026-11-09 23:00", tz="UTC"), "GOLD")
    assert not in_session(pd.Timestamp("2026-11-08 14:00", tz="UTC"), "GOLD")   # воскресенье
    assert not in_session(pd.Timestamp("2026-11-09 14:00", tz="UTC"), "AAPL")   # 9:00 в Нью-Йорке
    assert in_session(pd.Timestamp("2026-11-09 15:00", tz="UTC"), "AAPL")
    assert in_session(pd.Timestamp("2026-11-08 03:00", tz="UTC"), None)


def test_cross_votes():
    g, s = make(0.8), make(0.8, seed=3)
    s[["open", "high", "low", "close"]] /= 80
    v = cross_votes("BYBIT:GOLD", g, s)
    assert v and v[0].family == "cross" and v[0].score > 0
    assert cross_votes("BYBIT:AAPL", g, s) == []


def test_divergence_and_squeeze_votes_exist():
    full = ind.add_all(make(0.0, n=800, seed=7))
    from den.analysis import vote_scores

    vs = vote_scores(full)
    assert {"Дивергенция RSI", "Сжатие Боллинджера"} <= set(vs.columns)
    assert vs["Дивергенция RSI"].notna().any() and vs["Сжатие Боллинджера"].notna().any()
