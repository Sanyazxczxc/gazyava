import numpy as np

from den.analysis import (RANGE_VOTES, TREND_VOTES, Regime, Vote, analyze, apply_regime, market_regime)
from den import indicators as ind
from den.signals import evaluate, final_signal, mtf_confirmation, recent_hit_rate
from tests.test_analysis import make


def _results(trends):
    return {tf: analyze(make(t, seed=i), "BYBIT:GOLD", tf) for i, (tf, t) in enumerate(trends.items())}


def test_regime_trend_and_range():
    assert market_regime(ind.add_all(make(0.8))).kind == "trend"
    flat = make(0.0, seed=5)
    flat["close"] = 2000 + np.random.default_rng(1).normal(0, 1, len(flat))  # шум вокруг уровня
    flat["high"], flat["low"] = flat["close"] + 1, flat["close"] - 1
    assert market_regime(ind.add_all(flat)).kind == "range"


def test_apply_regime_reweights():
    votes = [Vote(n, 1.0, 1.0, "") for n in TREND_VOTES + RANGE_VOTES + ("OBV",)]
    t = {v.name: v.weight for v in apply_regime(votes, Regime("trend", 30, 1, "Тренд", ""))}
    r = {v.name: v.weight for v in apply_regime(votes, Regime("range", 15, 1, "Боковик", ""))}
    assert t["MACD"] > 1 > t["RSI 14"] and r["RSI 14"] > 1 > r["MACD"]
    assert t["OBV"] == r["OBV"] == 1.0
    assert apply_regime(votes, Regime("mixed", 22, 1, "", "")) == votes


def test_confirmation_statuses():
    assert mtf_confirmation(_results({"5m": .8, "15m": .8, "1h": .8, "1d": .8})).status == "confirmed"
    assert mtf_confirmation(_results({"5m": .8, "15m": .8, "1h": .8, "1d": -.8})).status == "against"
    assert mtf_confirmation(_results({"5m": .8, "15m": -.8, "1h": .8, "1d": .8})).status == "no_signal"
    assert mtf_confirmation(_results({"5m": -.8, "15m": -.8, "1h": -.8})).status == "partial"  # нет 1d


def test_strong_signal_only_when_confirmed():
    sig = evaluate(_results({"5m": .8, "15m": .8, "1h": .8, "1d": .8}), make(0.8, seed=1))
    assert sig.passed and sig.strong and sig.label == "Сильный лонг" and sig.confidence >= 70
    sig = evaluate(_results({"5m": .8, "15m": .8, "1h": .8, "1d": -.8}), make(0.8, seed=1))
    assert not sig.passed and sig.label == "Нет сигнала"


def test_threshold_and_hit_rate_block():
    res = _results({"5m": .8, "15m": .8, "1h": .8, "1d": .8})
    assert final_signal(res, (0.9, 50), threshold=70).passed
    assert not final_signal(res, (0.9, 50), threshold=101).passed
    low = final_signal(res, (0.45, 50), threshold=50, min_hit=0.55)
    assert not low.passed and any("ниже минимума" in r for r in low.reasons)
    # Мало истории: угадываемость не блокирует, но и не добавляет уверенности.
    assert final_signal(res, (0.2, 3), threshold=50).passed


def test_recent_hit_rate_counts_last_signals():
    df = make(0.8)
    import pandas as pd
    scores = pd.Series(0.5, index=df.index[100:500])
    hit, n = recent_hit_rate(df, scores, 1, horizon=8, last=40)
    assert n == 40 and hit > 0.6
