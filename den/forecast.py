"""Прогноз на ближайшее будущее: диапазон цены, проверка сигналов на истории и общий вердикт.

Это вероятностная оценка, а не предсказание: диапазоны считаются по недавней волатильности,
а надёжность сигнала по тому, как такие же сигналы срабатывали раньше на этом активе.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind
from .analysis import apply_regime, collect_votes, label_for, market_regime

HORIZONS = {"1 час": 60, "4 часа": 240, "1 день": 1440}


def bar_minutes(df: pd.DataFrame) -> float:
    d = pd.Series(df.index).diff().dropna().median()
    return max(d.total_seconds() / 60, 1.0) if pd.notna(d) else 15.0


def horizon_bars(df: pd.DataFrame, minutes: int) -> int:
    return max(int(round(minutes / bar_minutes(df))), 1)


def price_ranges(df: pd.DataFrame, lookback: int = 300) -> list[dict]:
    """Ожидаемый диапазон цены по волатильности логарифмических доходностей (68% и 95%)."""
    price = float(df["close"].iloc[-1])
    r = np.log(df["close"]).diff().dropna().tail(lookback)
    sigma, drift = float(r.std()), float(r.mean())
    out = []
    for name, minutes in HORIZONS.items():
        h = horizon_bars(df, minutes)
        if h > len(r):
            continue
        s, m = sigma * np.sqrt(h), drift * h
        out.append({
            "Горизонт": name,
            "68%: от": price * np.exp(m - s), "68%: до": price * np.exp(m + s),
            "95%: от": price * np.exp(m - 2 * s), "95%: до": price * np.exp(m + 2 * s),
            "Ход ±%": s * 100,
        })
    return out


def score_series(df: pd.DataFrame, last_n: int = 400, use_regime: bool = True) -> pd.Series:
    """Балл консенсуса на каждой из последних last_n свечей (как если бы анализ запускали тогда),
    с тем же перевзвешиванием под режим рынка, что и в живом анализе."""
    full = ind.add_all(df)
    start = max(len(full) - last_n, 60)
    vals = {}
    for i in range(start, len(full)):
        part = full.iloc[: i + 1]
        votes = collect_votes(part)
        if use_regime:
            votes = apply_regime(votes, market_regime(part))
        w = sum(v.weight for v in votes)
        vals[full.index[i]] = sum(v.score * v.weight for v in votes) / w if w else 0.0
    return pd.Series(vals, dtype=float)


def backtest(df: pd.DataFrame, scores: pd.Series, horizon: int, threshold: float = 0.15) -> dict:
    """Как часто сигнал лонг/шорт угадывал направление цены через horizon свечей."""
    close = df["close"].reindex(scores.index)
    fwd = df["close"].shift(-horizon).reindex(scores.index)
    ret = (fwd / close - 1).dropna()
    s = scores.reindex(ret.index)
    res = {}
    for side, mask, sign in (("long", s >= threshold, 1), ("short", s <= -threshold, -1)):
        r = ret[mask] * sign
        res[side] = {"n": int(len(r)), "hit": float((r > 0).mean()) if len(r) else float("nan"),
                     "avg": float(r.mean() * 100) if len(r) else float("nan")}
    res["base_up"] = float((ret > 0).mean()) if len(ret) else float("nan")
    return res


@dataclass
class Verdict:
    score: float
    label: str
    confidence: str
    reasons: list[str]


def verdict(tech_score: float, news_score: float | None, bt: dict | None) -> Verdict:
    """Объединить технику (по всем таймфреймам), тон новостей и историческую надёжность сигнала."""
    parts, weights, reasons = [tech_score], [0.6], []
    reasons.append(f"техника по всем таймфреймам: {label_for(tech_score).lower()} ({tech_score:+.2f})")
    if news_score is not None:
        parts.append(news_score)
        weights.append(0.25)
        tone = "позитивный" if news_score > 0.1 else "негативный" if news_score < -0.1 else "нейтральный"
        reasons.append(f"тон новостей {tone} ({news_score:+.2f})")
    side = "long" if tech_score >= 0.15 else "short" if tech_score <= -0.15 else None
    edge = 0.0
    if bt and side and bt[side]["n"] >= 10 and not np.isnan(bt[side]["hit"]):
        hit = bt[side]["hit"]
        edge = (hit - 0.5) * 2 * (1 if side == "long" else -1)
        parts.append(edge)
        weights.append(0.15)
        reasons.append(f"такой сигнал раньше угадывал направление в {hit:.0%} случаев из {bt[side]['n']}")
    score = float(np.dot(parts, weights) / sum(weights))
    agree = len({np.sign(p) for p in parts if abs(p) > 0.1})
    if abs(score) >= 0.35 and agree == 1:
        conf = "высокая"
    elif abs(score) >= 0.15 and agree <= 1:
        conf = "средняя"
    else:
        conf = "низкая"
    if agree > 1:
        reasons.append("факторы противоречат друг другу")
    if side and edge < -0.04:
        reasons.append("на истории этот сигнал работал хуже монетки: осторожно")
    return Verdict(score, label_for(score), conf, reasons)
