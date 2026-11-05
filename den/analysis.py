"""Сигналы индикаторов, консенсус лонг/шорт, уровни и текстовый разбор."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class Vote:
    name: str
    score: float  # от -1 (сильный шорт) до +1 (сильный лонг)
    weight: float
    reason: str


@dataclass
class Analysis:
    symbol: str
    interval: str
    price: float
    change_pct: float
    votes: list[Vote]
    score: float
    long_pct: float
    label: str
    atr: float
    levels: dict[str, list[float]] = field(default_factory=dict)
    trade: dict[str, float] = field(default_factory=dict)
    summary: str = ""


def _clip(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def label_for(score: float) -> str:
    if score >= 0.5:
        return "Активный лонг"
    if score >= 0.15:
        return "Лонг"
    if score <= -0.5:
        return "Активный шорт"
    if score <= -0.15:
        return "Шорт"
    return "Нейтрально"


def collect_votes(df: pd.DataFrame) -> list[Vote]:
    """Каждый индикатор голосует за лонг или шорт на последней закрытой свече."""
    last, prev = df.iloc[-1], df.iloc[-2]
    c = last["close"]
    votes: list[Vote] = []

    # Тренд по скользящим средним
    s = 0.5 * np.sign(c - last["ema21"]) + 0.5 * np.sign(last["ema9"] - last["ema21"])
    votes.append(Vote("EMA 9/21", s, 1.5, f"цена {'выше' if c > last['ema21'] else 'ниже'} EMA21, "
                      f"EMA9 {'выше' if last['ema9'] > last['ema21'] else 'ниже'} EMA21"))
    if not np.isnan(last["ema200"]) and len(df) >= 200:
        s = np.sign(last["ema50"] - last["ema200"])
        votes.append(Vote("EMA 50/200", s, 1.0, "старший тренд " + ("восходящий" if s > 0 else "нисходящий")))

    # MACD: положение относительно сигнальной и ускорение гистограммы
    s = 0.6 * np.sign(last["hist"]) + 0.4 * np.sign(last["hist"] - prev["hist"])
    votes.append(Vote("MACD", s, 1.5, f"гистограмма {last['hist']:+.3f}, "
                      + ("растёт" if last["hist"] > prev["hist"] else "снижается")))

    # RSI: импульс в середине диапазона, разворот у краёв
    r = last["rsi"]
    if r >= 70:
        s, why = -0.6, f"RSI {r:.0f}: перекупленность"
    elif r <= 30:
        s, why = 0.6, f"RSI {r:.0f}: перепроданность"
    else:
        s, why = (r - 50) / 20, f"RSI {r:.0f}"
    votes.append(Vote("RSI 14", _clip(s), 1.0, why))

    # Стохастик
    k, d = last["stoch_k"], last["stoch_d"]
    if k >= 80 and k < d:
        s, why = -0.8, "разворот вниз из зоны перекупленности"
    elif k <= 20 and k > d:
        s, why = 0.8, "разворот вверх из зоны перепроданности"
    else:
        s, why = 0.5 * np.sign(k - d), f"%K {'выше' if k > d else 'ниже'} %D"
    votes.append(Vote("Stochastic", s, 0.8, f"{why} (%K {k:.0f})"))

    # ADX и направленные линии: сила и направление тренда
    strength = min(last["adx"] / 40, 1.0) if not np.isnan(last["adx"]) else 0.0
    s = np.sign(last["plus_di"] - last["minus_di"]) * strength
    votes.append(Vote("ADX/DMI", s, 1.2, f"ADX {last['adx']:.0f}, "
                      + ("+DI выше -DI" if last["plus_di"] > last["minus_di"] else "-DI выше +DI")))

    # Боллинджер: в боковике работает от границ, в тренде пробой — продолжение
    width = last["bb_up"] - last["bb_low"]
    pb = (c - last["bb_low"]) / width if width > 0 else 0.5
    if last["adx"] >= 25:
        s, why = _clip((pb - 0.5) * 2) * 0.7, f"тренд, цена в {pb:.0%} канала"
    else:
        s, why = _clip((0.5 - pb) * 2) * 0.7, f"боковик, цена в {pb:.0%} канала"
    votes.append(Vote("Bollinger", s, 0.8, why))

    # Объём: OBV относительно своей средней
    if df["volume"].tail(20).sum() > 0:
        s = 0.6 * np.sign(last["obv"] - last["obv_ema"])
        votes.append(Vote("OBV", s, 0.7, "объём подтверждает " + ("рост" if s > 0 else "снижение")))

    return [v for v in votes if not np.isnan(v.score)]


def _swing_levels(df: pd.DataFrame, window: int = 5, lookback: int = 300) -> list[float]:
    d = df.tail(lookback)
    hi, lo = d["high"], d["low"]
    span = 2 * window + 1
    highs = hi[hi == hi.rolling(span, center=True).max()]
    lows = lo[lo == lo.rolling(span, center=True).min()]
    return sorted(set(highs.round(4)).union(lows.round(4)))


def _cluster(levels: list[float], tol: float) -> list[float]:
    out: list[list[float]] = []
    for x in sorted(levels):
        if out and x - out[-1][-1] <= tol:
            out[-1].append(x)
        else:
            out.append([x])
    return [float(np.mean(g)) for g in out]


def levels(df: pd.DataFrame, atr_value: float) -> dict[str, list[float]]:
    """Ближайшие поддержки и сопротивления: свинги + классические пивоты прошлого дня."""
    price = float(df["close"].iloc[-1])
    pts = _swing_levels(df)
    daily = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    pivots = {}
    if len(daily) >= 2:
        h, l, c = daily.iloc[-2][["high", "low", "close"]]
        p = (h + l + c) / 3
        pivots = {"P": p, "R1": 2 * p - l, "S1": 2 * p - h, "R2": p + (h - l), "S2": p - (h - l)}
        pts += list(pivots.values())
    pts = _cluster(pts, tol=max(atr_value * 0.5, price * 1e-4))
    return {
        "resistance": [x for x in pts if x > price][:3],
        "support": [x for x in pts if x < price][::-1][:3],
        "pivots": [round(v, 4) for v in pivots.values()],
    }


def weighted_score(votes: list[Vote]) -> float:
    total_w = sum(v.weight for v in votes)
    return float(sum(v.score * v.weight for v in votes) / total_w) if total_w else 0.0


def analyze(df: pd.DataFrame, symbol: str = "", interval: str = "",
            extra_votes: list[Vote] | None = None) -> Analysis:
    """Консенсус по индикаторам; extra_votes добавляют внешние голоса, например деривативы Bybit."""
    if len(df) < 50:
        raise ValueError("Слишком мало свечей для анализа (нужно хотя бы 50)")
    df = ind.add_all(df)
    votes = collect_votes(df) + list(extra_votes or [])
    score = weighted_score(votes)
    last = df.iloc[-1]
    price = float(last["close"])
    a = float(last["atr"])
    lv = levels(df, a)
    direction = 1 if score >= 0.15 else -1 if score <= -0.15 else 0
    trade = {}
    if direction:
        trade = {
            "entry": price,
            "stop": price - direction * 1.5 * a,
            "target1": price + direction * 1.5 * a,
            "target2": price + direction * 3.0 * a,
        }
    res = Analysis(
        symbol=symbol, interval=interval, price=price,
        change_pct=float(df["close"].pct_change().iloc[-1] * 100),
        votes=votes, score=float(score), long_pct=float((score + 1) / 2 * 100),
        label=label_for(score), atr=a, levels=lv, trade=trade,
    )
    res.summary = summarize(res, df)
    return res


def summarize(a: Analysis, df: pd.DataFrame) -> str:
    bulls = [v for v in a.votes if v.score > 0.1]
    bears = [v for v in a.votes if v.score < -0.1]
    vol_now = df["atr"].iloc[-1] / df["close"].iloc[-1] * 100
    vol_avg = (df["atr"] / df["close"]).tail(100).mean() * 100
    lines = [
        f"**{a.label}** на таймфрейме {a.interval}: за лонг {a.long_pct:.0f}%, за шорт {100 - a.long_pct:.0f}%.",
        f"За рост: {', '.join(v.name for v in bulls) or 'нет'}. За снижение: {', '.join(v.name for v in bears) or 'нет'}.",
        f"Волатильность (ATR) {vol_now:.2f}% от цены, "
        + ("выше" if vol_now > vol_avg * 1.2 else "ниже" if vol_now < vol_avg * 0.8 else "около")
        + f" обычной ({vol_avg:.2f}%).",
    ]
    if a.levels.get("resistance"):
        lines.append(f"Ближайшее сопротивление {a.levels['resistance'][0]:.2f}.")
    if a.levels.get("support"):
        lines.append(f"Ближайшая поддержка {a.levels['support'][0]:.2f}.")
    if a.trade:
        side = "лонг" if a.trade["target1"] > a.trade["entry"] else "шорт"
        lines.append(
            f"Ориентир для {side}а по ATR: стоп {a.trade['stop']:.2f}, "
            f"цели {a.trade['target1']:.2f} и {a.trade['target2']:.2f}."
        )
    else:
        lines.append("Сигналы противоречат друг другу, явного перевеса нет: лучше дождаться пробоя уровня.")
    return " ".join(lines)


def multi_timeframe(results: dict[str, Analysis]) -> tuple[float, str]:
    """Общий консенсус по нескольким таймфреймам; младшие весят больше для краткосрока."""
    weights = {"5m": 1.0, "15m": 1.5, "1h": 1.5, "1d": 1.0}
    w = sum(weights.get(k, 1.0) for k in results)
    score = sum(r.score * weights.get(k, 1.0) for k, r in results.items()) / w if w else 0.0
    return score, label_for(score)


if __name__ == "__main__":
    import sys

    from .data import TIMEFRAMES, load_ohlcv

    sym = sys.argv[1] if len(sys.argv) > 1 else "GC=F"
    out = {}
    for tf in TIMEFRAMES:
        out[tf] = analyze(load_ohlcv(sym, tf), sym, tf)
        print(f"{tf:>4}: {out[tf].label:<14} лонг {out[tf].long_pct:5.1f}%")
    s, lbl = multi_timeframe(out)
    print(f"Итог: {lbl} ({(s + 1) / 2 * 100:.0f}% за лонг)\n")
    print(out["15m"].summary)
