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
    family: str = ""   # семейство индикаторов; пусто — по имени из FAMILIES
    mult: float = 1.0  # множитель режима рынка, который применил apply_regime


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
    regime: "Regime | None" = None


# Индикаторы, которые смотрят на одно и то же, объединены в семейства (MACD тоже строится
# на разнице скользящих средних, поэтому он в трендовом семействе). Сначала голоса
# усредняются внутри семейства, потом семейства между собой: пять трендовых индикаторов
# больше не перекрикивают один осциллятор только потому, что их больше.
FAMILIES = {
    "EMA 9/21": "trend", "EMA 50/200": "trend", "ADX/DMI": "trend", "MACD": "trend",
    "RSI 14": "oscillators", "Stochastic": "oscillators", "Bollinger": "oscillators",
    "OBV": "volume",
    "Funding": "derivatives", "Open interest": "derivatives", "Long/Short": "derivatives",
    "Дивергенция RSI": "structure", "Сжатие Боллинджера": "structure",
}
FAMILY_WEIGHT = {"trend": 1.5, "oscillators": 1.0, "volume": 0.6,
                 "derivatives": 1.0, "structure": 0.7, "cross": 0.6, "other": 0.8}
FAMILY_WORD = {"trend": "тренд и импульс", "oscillators": "осцилляторы", "volume": "объём",
               "derivatives": "деривативы", "structure": "разворот/пробой", "cross": "связанный актив",
               "other": "прочее"}
# Базовые веса голосов внутри семейства.
VOTE_WEIGHT = {"EMA 9/21": 1.5, "EMA 50/200": 1.0, "ADX/DMI": 1.2, "MACD": 1.5, "RSI 14": 1.0,
               "Stochastic": 0.8, "Bollinger": 0.8, "OBV": 0.7, "Дивергенция RSI": 1.0,
               "Сжатие Боллинджера": 1.0}

# Режим рынка: в тренде доверяем трендовым индикаторам, в боковике осцилляторам.
TREND_VOTES = ("EMA 9/21", "EMA 50/200", "MACD", "ADX/DMI")
RANGE_VOTES = ("RSI 14", "Stochastic", "Bollinger")
TREND_FAMILIES = ("trend",)
RANGE_FAMILIES = ("oscillators",)
REGIME_BOOST, REGIME_CUT = 1.5, 0.5
ADX_TREND, ADX_RANGE = 25.0, 20.0
MIN_RR = 1.5  # минимальное соотношение прибыль/риск до первой цели


def family_of(v: Vote) -> str:
    return v.family or FAMILIES.get(v.name, "other")


@dataclass
class Regime:
    kind: str          # "trend", "range" или "mixed"
    adx: float
    vol_ratio: float   # ATR в % от цены относительно среднего за 100 свечей
    label: str
    reason: str


def regime_frame(full: pd.DataFrame) -> pd.DataFrame:
    """Режим рынка на каждой свече (то же правило, что market_regime, но для всей истории сразу)."""
    adx = full["adx"].fillna(0.0)
    atr_pct = full["atr"] / full["close"]
    avg = atr_pct.rolling(100, min_periods=1).mean()
    vol = (atr_pct / avg).where(avg > 0, 1.0).fillna(1.0)
    kind = np.select([adx >= ADX_TREND, adx <= ADX_RANGE, vol >= 1.3, vol <= 0.8],
                     ["trend", "range", "trend", "range"], "mixed")
    return pd.DataFrame({"kind": kind, "adx": adx, "vol": vol}, index=full.index)


def market_regime(df: pd.DataFrame) -> Regime:
    """Тренд или боковик по ADX; в промежуточной зоне решает волатильность (расширение = тренд)."""
    r = regime_frame(df).iloc[-1]
    kind, adx, vol = str(r["kind"]), float(r["adx"]), float(r["vol"])
    if adx >= ADX_TREND:
        why = f"ADX {adx:.0f} ≥ {ADX_TREND:.0f}"
    elif adx <= ADX_RANGE:
        why = f"ADX {adx:.0f} ≤ {ADX_RANGE:.0f}"
    elif kind == "trend":
        why = f"ADX {adx:.0f}, волатильность растёт ({vol:.1f}× от обычной)"
    elif kind == "range":
        why = f"ADX {adx:.0f}, волатильность сжата ({vol:.1f}× от обычной)"
    else:
        why = f"ADX {adx:.0f}, волатильность обычная"
    label = {"trend": "Тренд", "range": "Боковик", "mixed": "Переходный"}[kind]
    if vol >= 1.8:
        label += ", высокая волатильность"
    return Regime(kind, adx, vol, label, why)


def regime_mult(family: str, kind: str) -> float:
    if kind == "trend":
        return REGIME_BOOST if family in TREND_FAMILIES else REGIME_CUT if family in RANGE_FAMILIES else 1.0
    if kind == "range":
        return REGIME_BOOST if family in RANGE_FAMILIES else REGIME_CUT if family in TREND_FAMILIES else 1.0
    return 1.0


def apply_regime(votes: list[Vote], regime: Regime | None) -> list[Vote]:
    """Перевзвесить голоса под режим рынка. В переходном режиме веса не меняются."""
    if regime is None or regime.kind == "mixed":
        return votes
    out = []
    for v in votes:
        k = regime_mult(family_of(v), regime.kind)
        out.append(Vote(v.name, v.score, round(v.weight * k, 3), v.reason, v.family, k) if k != 1.0 else v)
    return out


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


DIV_RECENT, DIV_LOOKBACK = 5, 40
SQUEEZE_WINDOW, SQUEEZE_Q, SQUEEZE_RECENT = 120, 0.2, 10


def vote_scores(full: pd.DataFrame) -> pd.DataFrame:
    """Голос каждого индикатора на каждой свече (колонки — имена голосов, NaN — голоса нет).
    full: свечи с индикаторами из indicators.add_all. Одна и та же функция считает и живой
    анализ (последняя строка), и историю для бэктеста, поэтому они не расходятся."""
    c = full["close"]
    sg = np.sign
    out = pd.DataFrame(index=full.index)
    out["EMA 9/21"] = 0.5 * sg(c - full["ema21"]) + 0.5 * sg(full["ema9"] - full["ema21"])
    pos = np.arange(len(full))
    out["EMA 50/200"] = sg(full["ema50"] - full["ema200"]).where(pos >= 199)
    h = full["hist"]
    out["MACD"] = 0.6 * sg(h) + 0.4 * sg(h.diff())

    r = full["rsi"]
    out["RSI 14"] = np.where(r >= 70, -0.6, np.where(r <= 30, 0.6, ((r - 50) / 20).clip(-1, 1)))
    out.loc[r.isna(), "RSI 14"] = np.nan

    k, d = full["stoch_k"], full["stoch_d"]
    st = np.where((k >= 80) & (k < d), -0.8, np.where((k <= 20) & (k > d), 0.8, 0.5 * sg(k - d)))
    out["Stochastic"] = pd.Series(st, index=full.index).where(k.notna() & d.notna())

    adx = full["adx"]
    strength = (adx / 40).clip(upper=1.0).fillna(0.0)
    out["ADX/DMI"] = sg(full["plus_di"] - full["minus_di"]) * strength

    width = full["bb_up"] - full["bb_low"]
    pb = ((c - full["bb_low"]) / width).where(width > 0, 0.5)
    trend = (adx >= 25).to_numpy()
    out["Bollinger"] = np.where(trend, ((pb - 0.5) * 2).clip(-1, 1) * 0.7, ((0.5 - pb) * 2).clip(-1, 1) * 0.7)
    out.loc[full["bb_mid"].isna(), "Bollinger"] = np.nan

    has_vol = full["volume"].rolling(20, min_periods=1).sum() > 0
    out["OBV"] = (0.6 * sg(full["obv"] - full["obv_ema"])).where(has_vol)

    # Дивергенция RSI: новый минимум цены при более высоком минимуме RSI (бычья) и наоборот.
    lo, hi = full["low"], full["high"]
    win = DIV_LOOKBACK - DIV_RECENT
    lo_now, lo_prev = lo.rolling(DIV_RECENT).min(), lo.shift(DIV_RECENT).rolling(win).min()
    hi_now, hi_prev = hi.rolling(DIV_RECENT).max(), hi.shift(DIV_RECENT).rolling(win).max()
    r_lo_now, r_lo_prev = r.rolling(DIV_RECENT).min(), r.shift(DIV_RECENT).rolling(win).min()
    r_hi_now, r_hi_prev = r.rolling(DIV_RECENT).max(), r.shift(DIV_RECENT).rolling(win).max()
    bull = (lo_now < lo_prev) & (r_lo_now > r_lo_prev + 3) & (r_lo_prev < 35)
    bear = (hi_now > hi_prev) & (r_hi_now < r_hi_prev - 3) & (r_hi_prev > 65)
    # Голос есть только при дивергенции: «нет дивергенции» не размывает консенсус нулём.
    out["Дивергенция RSI"] = np.where(bull, 0.7, np.where(bear, -0.7, np.nan))

    # Сжатие Боллинджера: ширина канала в нижних 20% за 120 свечей, затем пробой границы.
    bw = width / full["bb_mid"]
    squeeze = bw <= bw.rolling(SQUEEZE_WINDOW, min_periods=SQUEEZE_WINDOW // 2).quantile(SQUEEZE_Q)
    recent = squeeze.astype(float).shift(1).rolling(SQUEEZE_RECENT, min_periods=1).max() > 0
    # Голос появляется только в момент пробоя после недавнего сжатия.
    brk = np.where(c > full["bb_up"], 0.8, np.where(c < full["bb_low"], -0.8, np.nan))
    out["Сжатие Боллинджера"] = pd.Series(brk, index=full.index).where(recent)
    return out


def score_frame(full: pd.DataFrame, use_regime: bool = True) -> pd.Series:
    """Балл консенсуса на каждой свече: среднее внутри семейств, затем взвешенное по семействам."""
    vs = vote_scores(full)
    kinds = regime_frame(full)["kind"].to_numpy() if use_regime else np.full(len(full), "mixed")
    num = np.zeros(len(full))
    den = np.zeros(len(full))
    for fam in dict.fromkeys(FAMILIES[c] for c in vs.columns):
        cols = [c for c in vs.columns if FAMILIES[c] == fam]
        x = vs[cols].to_numpy(dtype=float)
        w = np.array([VOTE_WEIGHT.get(c, 1.0) for c in cols])
        ok = ~np.isnan(x)
        wsum = (ok * w).sum(axis=1)
        fs = np.where(wsum > 0, np.nansum(x * w, axis=1) / np.where(wsum > 0, wsum, 1), 0.0)
        mult = np.array([regime_mult(fam, k) for k in ("trend", "range", "mixed")])
        fm = np.select([kinds == "trend", kinds == "range"], [mult[0], mult[1]], mult[2])
        fw = FAMILY_WEIGHT[fam] * fm * (wsum > 0)
        num += fs * fw
        den += fw
    return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), 0.0), index=full.index)


def collect_votes(df: pd.DataFrame) -> list[Vote]:
    """Каждый индикатор голосует за лонг или шорт на последней закрытой свече."""
    row = vote_scores(df).iloc[-1]
    last, prev = df.iloc[-1], df.iloc[-2]
    c = last["close"]
    why = {
        "EMA 9/21": f"цена {'выше' if c > last['ema21'] else 'ниже'} EMA21, "
                    f"EMA9 {'выше' if last['ema9'] > last['ema21'] else 'ниже'} EMA21",
        "EMA 50/200": "старший тренд " + ("восходящий" if last["ema50"] > last["ema200"] else "нисходящий"),
        "MACD": f"гистограмма {last['hist']:+.3f}, " + ("растёт" if last["hist"] > prev["hist"] else "снижается"),
        "ADX/DMI": f"ADX {last['adx']:.0f}, " + ("+DI выше -DI" if last["plus_di"] > last["minus_di"]
                                                 else "-DI выше +DI"),
    }
    r = last["rsi"]
    why["RSI 14"] = (f"RSI {r:.0f}: перекупленность" if r >= 70 else
                     f"RSI {r:.0f}: перепроданность" if r <= 30 else f"RSI {r:.0f}")
    k, d = last["stoch_k"], last["stoch_d"]
    why["Stochastic"] = ("разворот вниз из зоны перекупленности" if k >= 80 and k < d else
                         "разворот вверх из зоны перепроданности" if k <= 20 and k > d else
                         f"%K {'выше' if k > d else 'ниже'} %D") + f" (%K {k:.0f})"
    width = last["bb_up"] - last["bb_low"]
    pb = (c - last["bb_low"]) / width if width > 0 else 0.5
    why["Bollinger"] = f"{'тренд' if last['adx'] >= 25 else 'боковик'}, цена в {pb:.0%} канала"
    why["OBV"] = "объём подтверждает " + ("рост" if row.get("OBV", 0) > 0 else "снижение")
    dv = row.get("Дивергенция RSI", np.nan)
    why["Дивергенция RSI"] = ("бычья: цена обновила минимум, RSI нет" if dv > 0 else
                              "медвежья: цена обновила максимум, RSI нет" if dv < 0 else "дивергенции нет")
    sq = row.get("Сжатие Боллинджера", np.nan)
    why["Сжатие Боллинджера"] = "пробой " + ("вверх" if sq > 0 else "вниз") + " после сжатия канала"
    return [Vote(n, float(s), VOTE_WEIGHT.get(n, 1.0), why.get(n, ""))
            for n, s in row.items() if not np.isnan(s)]


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
    """Сначала среднее внутри семейства, затем среднее семейств с весами FAMILY_WEIGHT
    (с поправкой режима рынка). Голоса без известного семейства идут в «прочее»."""
    fams: dict[str, list[Vote]] = {}
    for v in votes:
        fams.setdefault(family_of(v), []).append(v)
    num = den = 0.0
    for fam, vs in fams.items():
        w = sum(v.weight for v in vs)
        if w <= 0:
            continue
        fw = FAMILY_WEIGHT.get(fam, FAMILY_WEIGHT["other"]) * float(np.mean([v.mult for v in vs]))
        num += fw * sum(v.score * v.weight for v in vs) / w
        den += fw
    return float(num / den) if den else 0.0


def family_scores(votes: list[Vote]) -> dict[str, float]:
    fams: dict[str, list[Vote]] = {}
    for v in votes:
        fams.setdefault(family_of(v), []).append(v)
    return {f: sum(v.score * v.weight for v in vs) / sum(v.weight for v in vs)
            for f, vs in fams.items() if sum(v.weight for v in vs) > 0}


def plan_trade(price: float, atr_value: float, lv: dict[str, list[float]], direction: int) -> dict[str, float]:
    """Стоп за ближайший уровень (если он в 0.75–3 ATR), цели у следующих уровней.
    rr — прибыль до первой цели к риску до стопа."""
    if not direction or atr_value <= 0:
        return {}
    behind = lv.get("support" if direction > 0 else "resistance", [])
    ahead = lv.get("resistance" if direction > 0 else "support", [])
    stop, basis = price - direction * 1.5 * atr_value, "ATR"
    for x in behind:
        dist = abs(price - x) + 0.25 * atr_value  # чуть за уровень, чтобы не выбило проколом
        if 0.75 * atr_value <= dist <= 3 * atr_value:
            stop, basis = price - direction * dist, "уровень"
            break
    risk = abs(price - stop)
    goals = [x for x in ahead if abs(x - price) >= 0.25 * risk]
    t1 = goals[0] if goals else price + direction * 2 * risk
    rest = [x for x in goals[1:] if abs(x - t1) >= 0.5 * risk]
    t2 = rest[0] if rest else t1 + direction * max(risk, abs(t1 - price))
    return {"entry": price, "stop": stop, "target1": t1, "target2": t2,
            "rr": abs(t1 - price) / risk if risk else 0.0, "stop_by_level": float(basis == "уровень")}


def analyze(df: pd.DataFrame, symbol: str = "", interval: str = "",
            extra_votes: list[Vote] | None = None, use_regime: bool = True) -> Analysis:
    """Консенсус по индикаторам; extra_votes добавляют внешние голоса, например деривативы Bybit.
    use_regime перевзвешивает индикаторы под режим рынка (тренд или боковик)."""
    if len(df) < 50:
        raise ValueError("Слишком мало свечей для анализа (нужно хотя бы 50)")
    df = ind.add_all(df)
    reg = market_regime(df)
    votes = apply_regime(collect_votes(df), reg if use_regime else None) + list(extra_votes or [])
    score = weighted_score(votes)
    last = df.iloc[-1]
    price = float(last["close"])
    a = float(last["atr"])
    lv = levels(df, a)
    direction = 1 if score >= 0.15 else -1 if score <= -0.15 else 0
    res = Analysis(
        symbol=symbol, interval=interval, price=price,
        change_pct=float(df["close"].pct_change().iloc[-1] * 100),
        votes=votes, score=float(score), long_pct=float((score + 1) / 2 * 100),
        label=label_for(score), atr=a, levels=lv, trade=plan_trade(price, a, lv, direction), regime=reg,
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
        (f"Режим рынка: {a.regime.label.lower()} ({a.regime.reason}), "
         + ("вес трендовых индикаторов повышен. " if a.regime.kind == "trend" else
            "вес осцилляторов повышен. " if a.regime.kind == "range" else "веса без изменений. ")
         if a.regime else "")
        + f"Волатильность (ATR) {vol_now:.2f}% от цены, "
        + ("выше" if vol_now > vol_avg * 1.2 else "ниже" if vol_now < vol_avg * 0.8 else "около")
        + f" обычной ({vol_avg:.2f}%).",
    ]
    if a.levels.get("resistance"):
        lines.append(f"Ближайшее сопротивление {a.levels['resistance'][0]:.2f}.")
    if a.levels.get("support"):
        lines.append(f"Ближайшая поддержка {a.levels['support'][0]:.2f}.")
    if a.trade:
        side = "лонг" if a.trade["target1"] > a.trade["entry"] else "шорт"
        basis = "за ближайшим уровнем" if a.trade.get("stop_by_level") else "по ATR"
        lines.append(
            f"Ориентир для {side}а: стоп {a.trade['stop']:.2f} ({basis}), "
            f"цели {a.trade['target1']:.2f} и {a.trade['target2']:.2f}, "
            f"прибыль/риск до первой цели {a.trade.get('rr', 0):.1f}."
        )
        if a.trade.get("rr", 0) < MIN_RR:
            lines.append(f"Соотношение хуже 1:{MIN_RR:g}: до цели слишком близко, вход не рекомендуется.")
    else:
        lines.append("Сигналы противоречат друг другу, явного перевеса нет: лучше дождаться пробоя уровня.")
    return " ".join(lines)


def multi_timeframe(results: dict[str, Analysis]) -> tuple[float, str]:
    """Общий консенсус по нескольким таймфреймам; младшие весят больше для краткосрока."""
    weights = {"5m": 1.0, "15m": 1.5, "1h": 1.5, "1d": 1.0, "1w": 0.5}
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
