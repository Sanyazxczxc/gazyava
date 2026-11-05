"""Данные деривативов Bybit: ставка финансирования, открытый интерес и соотношение лонг/шорт.

Всё берётся из публичного API Bybit v5 без ключа и есть только у бессрочных фьючерсов (category=linear).
У спот-токенов (например AAPLXUSDT) этих данных нет, тогда голоса просто не добавляются.

Как это голосует в консенсусе:
- Funding: высокая положительная ставка значит, что лонгов слишком много и они платят шортам.
  Это перегрев толпы, голос против неё (контрарный). Отрицательная ставка, наоборот, за лонг.
- Открытый интерес вместе с ценой: OI растёт и цена растёт — в рынок заходят новые лонги (за рост);
  OI растёт, цена падает — заходят шорты (за снижение); OI падает — позиции закрывают, сигнал слабее.
- Соотношение лонг/шорт по аккаунтам: большинство розницы обычно ошибается на перегибах,
  поэтому сильный перекос — слабый контрарный голос.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analysis import Vote

# Таймфрейм дашборда -> период OI и long/short у Bybit.
PERIODS = {"5m": "5min", "15m": "15min", "1h": "1h", "1d": "1d"}
# Сколько последних периодов сравнивать для изменения OI и цены.
OI_LOOKBACK = {"5m": 12, "15m": 8, "1h": 6, "1d": 5}
NORMAL_FUNDING = 0.0001  # 0.01% за 8 часов: базовая ставка Bybit, «нейтральное» значение


@dataclass
class Derivatives:
    symbol: str
    interval: str
    funding: float | None = None          # текущая ставка за период финансирования (доля, 0.0001 = 0.01%)
    funding_avg: float | None = None      # средняя за последние выплаты
    oi: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))     # открытый интерес по времени
    buy_ratio: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # доля аккаунтов в лонге


def _get(path: str, params: dict) -> dict:
    from .data import _bybit_get

    return _bybit_get(path, params)


def parse_series(rows: list[dict], key: str) -> pd.Series:
    if not rows:
        return pd.Series(dtype=float)
    ts = pd.to_datetime([int(r["timestamp"]) for r in rows], unit="ms", utc=True)
    return pd.Series([float(r[key]) for r in rows], index=ts, dtype=float).sort_index()


def load_derivatives(asset: str, interval: str = "15m") -> Derivatives | None:
    """Загрузить funding, OI и long/short для GOLD/SILVER/AAPL. None, если торгуется только спот-токен."""
    if os.environ.get("DEN_DEMO") == "1":
        return demo_derivatives(asset, interval)
    from .data import bybit_symbol

    category, sym = bybit_symbol(asset)
    if category != "linear":
        return None
    period = PERIODS.get(interval, "15min")
    d = Derivatives(sym, interval)
    tick = _get("tickers", {"category": "linear", "symbol": sym}).get("list", [])
    if tick and tick[0].get("fundingRate") not in (None, ""):
        d.funding = float(tick[0]["fundingRate"])
    hist = _get("funding/history", {"category": "linear", "symbol": sym, "limit": 9}).get("list", [])
    if hist:
        d.funding_avg = float(np.mean([float(h["fundingRate"]) for h in hist]))
    d.oi = parse_series(_get("open-interest", {"category": "linear", "symbol": sym, "intervalTime": period,
                                               "limit": 50}).get("list", []), "openInterest")
    try:
        d.buy_ratio = parse_series(_get("account-ratio", {"category": "linear", "symbol": sym, "period": period,
                                                          "limit": 50}).get("list", []), "buyRatio")
    except ValueError:  # для части контрактов Bybit не считает соотношение
        pass
    return d


def demo_derivatives(asset: str, interval: str = "15m") -> Derivatives:
    """Синтетические данные для демо-режима (DEN_DEMO=1)."""
    rng = np.random.default_rng(abs(hash((asset, interval, "deriv"))) % 2**32)
    idx = pd.date_range(end=pd.Timestamp("2026-11-05 18:00", tz="UTC"), periods=50,
                        freq={"5m": "5min", "15m": "15min", "1h": "1h", "1d": "1D"}.get(interval, "15min"))
    oi = pd.Series(1e6 * np.exp(np.cumsum(rng.normal(0, 0.01, 50))), index=idx)
    br = pd.Series(np.clip(0.55 + np.cumsum(rng.normal(0, 0.01, 50)), 0.2, 0.8), index=idx)
    fr = float(rng.normal(0.0001, 0.0002))
    return Derivatives(f"{asset}USDT (демо)", interval, fr, fr, oi, br)


def _clip(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def _price_at(price: pd.Series, ts: pd.Timestamp, n: int) -> float:
    """Цена на момент ts; если время не сопоставить (разные часовые пояса), то n свечей назад."""
    if getattr(price.index, "tz", None) is not None:
        before = price[price.index <= ts]
        if len(before):
            return float(before.iloc[-1])
    return float(price.iloc[max(len(price) - 1 - n, 0)])


def derivative_votes(d: Derivatives | None, price: pd.Series) -> list[Vote]:
    """Голоса деривативов для консенсуса. price: закрытия свечей того же таймфрейма."""
    if d is None:
        return []
    votes: list[Vote] = []

    if d.funding is not None:
        fr = d.funding if d.funding_avg is None else 0.5 * d.funding + 0.5 * d.funding_avg
        # Отклонение от базовой 0.01%: +0.05% сверху уже сильный перегрев лонгов.
        s = _clip(-(fr - NORMAL_FUNDING) / 0.0005) * 0.8
        crowd = "лонги переплачивают" if fr > NORMAL_FUNDING * 1.5 else \
            "шорты платят лонгам" if fr < 0 else "в норме"
        votes.append(Vote("Funding", s, 0.8, f"ставка {d.funding * 100:+.4f}% за 8 ч, {crowd}"))

    oi = d.oi.dropna()
    n = OI_LOOKBACK.get(d.interval, 8)
    if len(oi) > n and len(price) > 1:
        oi_ch = oi.iloc[-1] / oi.iloc[-1 - n] - 1
        p0 = _price_at(price, oi.index[-1 - n], n)
        p_ch = float(price.iloc[-1]) / p0 - 1
        # Изменение OI меньше 0.5% считаем шумом.
        strength = min(abs(oi_ch) / 0.03, 1.0) if abs(oi_ch) >= 0.005 else 0.0
        if oi_ch > 0:
            s = np.sign(p_ch) * strength
            why = "новые лонги" if p_ch > 0 else "новые шорты"
        else:
            s = 0.4 * np.sign(p_ch) * strength
            why = "закрытие шортов" if p_ch > 0 else "закрытие лонгов"
        if strength == 0:
            why = "почти не меняется"
        votes.append(Vote("Open interest", float(s), 1.0,
                          f"OI {oi_ch * 100:+.1f}%, цена {p_ch * 100:+.2f}% за {n} периодов: {why}"))

    br = d.buy_ratio.dropna()
    if len(br):
        b = float(br.iloc[-1])
        s = _clip(-(b - 0.5) / 0.2) * 0.6 if abs(b - 0.5) >= 0.05 else 0.0
        votes.append(Vote("Long/Short", s, 0.6, f"в лонге {b:.0%} аккаунтов, в шорте {1 - b:.0%}"
                          + (": толпа перекошена, голос против неё" if s else "")))
    return votes
