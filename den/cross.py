"""Связка золото–серебро: подтверждение вторым металлом и соотношение их цен (gold/silver ratio).

Золото и серебро обычно двигаются в одну сторону. Если второй металл смотрит туда же, сигнал
надёжнее; если против — это предупреждение. Соотношение цен золото/серебро исторически
возвращается к среднему: когда оно сильно выше обычного (серебро дешёвое относительно золота),
это слабый голос за серебро и против золота, и наоборот.
Голоса входят в семейство «связанный актив» и в бэктест не попадают.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .analysis import Vote, score_frame

PARTNERS = {
    "BYBIT:GOLD": "BYBIT:SILVER", "BYBIT:SILVER": "BYBIT:GOLD",
    "TD:XAU/USD": "TD:XAG/USD", "TD:XAG/USD": "TD:XAU/USD",
    "GC=F": "SI=F", "SI=F": "GC=F",
}
RATIO_WINDOW = 200
RATIO_Z = 1.5


def partner(symbol: str) -> str | None:
    return PARTNERS.get(symbol)


def _is_gold(symbol: str) -> bool:
    s = symbol.upper()
    return any(k in s for k in ("GOLD", "XAU", "GC=F"))


def cross_votes(symbol: str, own: pd.DataFrame, other: pd.DataFrame | None, use_regime: bool = True) -> list[Vote]:
    """Голоса связанного металла для свечей own (тот же таймфрейм у other)."""
    if other is None or len(other) < 60 or partner(symbol) is None:
        return []
    gold = _is_gold(symbol)
    name = "серебро" if gold else "золото"
    votes = []
    ps = float(score_frame(ind.add_all(other), use_regime).iloc[-1])
    votes.append(Vote(f"Подтверждение: {name}", float(np.clip(ps, -1, 1)), 1.0,
                      f"консенсус по {'серебру' if gold else 'золоту'} {ps:+.2f}", family="cross"))
    g, s = (own["close"], other["close"]) if gold else (other["close"], own["close"])
    j = pd.concat([g.rename("g"), s.rename("s")], axis=1, join="inner").dropna().tail(RATIO_WINDOW)
    if len(j) >= RATIO_WINDOW // 2:
        r = j["g"] / j["s"]
        sd = float(r.std())
        z = float((r.iloc[-1] - r.mean()) / sd) if sd > 0 else 0.0
        if abs(z) >= RATIO_Z:
            # Высокое соотношение: серебро дёшево относительно золота -> за серебро, против золота.
            k = float(np.clip(z / 3, -1, 1))
            sc = -0.4 * k if gold else 0.6 * k
            votes.append(Vote("Gold/Silver ratio", sc, 0.6,
                              f"соотношение {r.iloc[-1]:.1f}, {'выше' if z > 0 else 'ниже'} обычного "
                              f"на {abs(z):.1f} σ: ждём возврата к среднему", family="cross"))
    return votes
