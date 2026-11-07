"""Торговые сессии базового актива.

Токенизированные металлы и акции на Bybit торгуются круглосуточно, но настоящая ликвидность
есть, только когда открыт базовый рынок: у металлов это Лондон и Нью-Йорк, у Apple основная
сессия биржи США. Ночью и в выходные цена токена двигается на тонком рынке, и сигналы
индикаторов там чаще ложные, поэтому вне сессии итоговый сигнал не выдаётся.
Праздники не учитываются.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# актив -> (часовой пояс открытия, час, минута, часовой пояс закрытия, час, минута)
SESSIONS = {
    "GOLD": ("Europe/London", 8, 0, "America/New_York", 17, 0),
    "SILVER": ("Europe/London", 8, 0, "America/New_York", 17, 0),
    "AAPL": ("America/New_York", 9, 30, "America/New_York", 16, 0),
}
SESSION_WORD = {
    "GOLD": "Лондон 08:00 – Нью-Йорк 17:00, пн–пт",
    "SILVER": "Лондон 08:00 – Нью-Йорк 17:00, пн–пт",
    "AAPL": "биржа США 09:30–16:00 по Нью-Йорку, пн–пт",
}


def _utc(times) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(times)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def session_mask(times, asset: str | None) -> np.ndarray:
    """True для моментов внутри основной сессии актива. Для неизвестного актива всегда True."""
    idx = _utc(times)
    if asset not in SESSIONS:
        return np.ones(len(idx), dtype=bool)
    tz_o, h_o, m_o, tz_c, h_c, m_c = SESSIONS[asset]
    o, c = idx.tz_convert(tz_o), idx.tz_convert(tz_c)
    after_open = (o.hour * 60 + o.minute) >= h_o * 60 + m_o
    before_close = (c.hour * 60 + c.minute) < h_c * 60 + m_c
    weekday = (o.weekday < 5) & (c.weekday < 5)
    return np.asarray(after_open & before_close & weekday)


def in_session(ts, asset: str | None) -> bool:
    return bool(session_mask([pd.Timestamp(ts)], asset)[0])
