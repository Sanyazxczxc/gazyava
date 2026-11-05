"""Загрузка рыночных данных из открытых источников (Yahoo Finance через yfinance)."""
from __future__ import annotations

import pandas as pd

ASSETS = {
    "Золото (фьючерс GC=F)": "GC=F",
    "Серебро (фьючерс SI=F)": "SI=F",
    "Apple (AAPL)": "AAPL",
}

# Таймфрейм -> максимальная глубина истории, которую отдаёт Yahoo для этого интервала.
TIMEFRAMES = {
    "5m": "5d",
    "15m": "30d",
    "1h": "180d",
    "1d": "2y",
}


def load_ohlcv(symbol: str, interval: str = "15m", period: str | None = None) -> pd.DataFrame:
    """Вернуть свечи с колонками open, high, low, close, volume и индексом времени."""
    import os

    if os.environ.get("DEN_DEMO") == "1":
        return demo_ohlcv(symbol, interval)
    import yfinance as yf

    period = period or TIMEFRAMES.get(interval, "30d")
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"Нет данных для {symbol} ({interval})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    return df.dropna(subset=["close"])


def demo_ohlcv(symbol: str, interval: str = "15m", n: int = 600) -> pd.DataFrame:
    """Синтетические свечи для проверки дашборда без доступа к сети (DEN_DEMO=1)."""
    import numpy as np

    freq = {"5m": "5min", "15m": "15min", "1h": "1h", "1d": "1D"}.get(interval, "15min")
    base = {"GC=F": 2650.0, "SI=F": 31.0, "AAPL": 225.0}.get(symbol, 100.0)
    rng = np.random.default_rng(abs(hash((symbol, interval))) % 2**32)
    idx = pd.date_range(end=pd.Timestamp("2026-11-05 18:00"), periods=n, freq=freq)
    close = base * np.exp(np.cumsum(rng.normal(0.0002, 0.003, n)))
    spread = close * rng.uniform(0.0005, 0.003, n)
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]], "high": close + spread, "low": close - spread,
                         "close": close, "volume": rng.integers(100, 5000, n)}, index=idx)
