"""Загрузка рыночных данных из открытых источников.

Источники:
- Bybit (бессрочные фьючерсы, без ключа): контракт подбирается при запуске из списка кандидатов,
  потому что набор контрактов на металлы и акции у биржи меняется.
- Twelve Data: золото XAU/USD, серебро XAG/USD, AAPL. Нужен бесплатный ключ
  с https://twelvedata.com, его задают в переменной окружения DEN_TWELVEDATA_KEY.
- Yahoo Finance через yfinance (запасной): фьючерсы GC=F, SI=F и AAPL. Если Yahoo
  заблокирован, задайте прокси в DEN_PROXY, например http://127.0.0.1:8080 или socks5://127.0.0.1:1080.
"""
from __future__ import annotations

import os

import pandas as pd

ASSETS = {
    "Золото (фьючерс Bybit)": "BYBIT:GOLD",
    "Серебро (фьючерс Bybit)": "BYBIT:SILVER",
    "Apple (фьючерс Bybit)": "BYBIT:AAPL",
    "Золото (XAU/USD)": "TD:XAU/USD",
    "Серебро (XAG/USD)": "TD:XAG/USD",
    "Apple (AAPL)": "TD:AAPL",
    "Золото, фьючерс GC=F (Yahoo)": "GC=F",
    "Серебро, фьючерс SI=F (Yahoo)": "SI=F",
    "Apple (AAPL, Yahoo)": "AAPL",
}

# Таймфрейм -> максимальная глубина истории, которую отдаёт Yahoo для этого интервала.
TIMEFRAMES = {
    "5m": "5d",
    "15m": "30d",
    "1h": "180d",
    "1d": "2y",
}

BYBIT_URL = "https://api.bybit.com/v5/market"
BYBIT_INTERVALS = {"5m": "5", "15m": "15", "1h": "60", "1d": "D"}
# Кандидаты в порядке предпочтения: сначала фьючерсы (linear), потом токены на споте.
# Если ни один не торгуется, контракт ищется по ключевым словам в списке инструментов.
BYBIT_CANDIDATES = {
    "GOLD": ["XAUUSDT", "XAUTUSDT", "PAXGUSDT"],
    "SILVER": ["XAGUSDT", "XAGTUSDT"],
    "AAPL": ["AAPLUSDT", "AAPLXUSDT"],
}
BYBIT_KEYWORDS = {"GOLD": ["XAU", "PAXG"], "SILVER": ["XAG", "SILVER"], "AAPL": ["AAPL"]}

TD_INTERVALS = {"5m": "5min", "15m": "15min", "1h": "1h", "1d": "1day"}
TD_URL = "https://api.twelvedata.com/time_series"


def load_ohlcv(symbol: str, interval: str = "15m", period: str | None = None) -> pd.DataFrame:
    """Вернуть свечи с колонками open, high, low, close, volume и индексом времени."""
    if os.environ.get("DEN_DEMO") == "1":
        return demo_ohlcv(symbol, interval)
    if symbol.startswith("BYBIT:"):
        return load_bybit(symbol[6:], interval)
    if symbol.startswith("TD:"):
        return load_twelvedata(symbol[3:], interval)
    return load_yahoo(symbol, interval, period)


def _bybit_get(path: str, params: dict) -> dict:
    import requests

    proxies = {"https": os.environ["DEN_PROXY"]} if os.environ.get("DEN_PROXY") else None
    r = requests.get(f"{BYBIT_URL}/{path}", params=params, timeout=15, proxies=proxies)
    r.raise_for_status()
    js = r.json()
    if js.get("retCode") != 0:
        raise ValueError(f"Bybit: {js.get('retMsg')}")
    return js["result"]


_bybit_symbol_cache: dict[str, tuple[str, str]] = {}


def _bybit_instruments(category: str) -> list[dict]:
    out, cursor = [], ""
    while True:
        res = _bybit_get("instruments-info", {"category": category, "limit": 1000, "cursor": cursor})
        out += res.get("list", [])
        cursor = res.get("nextPageCursor") or ""
        if not cursor:
            return out


def bybit_symbol(asset: str) -> tuple[str, str]:
    """Найти торгующийся инструмент (категория, тикер) для GOLD, SILVER, AAPL или тикера как есть."""
    if asset in _bybit_symbol_cache:
        return _bybit_symbol_cache[asset]
    candidates = BYBIT_CANDIDATES.get(asset, [asset])
    keywords = BYBIT_KEYWORDS.get(asset, [asset])
    for category in ("linear", "spot"):
        trading = {i["symbol"] for i in _bybit_instruments(category)
                   if i.get("status") == "Trading" and i.get("symbol", "").endswith("USDT")}
        found = [c for c in candidates if c in trading]
        found += sorted(t for t in trading if any(k in t for k in keywords) and t not in found)
        if found:
            _bybit_symbol_cache[asset] = (category, found[0])
            return _bybit_symbol_cache[asset]
    raise ValueError(f"На Bybit не найден торгующийся инструмент для {asset}")


def load_bybit(asset: str, interval: str = "15m", limit: int = 1000) -> pd.DataFrame:
    category, sym = bybit_symbol(asset)
    res = _bybit_get("kline", {"category": category, "symbol": sym,
                               "interval": BYBIT_INTERVALS.get(interval, "15"), "limit": limit})
    df = parse_bybit(res)
    df.attrs["contract"] = sym + (" (фьючерс)" if category == "linear" else " (спот-токен)")
    return df


def parse_bybit(result: dict) -> pd.DataFrame:
    rows = result.get("list", [])
    if not rows:
        raise ValueError(f"Bybit не вернул свечи для {result.get('symbol', '')}")
    df = pd.DataFrame([r[:6] for r in rows], columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("ts").astype("int64"), unit="ms", utc=True)
    df = df.astype(float).sort_index()
    # Последняя свеча ещё не закрыта; анализ ведём по закрытым.
    return df.iloc[:-1]


def load_twelvedata(symbol: str, interval: str = "15m", outputsize: int = 1000) -> pd.DataFrame:
    import requests

    key = os.environ.get("DEN_TWELVEDATA_KEY")
    if not key:
        raise ValueError(
            "Не задан ключ Twelve Data. Получите бесплатный ключ на twelvedata.com "
            "и укажите его в переменной окружения DEN_TWELVEDATA_KEY."
        )
    r = requests.get(TD_URL, params={
        "symbol": symbol, "interval": TD_INTERVALS.get(interval, "15min"),
        "outputsize": outputsize, "apikey": key, "order": "asc",
    }, timeout=20)
    r.raise_for_status()
    return parse_twelvedata(r.json(), symbol)


def parse_twelvedata(js: dict, symbol: str = "") -> pd.DataFrame:
    if js.get("status") != "ok" or not js.get("values"):
        raise ValueError(f"Twelve Data ({symbol}): {js.get('message', 'нет данных')}")
    df = pd.DataFrame(js["values"])
    df.index = pd.to_datetime(df.pop("datetime"))
    if "volume" not in df:
        df["volume"] = 0  # у валютных пар и металлов объёма нет
    df = df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()
    return df.dropna(subset=["close"])


def load_yahoo(symbol: str, interval: str, period: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    proxy = os.environ.get("DEN_PROXY")
    if proxy:
        yf.set_config(proxy=proxy)
    period = period or TIMEFRAMES.get(interval, "30d")
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(
            f"Нет данных Yahoo для {symbol} ({interval}). Если Yahoo заблокирован, "
            "задайте прокси в DEN_PROXY или выберите актив Twelve Data."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    return df.dropna(subset=["close"])


def demo_ohlcv(symbol: str, interval: str = "15m", n: int = 600) -> pd.DataFrame:
    """Синтетические свечи для проверки дашборда без доступа к сети (DEN_DEMO=1)."""
    import numpy as np

    freq = {"5m": "5min", "15m": "15min", "1h": "1h", "1d": "1D"}.get(interval, "15min")
    base = {"GC=F": 2650.0, "SI=F": 31.0, "AAPL": 225.0, "BYBIT:GOLD": 2650.0, "BYBIT:SILVER": 31.0, "BYBIT:AAPL": 225.0, "TD:XAU/USD": 2650.0, "TD:XAG/USD": 31.0, "TD:AAPL": 225.0}.get(symbol, 100.0)
    rng = np.random.default_rng(abs(hash((symbol, interval))) % 2**32)
    idx = pd.date_range(end=pd.Timestamp("2026-11-05 18:00"), periods=n, freq=freq)
    close = base * np.exp(np.cumsum(rng.normal(0.0002, 0.003, n)))
    spread = close * rng.uniform(0.0005, 0.003, n)
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]], "high": close + spread, "low": close - spread,
                         "close": close, "volume": rng.integers(100, 5000, n)}, index=idx)
