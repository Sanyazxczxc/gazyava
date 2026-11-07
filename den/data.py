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

import json
import os
from pathlib import Path

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

# Сколько свечей истории держать по каждому таймфрейму. Bybit отдаёт максимум 1000 за запрос,
# поэтому история подгружается страницами и хранится в локальном кэше (~/.den/cache), а при
# следующих запусках догружаются только новые свечи. 5m: около 70 дней, 15m: 83 дня, 1h: 250 дней.
HISTORY_BARS = {"5m": 20000, "15m": 8000, "1h": 6000, "1d": 1000}
BYBIT_PAGE = 1000
BAR_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "1d": 1440}

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


def _cache_dir() -> Path:
    return Path(os.environ.get("DEN_HOME") or Path.home() / ".den") / "cache"


def _fetch_bybit(category: str, sym: str, interval: str, end_ms: int | None = None,
                 limit: int = BYBIT_PAGE) -> pd.DataFrame:
    params = {"category": category, "symbol": sym, "interval": BYBIT_INTERVALS.get(interval, "15"), "limit": limit}
    if end_ms is not None:
        params["end"] = end_ms
    res = _bybit_get("kline", params)
    if not res.get("list"):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype=float)
    return parse_bybit(res, drop_last=False)


def _closed(df: pd.DataFrame, interval: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Оставить только закрытые свечи: время открытия + длина свечи не позже текущего момента."""
    now = now or pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(minutes=BAR_MINUTES.get(interval, 15)) <= now]


def load_bybit(asset: str, interval: str = "15m", bars: int | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Свечи Bybit глубиной до bars (по умолчанию HISTORY_BARS), с локальным кэшем."""
    category, sym = bybit_symbol(asset)
    bars = bars or HISTORY_BARS.get(interval, BYBIT_PAGE)
    step = pd.Timedelta(minutes=BAR_MINUTES.get(interval, 15))
    path = _cache_dir() / f"bybit_{category}_{sym}_{interval}.csv.gz"
    meta_path = path.with_suffix(".json")
    cached, meta = None, {}
    if use_cache and path.exists():
        try:
            cached = pd.read_csv(path, index_col=0, parse_dates=True).astype(float)
            cached.index = pd.to_datetime(cached.index, utc=True)
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except Exception:  # noqa: BLE001  повреждённый кэш просто перекачиваем
            cached, meta = None, {}

    # Новые свечи: страницами от текущего момента назад, пока не дойдём до кэша или до bars.
    parts, end, got = [], None, 0
    while got < bars:
        page = _fetch_bybit(category, sym, interval, end)
        if page.empty:
            break
        parts.append(page)
        got += len(page)
        if cached is not None and len(cached) and page.index[0] <= cached.index[-1]:
            break
        if len(page) < BYBIT_PAGE:
            meta["start_reached"] = True
            break
        end = int(page.index[0].timestamp() * 1000) - 1
    frames = ([cached] if cached is not None else []) + parts
    if not frames:
        raise ValueError(f"Bybit не вернул свечи для {sym}")
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Старая история: догружаем назад, если её меньше bars и начало торгов ещё не достигнуто.
    while len(df) < bars and not meta.get("start_reached"):
        page = _fetch_bybit(category, sym, interval, int(df.index[0].timestamp() * 1000) - 1)
        if len(page) < BYBIT_PAGE:
            meta["start_reached"] = True
        if page.empty:
            break
        df = pd.concat([page, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()

    df = _closed(df, interval).tail(bars)
    if use_cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path)
            meta_path.write_text(json.dumps(meta))
        except OSError:
            pass  # кэш необязателен
    df.attrs["contract"] = sym + (" (фьючерс)" if category == "linear" else " (спот-токен)")
    return df


def parse_bybit(result: dict, drop_last: bool = True) -> pd.DataFrame:
    rows = result.get("list", [])
    if not rows:
        raise ValueError(f"Bybit не вернул свечи для {result.get('symbol', '')}")
    df = pd.DataFrame([r[:6] for r in rows], columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("ts").astype("int64"), unit="ms", utc=True)
    df = df.astype(float).sort_index()
    # Последняя свеча ещё не закрыта; анализ ведём по закрытым.
    return df.iloc[:-1] if drop_last else df


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


def demo_ohlcv(symbol: str, interval: str = "15m", n: int | None = None) -> pd.DataFrame:
    """Синтетические свечи для проверки дашборда без доступа к сети (DEN_DEMO=1)."""
    import numpy as np

    n = n or {"5m": 6000, "15m": 3000, "1h": 1500, "1d": 600}.get(interval, 1000)
    freq = {"5m": "5min", "15m": "15min", "1h": "1h", "1d": "1D"}.get(interval, "15min")
    base = {"GC=F": 2650.0, "SI=F": 31.0, "AAPL": 225.0, "BYBIT:GOLD": 2650.0, "BYBIT:SILVER": 31.0, "BYBIT:AAPL": 225.0, "TD:XAU/USD": 2650.0, "TD:XAG/USD": 31.0, "TD:AAPL": 225.0}.get(symbol, 100.0)
    rng = np.random.default_rng(abs(hash((symbol, interval))) % 2**32)
    idx = pd.date_range(end=pd.Timestamp("2026-11-05 18:00"), periods=n, freq=freq)
    close = base * np.exp(np.cumsum(rng.normal(0.0002, 0.003, n)))
    spread = close * rng.uniform(0.0005, 0.003, n)
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]], "high": close + spread, "low": close - spread,
                         "close": close, "volume": rng.integers(100, 5000, n)}, index=idx)
