"""Геополитический фон: индекс GPR (Caldara & Iacoviello) и разметка новостей о войнах, санкциях,
пошлинах и выборах.

Это не голос в консенсусе. Фон только предупреждает о повышенном риске и уменьшает рекомендуемый
размер позиции: при обычном фоне 100% от базового риска, при повышенном 50%, при высоком 25%.

Индекс GPR: https://www.matteoiacoviello.com/gpr.htm (дневной ряд, обновляется авторами примерно раз
в неделю). Файл кэшируется в ~/.den/gpr_daily.csv; другой адрес можно задать в DEN_GPR_URL,
локальный файл (xls, xlsx или csv) в DEN_GPR_FILE.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
GPR_MAX_AGE = 12 * 3600
GEO_QUERY = ("war OR invasion OR missile OR airstrike OR sanctions OR tariffs OR \"trade war\" "
             "OR election OR ceasefire when:2d")
ASSETS = ("GOLD", "SILVER", "AAPL")
CATEGORY_WORD = {"war": "война/конфликт", "sanctions": "санкции", "tariffs": "пошлины",
                 "elections": "выборы", "deescalation": "деэскалация", "other": "другое"}
LEVEL_LABEL = {0: "обычный фон", 1: "повышенный риск", 2: "высокий риск"}
SIZE_MULT = {0: 1.0, 1: 0.5, 2: 0.25}


def _home() -> Path:
    return Path(os.environ.get("DEN_HOME") or Path.home() / ".den")


def _proxies() -> dict | None:
    return {"https": os.environ["DEN_PROXY"]} if os.environ.get("DEN_PROXY") else None


# ---------- Индекс GPR ----------

def parse_gpr(raw: pd.DataFrame) -> pd.Series:
    """Дневной ряд GPR из таблицы авторов. Колонки ищем по имени: дата (date или DAY) и GPRD."""
    cols = {str(c).strip().lower(): c for c in raw.columns}
    val = cols.get("gprd") or next((c for k, c in cols.items() if k.startswith("gpr")), None)
    if val is None:
        raise ValueError("в файле GPR нет колонки GPRD")
    if "date" in cols:
        idx = pd.to_datetime(raw[cols["date"]], errors="coerce")
    elif "day" in cols:
        idx = pd.to_datetime(raw[cols["day"]].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    else:
        raise ValueError("в файле GPR нет колонки с датой")
    s = pd.Series(pd.to_numeric(raw[val], errors="coerce").to_numpy(), index=idx, name="GPR")
    s = s[s.index.notna()].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def _read_table(content: bytes, name: str) -> pd.DataFrame:
    if name.endswith(".csv") or content[:1] not in (b"\xd0", b"P"):
        return pd.read_csv(io.BytesIO(content))
    try:
        return pd.read_excel(io.BytesIO(content))  # .xls нужен xlrd, .xlsx openpyxl
    except ImportError as e:
        lib = "xlrd" if content[:1] == b"\xd0" else "openpyxl"
        raise RuntimeError(f"для чтения файла GPR не установлен пакет {lib}. Выполните в терминале PyCharm: "
                           f"python -m pip install {lib} (или pip install -r requirements.txt) "
                           "и перезапустите дашборд") from None


def load_gpr(force: bool = False) -> pd.Series:
    """Дневной GPR. Сначала локальный файл или свежий кэш, иначе скачиваем; при ошибке сети старый кэш."""
    local = os.environ.get("DEN_GPR_FILE")
    if local:
        p = Path(local)
        return parse_gpr(_read_table(p.read_bytes(), p.name.lower()))
    cache = _home() / "gpr_daily.csv"
    if not force and cache.exists() and time.time() - cache.stat().st_mtime < GPR_MAX_AGE:
        return pd.read_csv(cache, index_col=0, parse_dates=True)["GPR"]
    import requests

    url = os.environ.get("DEN_GPR_URL", GPR_URL)
    try:
        r = requests.get(url, timeout=30, proxies=_proxies(), headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        s = parse_gpr(_read_table(r.content, url.lower()))
    except Exception:
        if cache.exists():
            return pd.read_csv(cache, index_col=0, parse_dates=True)["GPR"]
        raise
    cache.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_csv(cache)
    return s


@dataclass
class GPRStatus:
    date: pd.Timestamp
    value: float          # последнее дневное значение
    ma7: float            # среднее за 7 дней
    median_1y: float
    percentile: float     # где ma7 среди значений ma7 за последний год, 0..1
    level: int
    reason: str


def gpr_status(s: pd.Series) -> GPRStatus | None:
    s = s.dropna()
    if len(s) < 60:
        return None
    ma7 = s.rolling(7, min_periods=3).mean()
    year = ma7[ma7.index >= ma7.index[-1] - pd.Timedelta(days=365)].dropna()
    cur = float(ma7.iloc[-1])
    pct = float((year < cur).mean()) if len(year) else 0.5
    med = float(s[s.index >= s.index[-1] - pd.Timedelta(days=365)].median())
    level = 2 if pct >= 0.97 else 1 if pct >= 0.85 else 0
    reason = (f"GPR за 7 дней {cur:.0f} ({cur / med:.1f}× от медианы года {med:.0f}), выше "
              f"{pct:.0%} дней за год; данные на {s.index[-1]:%d.%m.%Y}")
    return GPRStatus(s.index[-1], float(s.iloc[-1]), cur, med, pct, level, reason)


# ---------- Разметка новостей ----------

@dataclass
class GeoTag:
    title: str
    link: str
    source: str
    published: datetime
    category: str                 # war, sanctions, tariffs, elections, deescalation, other
    severity: int                 # 0..3
    impact: dict[str, float] = field(default_factory=dict)   # GOLD/SILVER/AAPL от -1 до 1


KEYWORDS = {
    "deescalation": ["ceasefire", "truce", "peace deal", "peace talks", "de-escalat", "tariff pause",
                     "lifts sanctions", "eases sanctions", "trade deal"],
    "war": ["war ", "invasion", "invade", "missile", "airstrike", "air strike", "troops", "military",
            "drone attack", "bombing", "shelling", "attack on", "conflict", "nuclear"],
    "sanctions": ["sanction", "embargo", "export ban", "export control", "blacklist", "asset freeze"],
    "tariffs": ["tariff", "trade war", "import dut", "levies", "retaliatory"],
    "elections": ["election", "ballot", "referendum", "runoff", "inauguration", "snap poll"],
}
ESCALATION = ("nuclear", "invasion", "declares war", "escalat", "mobiliz", "full-scale", "major attack")
# Типичное влияние категорий на активы: металлы как защитные, AAPL страдает от пошлин и конфликтов.
IMPACT = {
    "war": {"GOLD": 0.6, "SILVER": 0.3, "AAPL": -0.4},
    "sanctions": {"GOLD": 0.3, "SILVER": 0.2, "AAPL": -0.2},
    "tariffs": {"GOLD": 0.3, "SILVER": 0.0, "AAPL": -0.6},
    "elections": {"GOLD": 0.2, "SILVER": 0.1, "AAPL": -0.1},
    "deescalation": {"GOLD": -0.3, "SILVER": -0.1, "AAPL": 0.3},
}


def tag_headline(title: str) -> tuple[str, int, dict[str, float]]:
    t = f" {title.lower()} "
    for cat in ("deescalation", "war", "sanctions", "tariffs", "elections"):
        if any(k in t for k in KEYWORDS[cat]):
            sev = 1 if cat in ("deescalation", "elections") else 2 if cat == "war" else 1
            if cat == "war" and any(k in t for k in ESCALATION):
                sev = 3
            return cat, sev, dict(IMPACT[cat])
    return "other", 0, dict.fromkeys(ASSETS, 0.0)


def fetch_geo_news(limit: int = 40) -> list[GeoTag]:
    """Геополитические заголовки за 2 дня из Google News, размеченные словарём."""
    import requests

    from .news import RSS_URL, parse_rss

    r = requests.get(RSS_URL, params={"q": GEO_QUERY, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                     timeout=15, proxies=_proxies(), headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return tag_items(parse_rss(r.text, "GOLD")[:limit])


def tag_items(items: list) -> list[GeoTag]:
    out = []
    for n in items:
        cat, sev, imp = tag_headline(n.title)
        out.append(GeoTag(n.title, n.link, n.source, n.published, cat, sev, imp))
    return out


GEO_PROMPT = """Ты аналитик геополитических рисков для трейдера золотом, серебром и акциями Apple.
Для каждого заголовка определи:
- category: war (война, удары, военный конфликт), sanctions, tariffs (пошлины, торговые войны),
  elections, deescalation (перемирие, переговоры, отмена санкций или пошлин), other (не геополитика);
- severity: 0 — неважно, 1 — фон, 2 — заметное событие, 3 — крупная эскалация, способная двинуть рынки;
- влияние на цену в ближайшие сутки от -1 до 1 отдельно для gold, silver, aapl.
Ответь строго одним JSON-объектом без пояснений:
{"items": [{"n": номер, "category": "...", "severity": 0, "gold": 0, "silver": 0, "aapl": 0}, ...]}
ровно N элементов в том же порядке."""


_OBJ = re.compile(r"\{[^{}]*\}", re.S)


def _loads_lenient(chunk: str):
    """json.loads с мелким ремонтом: висячие запятые, одинарные кавычки, True/False/None."""
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([}\]])", r"\1", chunk)
        fixed = re.sub(r"'([^']*)'", r'"\1"', fixed)
        fixed = re.sub(r"\b(True|False|None)\b", lambda m: {"True": "true", "False": "false", "None": "null"}[m[1]], fixed)
        return json.loads(fixed)


def _row(row: dict) -> tuple[str, int, dict[str, float]]:
    def num(x, default=0.0) -> float:
        try:
            return float(x)
        except (TypeError, ValueError):
            return default
    cat = str(row.get("category", "other")).lower().strip()
    cat = cat if cat in CATEGORY_WORD else "other"
    sev = int(max(0, min(3, round(num(row.get("severity", 0))))))
    imp = {a: max(-1.0, min(1.0, num(row.get(a.lower(), 0)))) for a in ASSETS}
    return cat, sev, imp


def parse_ai_tags(text: str, n: int) -> dict[int, tuple[str, int, dict[str, float]]]:
    """Разметка по номерам заголовков (с 0). Бесплатные модели часто ломают JSON, поэтому при ошибке
    разбираем каждый объект {...} отдельно и берём те, что читаются; остальные останутся словарными."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    rows: list = []
    try:
        from .ai import _extract_json

        d = _extract_json(text)
        rows = d.get("items", []) if isinstance(d, dict) else d if isinstance(d, list) else []
    except (ValueError, json.JSONDecodeError):
        for chunk in _OBJ.findall(text):
            try:
                obj = _loads_lenient(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "category" in obj:
                rows.append(obj)
    out = {}
    for pos, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        try:
            i = int(row.get("n", pos + 1)) - 1
        except (TypeError, ValueError):
            i = pos
        if 0 <= i < n and i not in out:
            out[i] = _row(row)
    return out


AI_BATCH = 15


def tag_with_ai(tags: list[GeoTag], key: str | None = None, model: str | None = None,
                timeout: float = 90, batch: int = AI_BATCH) -> tuple[list[GeoTag], str]:
    """Разметка через модель пачками по batch заголовков. Что модель не разметила, остаётся словарным.
    Ошибка только если не удалось разобрать ни одной пачки."""
    from .ai import AIError, _chat

    if not tags:
        return [], model or ""
    out, used, parsed_any, last_err = list(tags), model or "", False, None
    for start in range(0, len(tags), batch):
        part = tags[start:start + batch]
        body = f"Заголовков: {len(part)}.\n" + "\n".join(f"{i + 1}. {t.title}" for i, t in enumerate(part))
        msgs = [{"role": "system", "content": GEO_PROMPT.replace("ровно N", f"ровно {len(part)}")},
                {"role": "user", "content": body}]
        got: dict = {}
        for _attempt in range(2):  # один повтор, если ответ совсем не разобрался
            text, used = _chat(msgs, key, model, timeout, max_tokens=2000, temperature=0.0)
            got = parse_ai_tags(text, len(part))
            if got:
                break
        if not got:
            last_err = "ответ не похож на JSON"
            continue
        parsed_any = True
        for i, (c, sv, im) in got.items():
            out[start + i] = replace(part[i], category=c, severity=sv, impact=im)
    if not parsed_any:
        raise AIError(f"Модель вернула разметку новостей в неожиданном виде: {last_err}")
    return out, used


def tag_with_ai_throttled(tags: list[GeoTag], store: dict, model: str | None = None,
                          min_interval: float = 1800, now: float | None = None) -> tuple[list[GeoTag], str]:
    """Не чаще раза в min_interval секунд; между запросами новые заголовки размечаются словарём."""
    now = time.time() if now is None else now
    slot = ("geo", model or "")
    entry = store.get(slot)
    if entry and now - entry[0] < min_interval:
        known = entry[1]
        return [replace(t, **known[t.title]) if t.title in known else t for t in tags], entry[2]
    out, used = tag_with_ai(tags, model=model)
    store[slot] = (now, {t.title: {"category": t.category, "severity": t.severity, "impact": t.impact}
                         for t in out}, used)
    return out, used


# ---------- Итоговый фон ----------

@dataclass
class RiskBackdrop:
    level: int
    asset: str
    gpr: GPRStatus | None
    tags: list[GeoTag]
    impact: float                 # средневзвешенное влияние событий на актив, только для информации
    reasons: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return LEVEL_LABEL[self.level]

    @property
    def size_mult(self) -> float:
        return SIZE_MULT[self.level]

    @property
    def elevated(self) -> bool:
        return self.level > 0


def relevant(tags: list[GeoTag], asset: str, hours: float = 24, now: datetime | None = None) -> list[GeoTag]:
    now = now or datetime.now(timezone.utc)
    return [t for t in tags if t.category != "other" and t.severity >= 1
            and abs(t.impact.get(asset, 0.0)) >= 0.2 and (now - t.published).total_seconds() <= hours * 3600]


def risk_backdrop(gpr: GPRStatus | None, tags: list[GeoTag], asset: str,
                  now: datetime | None = None) -> RiskBackdrop:
    rel = relevant(tags, asset, now=now)
    heavy = [t for t in rel if t.severity >= 2 and t.category != "deescalation"]
    major = [t for t in heavy if t.severity >= 3]
    news_level = 2 if len(major) >= 2 else 1 if major or len(heavy) >= 3 else 0
    gpr_level = gpr.level if gpr else 0
    level = max(news_level, gpr_level)
    if news_level >= 1 and gpr_level >= 1:
        level = 2
    reasons = []
    if gpr:
        reasons.append(gpr.reason)
    else:
        reasons.append("индекс GPR недоступен, фон оценён только по новостям")
    cats: dict[str, int] = {}
    for t in heavy:
        cats[t.category] = cats.get(t.category, 0) + 1
    if heavy:
        reasons.append(f"за сутки {len(heavy)} заметных событий, влияющих на актив: "
                       + ", ".join(f"{CATEGORY_WORD[c]} {k}" for c, k in cats.items())
                       + (f", из них крупных {len(major)}" if major else ""))
    else:
        reasons.append("за сутки заметных геополитических событий, влияющих на актив, нет")
    w = [t.severity for t in rel]
    impact = float(np.average([t.impact.get(asset, 0.0) for t in rel], weights=w)) if rel and sum(w) else 0.0
    return RiskBackdrop(level, asset, gpr, tags, impact, reasons)


def position_size(base_risk_pct: float, stop_dist_pct: float, mult: float = 1.0) -> tuple[float, float]:
    """(риск на сделку в % депозита, объём позиции в % депозита) при стопе stop_dist_pct от входа."""
    risk = base_risk_pct * mult
    return risk, (risk / stop_dist_pct * 100 if stop_dist_pct > 0 else 0.0)
