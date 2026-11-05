"""Новости из Google News RSS и оценка их тона для золота, серебра и AAPL.

Тон считается по словарю финансовых фраз: это грубая, но прозрачная оценка без внешних моделей.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

RSS_URL = "https://news.google.com/rss/search"

QUERIES = {
    "GOLD": "gold price OR XAU OR bullion when:3d",
    "SILVER": "silver price OR XAG when:3d",
    "AAPL": "Apple AAPL stock when:3d",
}

# Общие фразы о движении цены.
COMMON = {
    "surge": 1, "surges": 1, "soar": 1, "soars": 1, "rally": 1, "rallies": 1, "jumps": 1, "climbs": 1,
    "gains": 0.7, "rises": 0.7, "rebounds": 0.7, "higher": 0.5, "record high": 1, "all-time high": 1,
    "bullish": 1, "upbeat": 0.5, "strong": 0.4, "outperform": 0.7,
    "plunge": -1, "plunges": -1, "tumbles": -1, "slump": -1, "slumps": -1, "crash": -1, "sinks": -1,
    "falls": -0.7, "drops": -0.7, "declines": -0.7, "slides": -0.7, "lower": -0.5, "losses": -0.7,
    "bearish": -1, "weak": -0.4, "selloff": -1, "sell-off": -1, "underperform": -0.7,
}

# Макрофакторы для металлов: что обычно толкает золото и серебро вверх или вниз.
METALS = {
    "rate cut": 1, "rate cuts": 1, "dovish": 1, "safe haven": 0.8, "safe-haven": 0.8,
    "geopolitical": 0.6, "tensions": 0.5, "war": 0.5, "inflation rises": 0.7, "weaker dollar": 0.8,
    "dollar weakens": 0.8, "dollar falls": 0.8, "central bank buying": 0.8, "recession fears": 0.6,
    "rate hike": -1, "rate hikes": -1, "hawkish": -1, "stronger dollar": -0.8, "dollar strengthens": -0.8,
    "dollar rises": -0.8, "yields rise": -0.7, "yields climb": -0.7, "profit-taking": -0.5, "risk-on": -0.4,
    "industrial demand": 0.5,
}

AAPL = {
    "beats": 1, "beat estimates": 1, "upgrade": 1, "upgrades": 1, "price target raised": 1,
    "record revenue": 1, "strong demand": 0.8, "buyback": 0.6, "iphone sales rise": 0.8,
    "misses": -1, "downgrade": -1, "downgrades": -1, "price target cut": -1, "lawsuit": -0.6,
    "antitrust": -0.7, "probe": -0.5, "tariff": -0.6, "tariffs": -0.6, "weak demand": -0.8,
    "iphone sales fall": -0.8, "ban": -0.6,
}

NEGATIONS = ("not ", "no ", "fails to ", "despite ")


@dataclass
class NewsItem:
    title: str
    link: str
    source: str
    published: datetime
    score: float


def asset_key(symbol: str) -> str | None:
    s = symbol.upper()
    if any(k in s for k in ("GOLD", "XAU", "GC=F", "PAXG")):
        return "GOLD"
    if any(k in s for k in ("SILVER", "XAG", "SI=F")):
        return "SILVER"
    if "AAPL" in s:
        return "AAPL"
    return None


def score_headline(text: str, asset: str) -> float:
    """Тон заголовка от -1 до +1."""
    t = " " + text.lower() + " "
    lex = {**COMMON, **(AAPL if asset == "AAPL" else METALS)}
    total, hits = 0.0, 0
    for phrase in sorted(lex, key=len, reverse=True):
        for m in re.finditer(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", t):
            before = t[max(0, m.start() - 12):m.start()]
            sign = -1 if any(n in before for n in NEGATIONS) else 1
            total += sign * lex[phrase]
            hits += 1
            t = t[:m.start()] + " " * len(phrase) + t[m.end():]  # чтобы короткие фразы не считались повторно
    return max(-1.0, min(1.0, total / max(hits, 1))) if hits else 0.0


def parse_rss(xml_text: str, asset: str) -> list[NewsItem]:
    root = ET.fromstring(xml_text)
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src_el = it.find("source")
        source = src_el.text.strip() if src_el is not None and src_el.text else ""
        if source and title.endswith(" - " + source):
            title = title[: -len(source) - 3]
        try:
            pub = parsedate_to_datetime(it.findtext("pubDate") or "")
        except (TypeError, ValueError):
            pub = datetime.now(timezone.utc)
        out.append(NewsItem(title, it.findtext("link") or "", source, pub, score_headline(title, asset)))
    return sorted(out, key=lambda n: n.published, reverse=True)


def fetch_news(symbol: str, limit: int = 40) -> list[NewsItem]:
    import os

    import requests

    asset = asset_key(symbol)
    if asset is None:
        return []
    proxies = {"https": os.environ["DEN_PROXY"]} if os.environ.get("DEN_PROXY") else None
    r = requests.get(RSS_URL, params={"q": QUERIES[asset], "hl": "en-US", "gl": "US", "ceid": "US:en"},
                     timeout=15, proxies=proxies, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return parse_rss(r.text, asset)[:limit]


def sentiment(items: list[NewsItem], half_life_h: float = 24.0, now: datetime | None = None) -> dict:
    """Средний тон с затуханием по времени: свежие новости весят больше."""
    now = now or datetime.now(timezone.utc)
    num = den = 0.0
    pos = neg = 0
    for n in items:
        age_h = max((now - n.published).total_seconds() / 3600, 0)
        w = math.exp(-math.log(2) * age_h / half_life_h)
        if n.score:
            num += n.score * w
            den += w
            pos += n.score > 0
            neg += n.score < 0
    return {"score": num / den if den else 0.0, "positive": pos, "negative": neg,
            "neutral": len(items) - pos - neg, "total": len(items)}
