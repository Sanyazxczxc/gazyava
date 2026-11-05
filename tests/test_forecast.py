from datetime import datetime, timedelta, timezone

from den.forecast import backtest, horizon_bars, price_ranges, score_series, verdict
from den.news import NewsItem, asset_key, parse_rss, score_headline, sentiment
from tests.test_analysis import make


def test_ranges_widen_with_horizon():
    rows = price_ranges(make(0.0, n=600))
    assert [r["Горизонт"] for r in rows] == ["1 час", "4 часа", "1 день"]
    assert rows[0]["95%: от"] < rows[0]["68%: от"] < rows[0]["68%: до"] < rows[0]["95%: до"]
    assert rows[0]["Ход ±%"] < rows[1]["Ход ±%"] < rows[2]["Ход ±%"]


def test_backtest_trend_signal_hits():
    df = make(1.0, n=500)
    s = score_series(df, last_n=200)
    bt = backtest(df, s, horizon_bars(df, 60))
    assert bt["long"]["n"] > 50 and bt["long"]["hit"] > 0.6


def test_verdict_agreement_and_conflict():
    bt = {"long": {"n": 50, "hit": 0.7, "avg": 0.1}, "short": {"n": 0, "hit": float("nan"), "avg": float("nan")}}
    v = verdict(0.6, 0.5, bt)
    assert "лонг" in v.label.lower() and v.confidence == "высокая"
    v = verdict(0.3, -0.8, None)
    assert v.confidence == "низкая"


def test_headline_tone():
    assert score_headline("Gold surges as Fed signals rate cuts", "GOLD") > 0.5
    assert score_headline("Gold slides as dollar strengthens on hawkish Fed", "GOLD") < -0.5
    assert score_headline("Apple downgraded? Analyst issues downgrade on weak demand", "AAPL") < 0
    assert score_headline("Silver price today", "SILVER") == 0
    assert asset_key("BYBIT:SILVER") == "SILVER" and asset_key("TD:XAU/USD") == "GOLD"


def test_parse_rss_and_sentiment():
    xml = """<rss><channel>
    <item><title>Gold hits record high - Reuters</title><link>http://a</link>
      <pubDate>Thu, 05 Nov 2026 10:00:00 GMT</pubDate><source url="x">Reuters</source></item>
    <item><title>Gold falls on profit-taking - Kitco</title><link>http://b</link>
      <pubDate>Mon, 02 Nov 2026 10:00:00 GMT</pubDate><source url="y">Kitco</source></item>
    </channel></rss>"""
    items = parse_rss(xml, "GOLD")
    assert items[0].title == "Gold hits record high" and items[0].source == "Reuters"
    s = sentiment(items, now=datetime(2026, 11, 5, 12, tzinfo=timezone.utc))
    assert s["score"] > 0 and s["positive"] == 1 and s["negative"] == 1
