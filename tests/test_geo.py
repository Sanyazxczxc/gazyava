from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from den import geo


def _gpr(spike: float = 1.0) -> pd.Series:
    d = pd.date_range("2025-01-01", "2026-11-01", freq="D")
    v = 100 + np.random.default_rng(0).gamma(2, 10, len(d))
    v[-7:] *= spike
    return pd.Series(v, index=d)


def test_parse_gpr_columns():
    d = pd.date_range("2026-01-01", periods=5, freq="D")
    raw = pd.DataFrame({"DAY": d.strftime("%Y%m%d").astype(int), "N10D": 1, "GPRD": [1, 2, 3, 4, 5],
                        "GPRD_MA7": 0, "event": ""})
    s = geo.parse_gpr(raw)
    assert list(s) == [1, 2, 3, 4, 5] and s.index[0] == pd.Timestamp("2026-01-01")
    raw2 = pd.DataFrame({"date": d, "GPRD": [5, 4, 3, 2, 1]})
    assert geo.parse_gpr(raw2).iloc[-1] == 1


def test_gpr_levels():
    assert geo.gpr_status(_gpr()).level == 0
    assert geo.gpr_status(_gpr(3.0)).level == 2


def test_keyword_tags():
    cat, sev, imp = geo.tag_headline("Russia launches missile attack, fears of escalation grow")
    assert cat == "war" and sev == 3 and imp["GOLD"] > 0 > imp["AAPL"]
    assert geo.tag_headline("US slaps new tariffs on Chinese electronics")[0] == "tariffs"
    assert geo.tag_headline("Ceasefire agreed after talks")[0] == "deescalation"
    assert geo.tag_headline("Apple unveils new iPhone")[0] == "other"


def _tag(cat, sev, hours=1, asset_imp=0.6):
    now = datetime.now(timezone.utc)
    return geo.GeoTag("t", "", "", now - timedelta(hours=hours), cat, sev,
                      {"GOLD": asset_imp, "SILVER": 0.3, "AAPL": -asset_imp})


def test_backdrop_levels_and_size():
    calm = geo.risk_backdrop(geo.gpr_status(_gpr()), [], "GOLD")
    assert calm.level == 0 and calm.size_mult == 1.0
    news = geo.risk_backdrop(None, [_tag("war", 3)], "GOLD")
    assert news.level == 1 and news.size_mult == 0.5 and news.elevated
    both = geo.risk_backdrop(geo.gpr_status(_gpr(1.6)), [_tag("war", 3)], "GOLD")
    assert both.level == 2 and both.size_mult == 0.25
    old = geo.risk_backdrop(None, [_tag("war", 3, hours=40)], "GOLD")
    assert old.level == 0  # событие старше суток не учитывается
    assert geo.position_size(1.0, 0.5, 0.5) == (0.5, 100.0)


def test_ai_tags_parsing_and_throttle(monkeypatch):
    text = ('```json\n{"items": [{"n": 1, "category": "war", "severity": 3, "gold": 0.7, "silver": 0.4, '
            '"aapl": -0.5}, {"n": 2, "category": "nonsense", "severity": 9, "gold": 5}]}\n```')
    rows = geo.parse_ai_tags(text, 2)
    assert rows[0] == ("war", 3, {"GOLD": 0.7, "SILVER": 0.4, "AAPL": -0.5})
    assert rows[1][0] == "other" and rows[1][1] == 3 and rows[1][2]["GOLD"] == 1.0

    calls = []
    import den.ai as ai

    def fake_chat(*a, **k):
        calls.append(1)
        return text, "m"
    monkeypatch.setattr(ai, "_chat", fake_chat)
    tags = [_tag("other", 0), _tag("other", 0)]
    tags[1].title = "u"
    store = {}
    out, _ = geo.tag_with_ai_throttled(tags, store, now=0)
    assert out[0].category == "war"
    geo.tag_with_ai_throttled(tags, store, now=100)
    assert len(calls) == 1
