from datetime import datetime, timezone

import numpy as np
import pandas as pd

from den import ai, ailog


def _res(p, d):
    return ai.AIResult(d, "средняя", "", prob_up=p, model="gpt-6-astra", horizon_h=4)


class _Sig:
    def __init__(self, d, passed=True):
        self.direction, self.passed, self.confidence = d, passed, 66.0


def test_log_resolve_stats(tmp_path, monkeypatch):
    monkeypatch.setenv("DEN_HOME", str(tmp_path))
    idx = pd.date_range("2026-11-02", periods=200, freq="15min", tz="UTC")
    df = pd.DataFrame({"close": np.linspace(100, 120, 200)}, index=idx)  # цена только растёт
    t0 = datetime(2026, 11, 2, 1, tzinfo=timezone.utc)
    ailog.append(ailog.make_entry("GOLD", "BYBIT:X", 101.0, _res(70, "лонг"), _Sig(1), now=t0, prompt_version="v2"))
    ailog.append(ailog.make_entry("GOLD", "BYBIT:X", 101.0, _res(30, "шорт"), _Sig(0, False), now=t0,
                                  prompt_version="v2"))
    late = datetime(2026, 11, 3, 23, tzinfo=timezone.utc)  # горизонт ещё не прошёл
    ailog.append(ailog.make_entry("GOLD", "BYBIT:X", 101.0, _res(60, "лонг"), None, now=late))
    log = ailog.load()
    assert len(log) == 3
    assert ailog.resolve(log, df, "BYBIT:X", now=datetime(2026, 11, 4, tzinfo=timezone.utc)) == 2
    ailog.save_all(log)
    s = ailog.stats(ailog.load())
    assert s["n"] == 2 and s["pending"] == 1 and s["ai_hit"] == 0.5 and s["den_hit"] == 1.0 and s["den_n"] == 1
    assert abs(s["brier_ai"] - (0.09 + 0.49) / 2) < 1e-9 and s["brier_base"] == 0.0
    assert s["both_n"] == 1 and s["both_hit"] == 1.0
    rows = ailog.stats_table(ailog.load(), "prompt")
    assert {r["prompt"] for r in rows} == {"v2", "–"}
