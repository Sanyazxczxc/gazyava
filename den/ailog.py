"""Журнал прогнозов ИИ и их проверка по факту.

Каждый ИИ-анализ записывается в ~/.den/ai_log.jsonl (или $DEN_HOME/ai_log.jsonl): время, цена,
горизонт, вероятность роста от ИИ, направление ИИ и сигнал DEN в тот же момент. Когда горизонт прошёл,
по свечам находится цена закрытия на этот момент, и запись получает исход. По исходам считаются
доля угаданных направлений и Brier score вероятности: так видно, помогает ли ИИ на самом деле,
и можно сравнивать версии промпта и модели.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

DIR_NUM = {"лонг": 1, "шорт": -1, "нейтрально": 0}


def log_path() -> Path:
    return Path(os.environ.get("DEN_HOME") or Path.home() / ".den") / "ai_log.jsonl"


def load(path: Path | None = None) -> list[dict]:
    path = path or log_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def save_all(entries: list[dict], path: Path | None = None) -> None:
    path = path or log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
    tmp.replace(path)


def make_entry(asset: str, symbol: str, price: float, res, signal=None, now: datetime | None = None,
               prompt_version: str = "") -> dict:
    now = now or datetime.now(timezone.utc)
    den_dir = int(signal.direction) if signal is not None and signal.passed else 0
    return {"ts": now.isoformat(timespec="seconds"), "asset": asset, "symbol": symbol, "price": float(price),
            "horizon_h": int(res.horizon_h), "prob_up": res.prob_up, "ai_dir": DIR_NUM.get(res.direction, 0),
            "reliable": bool(res.reliable), "samples": list(res.samples), "model": res.model,
            "prompt": prompt_version, "den_dir": den_dir,
            "den_conf": float(signal.confidence) if signal is not None else None,
            "exit_price": None, "up": None}


def append(entry: dict, path: Path | None = None) -> None:
    path = path or log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def resolve(entries: list[dict], candles: pd.DataFrame, symbol: str, now: datetime | None = None) -> int:
    """Проставить исход записям этого символа, у которых прошёл горизонт. Цена исхода: закрытие
    первой свечи, которая открылась не раньше конца горизонта. Возвращает число новых исходов."""
    if candles is None or not len(candles):
        return 0
    now = now or datetime.now(timezone.utc)
    idx = pd.DatetimeIndex(candles.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    close = candles["close"].to_numpy()
    n = 0
    for e in entries:
        if e.get("symbol") != symbol or e.get("up") is not None:
            continue
        end = datetime.fromisoformat(e["ts"]) + timedelta(hours=e.get("horizon_h", 4))
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        if end > now:
            continue
        pos = idx.searchsorted(pd.Timestamp(end))
        if pos >= len(idx) or idx[pos] - pd.Timestamp(end) > pd.Timedelta(hours=2):
            continue  # свечи на этот момент нет (пробел в истории)
        e["exit_price"] = float(close[pos])
        e["up"] = bool(e["exit_price"] > e["price"])
        n += 1
    return n


def _hit(rows: list[dict], key: str) -> tuple[float | None, int]:
    picked = [r for r in rows if r.get(key)]
    if not picked:
        return None, 0
    ok = sum((r[key] > 0) == r["up"] for r in picked)
    return ok / len(picked), len(picked)


def stats(entries: list[dict]) -> dict:
    """Сводка по записям с исходом: угадываемость ИИ, DEN и их совпадения, Brier score ИИ,
    Brier наивного прогноза «всегда базовая частота роста» для сравнения."""
    done = [e for e in entries if e.get("up") is not None]
    out = {"n": len(done), "pending": sum(1 for e in entries if e.get("up") is None)}
    if not done:
        return out
    base = sum(e["up"] for e in done) / len(done)
    out["base_up"] = base
    probs = [e for e in done if e.get("prob_up") is not None]
    if probs:
        out["brier_ai"] = sum((e["prob_up"] / 100 - e["up"]) ** 2 for e in probs) / len(probs)
        out["brier_base"] = sum((base - e["up"]) ** 2 for e in probs) / len(probs)
        out["brier_n"] = len(probs)
    out["ai_hit"], out["ai_n"] = _hit(done, "ai_dir")
    out["den_hit"], out["den_n"] = _hit(done, "den_dir")
    both = [dict(e, both=e["ai_dir"]) for e in done if e.get("ai_dir") and e["ai_dir"] == e.get("den_dir")]
    out["both_hit"], out["both_n"] = _hit(both, "both")
    return out


def stats_table(entries: list[dict], by: str = "prompt") -> list[dict]:
    """Таблица для дашборда: строка на каждое значение поля by (версия промпта, модель, актив)."""
    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(str(e.get(by) or "–"), []).append(e)

    def pct(x):
        return f"{x:.0%}" if x is not None else "–"

    rows = []
    for g, es in sorted(groups.items()):
        s = stats(es)
        rows.append({by: g, "с исходом": s["n"], "ждут": s["pending"],
                     "ИИ угадал": f"{pct(s.get('ai_hit'))} из {s.get('ai_n', 0)}",
                     "DEN угадал": f"{pct(s.get('den_hit'))} из {s.get('den_n', 0)}",
                     "ИИ и DEN вместе": f"{pct(s.get('both_hit'))} из {s.get('both_n', 0)}",
                     "Brier ИИ": round(s["brier_ai"], 3) if "brier_ai" in s else None,
                     "Brier базы": round(s["brier_base"], 3) if "brier_base" in s else None})
    return rows
