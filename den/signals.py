"""Итоговый торговый сигнал: подтверждение старшими таймфреймами и порог уверенности.

Логика:
1. Направление задают младшие таймфреймы (5m и 15m): они должны смотреть в одну сторону.
2. Старшие (1h и 1d) его подтверждают. Сильный сигнал бывает только при подтверждении обоими.
3. Уверенность, %, складывается из силы консенсуса младших ТФ, подтверждения старшими и того,
   как такой же сигнал угадывал направление на недавней истории.
4. Если уверенность ниже порога (по умолчанию 70%) или недавняя угадываемость ниже минимума,
   показываем «нет сигнала».
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analysis import Analysis

LOWER_TFS = ("5m", "15m")
HIGHER_TFS = ("1h", "1d")
DEFAULT_THRESHOLD = 70.0   # %, порог уверенности
DEFAULT_MIN_HIT = 0.55     # минимальная доля угаданных направлений на недавней истории
MIN_SAMPLES = 10           # сколько прошлых сигналов нужно, чтобы доверять угадываемости
HIT_TF = "15m"             # на каком таймфрейме проверяем угадываемость
HIT_HORIZON_MIN = 240      # через сколько минут смотрим, угадал ли сигнал
RECENT_SIGNALS = 60        # сколько последних сигналов одной стороны учитываем

DIR_WORD = {1: "лонг", -1: "шорт", 0: "нейтрально"}


def direction(score: float, threshold: float = 0.15) -> int:
    return 1 if score >= threshold else -1 if score <= -threshold else 0


@dataclass
class Confirmation:
    direction: int            # куда смотрят младшие ТФ, 0 если они расходятся или нейтральны
    status: str               # confirmed, partial, against, none, no_signal
    label: str
    factor: float             # вклад в уверенность от 0 до 1
    dirs: dict[str, int] = field(default_factory=dict)

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"


def mtf_confirmation(results: dict[str, Analysis]) -> Confirmation:
    dirs = {tf: direction(r.score) for tf, r in results.items()}
    low = [dirs[tf] for tf in LOWER_TFS if tf in dirs]
    d = low[0] if low and len(set(low)) == 1 and low[0] != 0 and len(low) == len(LOWER_TFS) else 0
    if d == 0:
        return Confirmation(0, "no_signal", "младшие ТФ не дают общего сигнала", 0.0, dirs)
    high = {tf: dirs.get(tf) for tf in HIGHER_TFS}
    agree = [tf for tf, x in high.items() if x == d]
    against = [tf for tf, x in high.items() if x == -d]
    if len(agree) == len(HIGHER_TFS):
        return Confirmation(d, "confirmed", "подтверждён старшими ТФ", 1.0, dirs)
    if against:
        return Confirmation(d, "against", f"против старших ТФ ({', '.join(against)})", 0.0, dirs)
    if agree:
        return Confirmation(d, "partial", f"частично подтверждён ({', '.join(agree)})", 0.6, dirs)
    return Confirmation(d, "none", "старшие ТФ нейтральны", 0.35, dirs)


def recent_hit_rate(df: pd.DataFrame, scores: pd.Series, side: int, horizon: int,
                    last: int = RECENT_SIGNALS, threshold: float = 0.15) -> tuple[float, int]:
    """Доля последних `last` сигналов стороны side, после которых цена через horizon свечей ушла в их сторону."""
    close = df["close"].reindex(scores.index)
    fwd = df["close"].shift(-horizon).reindex(scores.index)
    ret = (fwd / close - 1).dropna()
    s = scores.reindex(ret.index)
    mask = s >= threshold if side > 0 else s <= -threshold
    r = (ret[mask] * side).tail(last)
    return (float((r > 0).mean()) if len(r) else float("nan")), int(len(r))


@dataclass
class Signal:
    direction: int
    passed: bool              # прошёл порог уверенности и угадываемости
    strong: bool              # подтверждён старшими ТФ
    confidence: float         # 0..100
    threshold: float
    hit_rate: float
    hit_n: int
    confirmation: Confirmation
    reasons: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if not self.passed:
            return "Нет сигнала"
        return ("Сильный " if self.strong else "") + DIR_WORD[self.direction]


def final_signal(results: dict[str, Analysis], hit: tuple[float, int] | None = None,
                 threshold: float = DEFAULT_THRESHOLD, min_hit: float = DEFAULT_MIN_HIT) -> Signal:
    conf = mtf_confirmation(results)
    d = conf.direction
    hit_rate, hit_n = hit if hit else (float("nan"), 0)
    reasons = []
    if d == 0:
        reasons.append("5m и 15m не смотрят в одну сторону: входа нет")
        return Signal(0, False, False, 0.0, threshold, hit_rate, hit_n, conf, reasons)
    low = [results[tf].score for tf in LOWER_TFS if tf in results]
    strength = min(abs(float(np.mean(low))) / 0.5, 1.0)
    known = hit_n >= MIN_SAMPLES and not np.isnan(hit_rate)
    hit_f = float(np.clip((hit_rate - 0.5) / 0.15, 0, 1)) if known else 0.5
    confidence = 100 * (0.4 * strength + 0.35 * conf.factor + 0.25 * hit_f)
    reasons.append(f"5m и 15m за {DIR_WORD[d]}, сила консенсуса {strength:.0%}")
    reasons.append(conf.label)
    if known:
        reasons.append(f"недавние сигналы {DIR_WORD[d]} на {HIT_TF} угадывали направление "
                       f"через 4 часа в {hit_rate:.0%} случаев из {hit_n}")
    else:
        reasons.append(f"мало недавних сигналов {DIR_WORD[d]} для проверки ({hit_n}), угадываемость не учтена")
    passed = confidence >= threshold
    if not passed:
        reasons.append(f"уверенность {confidence:.0f}% ниже порога {threshold:.0f}%")
    if known and hit_rate < min_hit:
        passed = False
        reasons.append(f"угадываемость {hit_rate:.0%} ниже минимума {min_hit:.0%}")
    return Signal(d, passed, conf.confirmed, float(confidence), threshold, hit_rate, hit_n, conf, reasons)


def evaluate(results: dict[str, Analysis], candles: pd.DataFrame | None, scores: pd.Series | None = None,
             threshold: float = DEFAULT_THRESHOLD, min_hit: float = DEFAULT_MIN_HIT,
             use_regime: bool = True) -> Signal:
    """Итоговый сигнал с угадываемостью, посчитанной по свечам HIT_TF (обычно 15m)."""
    from .forecast import horizon_bars, score_series

    hit = None
    d = mtf_confirmation(results).direction
    if d and candles is not None and len(candles) > 120:
        if scores is None:
            scores = score_series(candles, last_n=len(candles), use_regime=use_regime)
        hit = recent_hit_rate(candles, scores, d, horizon_bars(candles, HIT_HORIZON_MIN))
    return final_signal(results, hit, threshold, min_hit)
