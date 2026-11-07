"""Итоговый торговый сигнал: подтверждение старшими таймфреймами, калиброванная уверенность и фильтры.

Логика:
1. Направление задают младшие таймфреймы (5m и 15m): они должны смотреть в одну сторону.
2. Старшие (1h и 1d) его подтверждают. Сильный сигнал бывает только при подтверждении обоими.
3. Уверенность, %, это вероятность того, что цена пройдёт 1.5 ATR (15m) в сторону сигнала раньше,
   чем 1.5 ATR против него, в пределах 4 часов. Её даёт логистическая регрессия, обученная на
   истории этого актива только по прошлым данным (walk-forward). Пока истории мало, работает
   прежняя формула: 40% сила консенсуса младших ТФ, 35% подтверждение старшими, 25% угадываемость.
4. Угадываемость считается так же, по стопу и цели (что сработало раньше), с учётом комиссий.
5. Фильтры: уверенность не ниже порога; недавняя угадываемость не ниже минимума; сигнал держится
   две закрытые свечи 15m подряд; если сигнал уже был на прошлой свече, порог удержания на
   HYSTERESIS ниже (защита от мигания); основная сессия базового рынка открыта; прибыль/риск до
   первой цели не хуже MIN_RR.

signal_history() прогоняет ту же логику по всей истории без заглядывания в будущее: старшие ТФ
берутся только по уже закрытым свечам. По её результату строится бэктест итогового сигнала.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from .analysis import MIN_RR, Analysis, score_frame
from .sessions import session_mask

LOWER_TFS = ("5m", "15m")
HIGHER_TFS = ("1h", "1d")
ALL_TFS = LOWER_TFS + HIGHER_TFS
DEFAULT_THRESHOLD = 70.0   # %, порог уверенности
DEFAULT_MIN_HIT = 0.55     # минимальная доля удачных сигналов на недавней истории
MIN_SAMPLES = 10           # сколько прошлых сигналов нужно, чтобы доверять угадываемости
HIT_TF = "15m"             # базовый таймфрейм истории сигнала и проверки
HIT_HORIZON_MIN = 240      # сколько минут ждём, пока сработает стоп или цель
RECENT_SIGNALS = 60        # сколько последних сигналов одной стороны учитываем
BARRIER_ATR = 1.5          # стоп и цель для проверки, в ATR от цены сигнала
HYSTERESIS = 10.0          # на сколько процентных пунктов ниже порог удержания уже открытого сигнала
MODEL_MIN_SAMPLES = 200    # с какого числа размеченных сигналов доверяем модели
REFIT_EVERY = 96           # переобучать модель на истории раз в столько свечей 15m (сутки)
CHECK_FEE = 0.00075        # комиссия + проскальзывание за сторону при проверке «по стопу и цели»

DIR_WORD = {1: "лонг", -1: "шорт", 0: "нейтрально"}
TF_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "1d": 1440}
FEATURES = ("1", "5m", "15m", "1h", "1d", "подтверждение", "сила")


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


def _confirm(d: int, high: dict[str, int | None]) -> tuple[str, float, list[str], list[str]]:
    agree = [tf for tf, x in high.items() if x == d]
    against = [tf for tf, x in high.items() if x == -d]
    if len(agree) == len(HIGHER_TFS):
        return "confirmed", 1.0, agree, against
    if against:
        return "against", 0.0, agree, against
    if agree:
        return "partial", 0.6, agree, against
    return "none", 0.35, agree, against


def mtf_confirmation(results: dict[str, Analysis]) -> Confirmation:
    dirs = {tf: direction(r.score) for tf, r in results.items()}
    low = [dirs[tf] for tf in LOWER_TFS if tf in dirs]
    d = low[0] if low and len(set(low)) == 1 and low[0] != 0 and len(low) == len(LOWER_TFS) else 0
    if d == 0:
        return Confirmation(0, "no_signal", "младшие ТФ не дают общего сигнала", 0.0, dirs)
    status, factor, agree, against = _confirm(d, {tf: dirs.get(tf) for tf in HIGHER_TFS})
    label = {"confirmed": "подтверждён старшими ТФ",
             "against": f"против старших ТФ ({', '.join(against)})",
             "partial": f"частично подтверждён ({', '.join(agree)})",
             "none": "старшие ТФ нейтральны"}[status]
    return Confirmation(d, status, label, factor, dirs)


# ---------- Проверка по стопу и цели ----------

def barrier_outcomes(df: pd.DataFrame, horizon: int, k: float = BARRIER_ATR, fee: float = CHECK_FEE) -> pd.DataFrame:
    """Для каждой свечи и каждой стороны: 1, если от закрытия цена прошла k ATR в сторону сделки раньше,
    чем k ATR против (за horizon свечей); 0, если раньше сработал стоп. Если не сработало ни то ни другое,
    решает результат через horizon свечей после комиссий. Стоп и цель на одной свече считаются стопом.
    NaN — исход ещё неизвестен (последние horizon свечей)."""
    full = df if "atr" in df else ind.add_all(df)
    c, h, lo, a = (full[x].to_numpy(dtype=float) for x in ("close", "high", "low", "atr"))
    n = len(full)
    out = {}
    for side in (1, -1):
        res = np.full(n, np.nan)
        done = np.zeros(n, dtype=bool)
        stop, tgt = c - side * k * a, c + side * k * a
        for j in range(1, horizon + 1):
            valid = np.arange(n) + j < n
            hj = np.r_[h[j:], np.full(j, np.nan)]
            lj = np.r_[lo[j:], np.full(j, np.nan)]
            hit_stop = (lj <= stop) if side > 0 else (hj >= stop)
            hit_tgt = (hj >= tgt) if side > 0 else (lj <= tgt)
            m = ~done & valid
            loss, win = m & hit_stop, m & ~hit_stop & hit_tgt
            res[loss], res[win] = 0.0, 1.0
            done |= loss | win
        cf = np.r_[c[horizon:], np.full(horizon, np.nan)]
        ret = side * (cf / c - 1) - 2 * fee
        m = ~done & ~np.isnan(cf)
        res[m] = (ret[m] > 0).astype(float)
        out[side] = res
    return pd.DataFrame({"long": out[1], "short": out[-1]}, index=full.index)


def recent_hit_rate(df: pd.DataFrame, scores: pd.Series, side: int, horizon: int,
                    last: int = RECENT_SIGNALS, threshold: float = 0.15) -> tuple[float, int]:
    """Доля последних `last` сигналов стороны side, после которых раньше сработала цель, а не стоп."""
    o = barrier_outcomes(df, horizon)["long" if side > 0 else "short"].reindex(scores.index)
    s = scores.reindex(o.index)
    mask = (s >= threshold) if side > 0 else (s <= -threshold)
    r = o[mask].dropna().tail(last)
    return (float(r.mean()) if len(r) else float("nan")), int(len(r))


# ---------- Модель уверенности ----------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logit(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 30) -> np.ndarray:
    """Логистическая регрессия методом Ньютона с L2-регуляризацией (свободный член не штрафуется)."""
    w = np.zeros(x.shape[1])
    reg = np.full(x.shape[1], l2)
    reg[0] = 0.0
    for _ in range(iters):
        p = _sigmoid(x @ w)
        g = x.T @ (p - y) + reg * w
        hess = (x * (p * (1 - p))[:, None]).T @ x + np.diag(reg + 1e-9)
        step = np.linalg.solve(hess, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


@dataclass
class ConfidenceModel:
    weights: np.ndarray
    samples: int
    base_rate: float   # доля удачных сигналов в обучающей выборке

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(np.atleast_2d(x) @ self.weights)


def features(d, s5, s15, s1h, s1d, factor, strength) -> np.ndarray:
    """Признаки сигнала в его собственную сторону: d * балл, чтобы лонги и шорты учились вместе."""
    d = np.asarray(d, dtype=float)
    z = [np.nan_to_num(np.asarray(x, dtype=float)) for x in (s5, s15, s1h, s1d)]
    return np.column_stack([np.ones_like(d)] + [d * x for x in z]
                           + [np.asarray(factor, dtype=float), np.asarray(strength, dtype=float)])


def _train(x: np.ndarray, y: np.ndarray) -> ConfidenceModel | None:
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    if len(y) < MODEL_MIN_SAMPLES or y.min() == y.max():
        return None
    return ConfidenceModel(fit_logit(x, y), int(len(y)), float(y.mean()))


# ---------- История сигнала ----------

def _bar(frame: pd.DataFrame, tf: str) -> pd.Timedelta:
    return pd.Timedelta(minutes=TF_MINUTES.get(tf, 15))


def align_scores(frames: dict[str, pd.DataFrame], scores: dict[str, pd.Series], base_tf: str = HIT_TF) -> pd.DataFrame:
    """Баллы всех ТФ на момент закрытия каждой свечи base_tf. Берётся последняя свеча ТФ,
    которая к этому моменту уже закрылась, так что подглядывания в будущее нет."""
    base = frames[base_tf]
    t_close = base.index + _bar(base, base_tf)
    out = pd.DataFrame(index=base.index)
    for tf, s in scores.items():
        bar = _bar(frames[tf], tf)
        s_closed = pd.Series(s.to_numpy(), index=s.index + bar).sort_index()
        s_closed = s_closed[~s_closed.index.duplicated(keep="last")]
        tol = bar * 3 if tf != "1d" else pd.Timedelta(days=4)
        out[tf] = s_closed.reindex(t_close, method="ffill", tolerance=tol).to_numpy()
    return out


@dataclass
class History:
    frame: pd.DataFrame                  # по свечам 15m: баллы ТФ, направление, уверенность, итоговая сторона
    model: ConfidenceModel | None        # модель, обученная на всей размеченной истории
    horizon: int


def signal_history(frames: dict[str, pd.DataFrame], use_regime: bool = True, asset: str | None = None,
                   threshold: float = DEFAULT_THRESHOLD, min_hit: float = DEFAULT_MIN_HIT,
                   session_filter: bool = True, scores: dict[str, pd.Series] | None = None) -> History:
    """Итоговый сигнал на каждой закрытой свече 15m по той же логике, что и живой (кроме деривативов,
    связанного актива и соотношения прибыль/риск: у них нет истории)."""
    base = frames[HIT_TF]
    full = ind.add_all(base)
    if scores is None:
        scores = {tf: score_frame(ind.add_all(f), use_regime) for tf, f in frames.items() if tf in ALL_TFS and len(f)}
    al = align_scores(frames, scores)
    for tf in ALL_TFS:
        if tf not in al:
            al[tf] = np.nan
    dirs = {tf: np.where(al[tf] >= 0.15, 1, np.where(al[tf] <= -0.15, -1, 0)) for tf in ALL_TFS}
    d = np.where((dirs["5m"] == dirs["15m"]) & (dirs["5m"] != 0) & al["5m"].notna() & al["15m"].notna(),
                 dirs["5m"], 0)
    agree = sum((dirs[tf] == d) & (d != 0) for tf in HIGHER_TFS)
    against = sum((dirs[tf] == -d) & (d != 0) for tf in HIGHER_TFS)
    factor = np.select([d == 0, agree == len(HIGHER_TFS), against > 0, agree > 0], [0.0, 1.0, 0.0, 0.6], 0.35)
    strength = np.minimum(np.abs(np.nanmean(np.column_stack([al["5m"], al["15m"]]), axis=1)) / 0.5, 1.0)
    strength = np.nan_to_num(strength)

    horizon = max(int(round(HIT_HORIZON_MIN / 15)), 1)
    outc = barrier_outcomes(full, horizon)
    y = np.where(d > 0, outc["long"], np.where(d < 0, outc["short"], np.nan))
    n = len(full)
    pos = np.arange(n)

    # Угадываемость: последние RECENT_SIGNALS сигналов стороны, исход которых уже известен к свече t.
    hit = np.full(n, np.nan)
    hit_n = np.zeros(n, dtype=int)
    for side in (1, -1):
        ev = np.flatnonzero((d == side) & ~np.isnan(y))
        if not len(ev):
            continue
        wins = pd.Series(y[ev]).rolling(RECENT_SIGNALS, min_periods=1).mean().to_numpy()
        cnt = np.minimum(np.arange(1, len(ev) + 1), RECENT_SIGNALS)
        k = np.searchsorted(ev + horizon, pos, side="right") - 1
        m = (d == side) & (k >= 0)
        hit[m], hit_n[m] = wins[k[m]], cnt[k[m]]

    # Модель уверенности walk-forward: обучаем только на сигналах, исход которых уже известен.
    x = features(d, al["5m"], al["15m"], al["1h"], al["1d"], factor, strength)
    prob = np.full(n, np.nan)
    sig_idx = np.flatnonzero(d != 0)
    for start in range(0, n, REFIT_EVERY):
        known = sig_idx[sig_idx + horizon < start]
        model = _train(x[known], y[known]) if len(known) >= MODEL_MIN_SAMPLES else None
        seg = sig_idx[(sig_idx >= start) & (sig_idx < start + REFIT_EVERY)]
        if model is not None and len(seg):
            prob[seg] = model.predict(x[seg])
    final_model = _train(x[sig_idx], y[sig_idx]) if len(sig_idx) else None

    known_hit = (hit_n >= MIN_SAMPLES) & ~np.isnan(hit)
    hit_f = np.where(known_hit, np.clip((np.nan_to_num(hit) - 0.5) / 0.15, 0, 1), 0.5)
    heuristic = 100 * (0.4 * strength + 0.35 * factor + 0.25 * hit_f)
    conf = np.where(np.isnan(prob), heuristic, prob * 100) * (d != 0)
    sess = session_mask(base.index + _bar(base, HIT_TF), asset) if session_filter else np.ones(n, dtype=bool)
    persisted = np.r_[False, d[1:] == d[:-1]] & (d != 0)
    hit_ok = ~(known_hit & (np.nan_to_num(hit, nan=1.0) < min_hit))

    side = np.zeros(n, dtype=int)
    for i in range(n):
        if d[i] == 0 or not sess[i] or not hit_ok[i]:
            continue
        held = i > 0 and side[i - 1] == d[i]
        if conf[i] >= (threshold - HYSTERESIS if held else threshold) and (held or persisted[i]):
            side[i] = d[i]

    frame = pd.DataFrame({**{f"s_{tf}": al[tf] for tf in ALL_TFS}, "dir": d, "factor": factor,
                          "strength": strength, "hit": hit, "hit_n": hit_n, "prob": prob, "confidence": conf,
                          "session": sess, "persisted": persisted, "side": side, "outcome": y}, index=base.index)
    return History(frame, final_model, horizon)


# ---------- Живой сигнал ----------

@dataclass
class Signal:
    direction: int
    passed: bool              # прошёл все фильтры
    strong: bool              # подтверждён старшими ТФ
    confidence: float         # 0..100
    threshold: float
    hit_rate: float
    hit_n: int
    confirmation: Confirmation
    reasons: list[str] = field(default_factory=list)
    calibrated: bool = False  # уверенность дала модель, а не формула
    rr: float | None = None

    @property
    def label(self) -> str:
        if not self.passed:
            return "Нет сигнала"
        return ("Сильный " if self.strong else "") + DIR_WORD[self.direction]


def final_signal(results: dict[str, Analysis], hit: tuple[float, int] | None = None,
                 threshold: float = DEFAULT_THRESHOLD, min_hit: float = DEFAULT_MIN_HIT,
                 prob: float | None = None, model_n: int = 0, persisted: bool | None = None,
                 held: bool = False, session_ok: bool | None = None, session_word: str = "",
                 check_rr: bool = True) -> Signal:
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
    reasons.append(f"5m и 15m за {DIR_WORD[d]}, сила консенсуса {strength:.0%}")
    reasons.append(conf.label)
    if prob is not None and not np.isnan(prob):
        confidence = 100 * float(prob)
        reasons.append(f"модель на {model_n} прошлых сигналах: вероятность пройти {BARRIER_ATR:g} ATR в сторону "
                       f"сигнала раньше, чем против, {confidence:.0f}%")
    else:
        hit_f = float(np.clip((hit_rate - 0.5) / 0.15, 0, 1)) if known else 0.5
        confidence = 100 * (0.4 * strength + 0.35 * conf.factor + 0.25 * hit_f)
        reasons.append("истории мало для модели, уверенность по формуле")
    if known:
        reasons.append(f"недавние сигналы {DIR_WORD[d]} на {HIT_TF} доходили до цели раньше стопа "
                       f"в {hit_rate:.0%} случаев из {hit_n}")
    else:
        reasons.append(f"мало недавних сигналов {DIR_WORD[d]} для проверки ({hit_n}), угадываемость не учтена")
    need = threshold - HYSTERESIS if held else threshold
    passed = confidence >= need
    if held:
        reasons.append(f"сигнал уже действует, порог удержания {need:.0f}%")
    if not passed:
        reasons.append(f"уверенность {confidence:.0f}% ниже порога {need:.0f}%")
    if known and hit_rate < min_hit:
        passed = False
        reasons.append(f"угадываемость {hit_rate:.0%} ниже минимума {min_hit:.0%}")
    if persisted is False and not held:
        passed = False
        reasons.append("направление держится только одну свечу 15m: ждём подтверждения на следующей")
    if session_ok is False:
        passed = False
        reasons.append("основная сессия базового рынка закрыта" + (f" ({session_word})" if session_word else "")
                       + ": на тонком рынке сигналы ненадёжны")
    rr = None
    tr = results.get(HIT_TF).trade if HIT_TF in results else {}
    if tr and np.sign(tr["target1"] - tr["entry"]) == d:
        rr = float(tr.get("rr", 0.0))
        if check_rr and rr < MIN_RR:
            passed = False
            reasons.append(f"прибыль/риск до первой цели на {HIT_TF} {rr:.1f}, хуже 1:{MIN_RR:g}")
    return Signal(d, passed, conf.confirmed, float(confidence), threshold, hit_rate, hit_n, conf, reasons,
                  calibrated=prob is not None and not np.isnan(prob), rr=rr)


def live_features(results: dict[str, Analysis]) -> np.ndarray:
    conf = mtf_confirmation(results)
    s = {tf: results[tf].score if tf in results else np.nan for tf in ALL_TFS}
    strength = min(abs(float(np.nanmean([s["5m"], s["15m"]]))) / 0.5, 1.0)
    return features([conf.direction], [s["5m"]], [s["15m"]], [s["1h"]], [s["1d"]], [conf.factor], [strength])


def evaluate(results: dict[str, Analysis], candles: pd.DataFrame | None = None, scores: pd.Series | None = None,
             threshold: float = DEFAULT_THRESHOLD, min_hit: float = DEFAULT_MIN_HIT,
             use_regime: bool = True, frames: dict[str, pd.DataFrame] | None = None,
             history: History | None = None, asset: str | None = None, session_filter: bool = True,
             now: pd.Timestamp | None = None, check_rr: bool = True) -> Signal:
    """Итоговый сигнал. С frames (свечи всех ТФ) или готовой history используются модель уверенности,
    угадываемость по стопу и цели, защита от мигания и фильтр сессий. Только с candles 15m —
    упрощённый расчёт: угадываемость по стопу и цели и формула уверенности."""
    from .forecast import horizon_bars, score_series
    from .sessions import SESSION_WORD, in_session

    d = mtf_confirmation(results).direction
    if history is None and frames and HIT_TF in frames and len(frames[HIT_TF]) > 120:
        history = signal_history(frames, use_regime, asset, threshold, min_hit, session_filter)
    session_ok = None
    if session_filter and asset:
        session_ok = in_session(now or pd.Timestamp.now(tz="UTC"), asset)
    word = SESSION_WORD.get(asset or "", "")
    if history is not None:
        h = history.frame
        hit = (float("nan"), 0)
        if d:
            rows = h[h["dir"] == d]
            if len(rows):
                hit = (float(rows["hit"].iloc[-1]), int(rows["hit_n"].iloc[-1]))
        prob = float(history.model.predict(live_features(results))[0]) if history.model is not None and d else None
        prev = h.iloc[-2] if len(h) > 1 else None
        persisted = bool(prev is not None and prev["dir"] == d)
        held = bool(prev is not None and prev["side"] == d and d != 0)
        return final_signal(results, hit, threshold, min_hit, prob,
                            history.model.samples if history.model else 0, persisted, held, session_ok, word, check_rr)
    hit = None
    if d and candles is not None and len(candles) > 120:
        if scores is None:
            scores = score_series(candles, last_n=len(candles), use_regime=use_regime)
        hit = recent_hit_rate(candles, scores, d, horizon_bars(candles, HIT_HORIZON_MIN))
    return final_signal(results, hit, threshold, min_hit, session_ok=session_ok, session_word=word,
                        check_rr=check_rr)
