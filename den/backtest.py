"""Бэктест консенсуса как торговой системы: сделки с комиссиями и проскальзыванием.

Правила (все параметры в BTConfig):
- сигнал считается на закрытии свечи: балл консенсуса >= threshold — лонг, <= -threshold — шорт;
- вход по открытию следующей свечи, чтобы не подглядывать в будущее;
- выход по стопу или цели в ATR от цены входа, по противоположному сигналу или по истечении max_bars;
- если на одной свече задеты и стоп, и цель, считаем, что сработал стоп (осторожная оценка);
- если цена открылась за стопом или целью (гэп), исполнение по цене открытия;
- комиссия и проскальзывание списываются на входе и на выходе, без плеча, весь капитал в сделке.

Запуск из консоли: python -m den.backtest BYBIT:GOLD 15m
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind

BYBIT_TAKER_FEE = 0.00055  # 0.055% за сторону, базовая комиссия тейкера на фьючерсах Bybit


@dataclass
class BTConfig:
    threshold: float = 0.15
    fee: float = BYBIT_TAKER_FEE   # доля за одну сторону сделки
    slippage: float = 0.0002       # доля цены за одну сторону сделки
    stop_atr: float = 1.5
    target_atr: float = 3.0
    max_bars: int = 32             # сколько свечей держать позицию максимум
    exit_on_flip: bool = True      # выходить при противоположном сигнале
    allow_short: bool = True


@dataclass
class BTResult:
    config: BTConfig
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict = field(default_factory=dict)


def _side(score: float, cfg: BTConfig) -> int:
    if score >= cfg.threshold:
        return 1
    if score <= -cfg.threshold and cfg.allow_short:
        return -1
    return 0


def simulate(df: pd.DataFrame, scores: pd.Series, cfg: BTConfig | None = None) -> BTResult:
    """Прогнать сделки по баллам консенсуса scores (индекс — время свечей df)."""
    cfg = cfg or BTConfig()
    full = df if "atr" in df else ind.add_all(df)
    d = full.loc[full.index.isin(scores.index)]
    s = scores.reindex(d.index).fillna(0.0).to_numpy()
    o, h, l, c, a = (d[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "atr"))
    idx = d.index
    n = len(d)

    trades: list[dict] = []
    equity = np.ones(n)
    cash = 1.0
    pos = 0          # +1 лонг, -1 шорт, 0 вне рынка
    entry = stop = target = 0.0
    entry_i = 0
    pending = 0      # направление входа, который исполнится на открытии следующей свечи

    def close_trade(i: int, price: float, why: str) -> None:
        nonlocal cash, pos
        exit_px = price * (1 - pos * cfg.slippage)
        gross = pos * (exit_px / entry - 1)
        net = (1 - cfg.fee) * (1 + gross) * (1 - cfg.fee) - 1
        cash *= 1 + net
        trades.append({"Вход": idx[entry_i], "Выход": idx[i], "Сторона": "лонг" if pos > 0 else "шорт",
                       "Цена входа": entry, "Цена выхода": exit_px, "Свечей": i - entry_i + 1,
                       "Результат %": net * 100, "Причина": why})
        pos = 0

    for i in range(n):
        if pending and pos == 0:
            pos = pending
            entry = o[i] * (1 + pos * cfg.slippage)
            atr_i = a[i - 1] if i else a[i]
            stop, target = entry - pos * cfg.stop_atr * atr_i, entry + pos * cfg.target_atr * atr_i
            entry_i = i
        pending = 0

        if pos:
            # Гэп за стоп или цель: исполнение по открытию.
            if pos * (o[i] - stop) <= 0:
                close_trade(i, o[i], "стоп")
            elif pos * (o[i] - target) >= 0:
                close_trade(i, o[i], "цель")
            elif (l[i] <= stop) if pos > 0 else (h[i] >= stop):
                close_trade(i, stop, "стоп")
            elif (h[i] >= target) if pos > 0 else (l[i] <= target):
                close_trade(i, target, "цель")
            elif i - entry_i + 1 >= cfg.max_bars:
                close_trade(i, c[i], "время")
            elif cfg.exit_on_flip and _side(s[i], cfg) == -pos:
                close_trade(i, c[i], "разворот сигнала")

        if pos == 0 and i < n - 1:
            pending = _side(s[i], cfg)

        # Капитал по закрытию свечи, с учётом открытой позиции и будущей комиссии на выход.
        if pos:
            mtm = pos * (c[i] * (1 - pos * cfg.slippage) / entry - 1)
            equity[i] = cash * (1 - cfg.fee) * (1 + mtm) * (1 - cfg.fee)
        else:
            equity[i] = cash

    if pos:
        close_trade(n - 1, c[-1], "конец данных")
        equity[-1] = cash

    eq = pd.Series(equity, index=idx, name="Капитал")
    tr = pd.DataFrame(trades)
    return BTResult(cfg, tr, eq, metrics(tr, eq, d["close"]))


def metrics(trades: pd.DataFrame, equity: pd.Series, close: pd.Series) -> dict:
    nan = float("nan")
    hold = float(close.iloc[-1] / close.iloc[0] - 1) * 100 if len(close) > 1 else nan
    dd = float((equity / equity.cummax() - 1).min() * 100) if len(equity) else 0.0
    if trades.empty:
        return {"trades": 0, "win_rate": nan, "profit_factor": nan, "avg_trade": nan, "total_return": 0.0,
                "max_drawdown": dd, "buy_hold": hold, "avg_bars": nan, "longs": 0, "shorts": 0}
    r = trades["Результат %"]
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "avg_trade": float(r.mean()),
        "total_return": float(equity.iloc[-1] - 1) * 100,
        "max_drawdown": dd,
        "buy_hold": hold,
        "avg_bars": float(trades["Свечей"].mean()),
        "longs": int((trades["Сторона"] == "лонг").sum()),
        "shorts": int((trades["Сторона"] == "шорт").sum()),
    }


def describe(m: dict) -> str:
    if not m["trades"]:
        return "Сделок не было: консенсус ни разу не дошёл до порога."
    pf = m["profit_factor"]
    pf_s = "∞" if pf == float("inf") else f"{pf:.2f}"
    verdict = ("система в плюсе после издержек" if m["total_return"] > 0 and pf > 1
               else "после комиссий и проскальзывания система в минусе")
    return (f"{m['trades']} сделок (лонг {m['longs']}, шорт {m['shorts']}), прибыльных {m['win_rate']:.0%}, "
            f"profit factor {pf_s}, средняя сделка {m['avg_trade']:+.3f}%, итог {m['total_return']:+.2f}% "
            f"при просадке до {m['max_drawdown']:.2f}% (купить и держать: {m['buy_hold']:+.2f}%): {verdict}.")


if __name__ == "__main__":
    import sys

    from .data import load_ohlcv
    from .forecast import score_series

    sym = sys.argv[1] if len(sys.argv) > 1 else "BYBIT:GOLD"
    tf = sys.argv[2] if len(sys.argv) > 2 else "15m"
    raw = load_ohlcv(sym, tf)
    res = simulate(raw, score_series(raw, last_n=len(raw)), BTConfig())
    print(f"{sym} {tf}, параметры: {asdict(res.config)}")
    print(describe(res.metrics))
    if not res.trades.empty:
        print(res.trades.tail(10).to_string(index=False))
