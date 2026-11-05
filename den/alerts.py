"""Уведомления в Telegram: смена консенсуса и совпадение сигналов на нескольких таймфреймах.

Работает отдельным процессом, дашборд для этого держать открытым не нужно:
    python -m den.alerts            # проверять каждые DEN_ALERT_INTERVAL секунд (по умолчанию 300)
    python -m den.alerts --once     # одна проверка, удобно для cron
    python -m den.alerts --test     # отправить тестовое сообщение
    python -m den.alerts --chat-id  # показать chat id тех, кто написал боту

Настройки в переменных окружения:
    DEN_TG_TOKEN       токен бота от @BotFather
    DEN_TG_CHAT        chat id, куда слать (узнать: написать боту и запустить --chat-id)
    DEN_ALERT_ASSETS   активы через запятую, по умолчанию BYBIT:GOLD,BYBIT:SILVER,BYBIT:AAPL
    DEN_ALERT_AGREE    таймфреймы для совпадения сигналов, по умолчанию 5m,15m,1h
    DEN_ALERT_INTERVAL период проверки в секундах, по умолчанию 300
    DEN_ALERT_STATE    файл с последним состоянием, по умолчанию ~/.den_alerts.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from .analysis import Analysis, analyze, multi_timeframe
from .data import ASSETS, TIMEFRAMES, load_ohlcv
from .derivatives import derivative_votes, load_derivatives

TG_API = "https://api.telegram.org/bot{token}/{method}"
DEFAULT_ASSETS = "BYBIT:GOLD,BYBIT:SILVER,BYBIT:AAPL"
DEFAULT_AGREE = "5m,15m,1h"
NAMES = {v: k for k, v in ASSETS.items()}


class TelegramError(RuntimeError):
    pass


def direction(score: float) -> int:
    return 1 if score >= 0.15 else -1 if score <= -0.15 else 0


DIR_WORD = {1: "лонг", -1: "шорт", 0: "нейтрально"}
EVENT_WORD = {"consensus": "смена консенсуса", "agree": "совпадение таймфреймов"}
DIR_MARK = {1: "🟢", -1: "🔴", 0: "⚪"}


def _env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in (os.environ.get(name) or default).split(",") if x.strip()]


def _proxies() -> dict | None:
    return {"https": os.environ["DEN_PROXY"]} if os.environ.get("DEN_PROXY") else None


def tg_call(method: str, token: str | None = None, **params) -> dict:
    import requests

    token = token or os.environ.get("DEN_TG_TOKEN", "").strip()
    if not token:
        raise TelegramError("Не задан токен бота. Создайте бота у @BotFather и задайте DEN_TG_TOKEN.")
    try:
        r = requests.post(TG_API.format(token=token, method=method), json=params, timeout=20, proxies=_proxies())
        js = r.json()
    except (requests.RequestException, ValueError) as e:
        raise TelegramError(f"Telegram недоступен: {e}") from e
    if not js.get("ok"):
        raise TelegramError(f"Telegram: {js.get('description', r.status_code)}")
    return js["result"]


def send_telegram(text: str, token: str | None = None, chat_id: str | None = None) -> None:
    chat_id = chat_id or os.environ.get("DEN_TG_CHAT", "").strip()
    if not chat_id:
        raise TelegramError("Не задан chat id. Напишите боту любое сообщение, запустите "
                            "python -m den.alerts --chat-id и задайте DEN_TG_CHAT.")
    tg_call("sendMessage", token, chat_id=chat_id, text=text, disable_web_page_preview=True)


def chat_ids(token: str | None = None) -> list[tuple[str, str]]:
    out = {}
    for u in tg_call("getUpdates", token):
        chat = (u.get("message") or u.get("channel_post") or {}).get("chat")
        if chat:
            out[str(chat["id"])] = chat.get("username") or chat.get("title") or chat.get("first_name") or ""
    return list(out.items())


def analyze_symbol(symbol: str, use_deriv: bool = True) -> tuple[dict[str, Analysis], float, str]:
    """Тот же расчёт, что в дашборде: консенсус по всем таймфреймам, с деривативами Bybit."""
    results = {}
    for tf in TIMEFRAMES:
        try:
            df = load_ohlcv(symbol, tf)
        except Exception:  # noqa: BLE001  таймфрейм не загрузился: считаем по остальным
            continue
        extra = []
        if use_deriv and symbol.startswith("BYBIT:"):
            try:
                extra = derivative_votes(load_derivatives(symbol[6:], tf), df["close"])
            except Exception:  # noqa: BLE001
                pass
        try:
            results[tf] = analyze(df, symbol, tf, extra_votes=extra)
        except ValueError:
            continue
    if not results:
        raise ValueError(f"не удалось загрузить данные {symbol}")
    score, label = multi_timeframe(results)
    return results, score, label


def snapshot(results: dict[str, Analysis], score: float, label: str, agree_tfs: list[str]) -> dict:
    dirs = {tf: direction(r.score) for tf, r in results.items()}
    agree_dirs = {dirs[tf] for tf in agree_tfs if tf in dirs}
    agree = agree_dirs.pop() if len(agree_dirs) == 1 and all(tf in dirs for tf in agree_tfs) else 0
    price = next(iter(results.values())).price
    return {"label": label, "dir": direction(score), "score": round(score, 4), "agree": agree,
            "tf": {tf: r.label for tf, r in results.items()}, "price": price}


def detect_events(prev: dict | None, cur: dict) -> list[str]:
    """Что изменилось с прошлой проверки. На первой проверке (prev=None) ничего не шлём."""
    if prev is None:
        return []
    events = []
    if cur["dir"] != prev.get("dir"):
        events.append("consensus")
    if cur["agree"] and cur["agree"] != prev.get("agree"):
        events.append("agree")
    return events


def format_message(symbol: str, cur: dict, prev: dict | None, events: list[str], agree_tfs: list[str],
                   results: dict[str, Analysis] | None = None) -> str:
    name = NAMES.get(symbol, symbol)
    lines = []
    if "consensus" in events:
        was = prev.get("label", "?") if prev else "?"
        lines.append(f"{DIR_MARK[cur['dir']]} {name}: консенсус сменился, {was} → {cur['label']}")
    if "agree" in events:
        lines.append(f"{DIR_MARK[cur['agree']]} {name}: все таймфреймы {', '.join(agree_tfs)} "
                     f"за {DIR_WORD[cur['agree']]}")
    lines.append(f"Цена {cur['price']:,.2f}, за лонг {(cur['score'] + 1) / 2 * 100:.0f}%")
    lines.append(" · ".join(f"{tf}: {lbl}" for tf, lbl in cur["tf"].items()))
    if results:
        r = results.get("15m") or next(iter(results.values()))
        deriv = [v for v in r.votes if v.name in ("Funding", "Open interest", "Long/Short")]
        lines += [f"{v.name}: {v.reason}" for v in deriv]
        if r.trade:
            lines.append(f"Ориентир {r.interval}: стоп {r.trade['stop']:,.2f}, цель {r.trade['target1']:,.2f}")
    lines.append(f"DEN · {datetime.now():%d.%m %H:%M} · не инвестиционная рекомендация")
    return "\n".join(lines)


def state_path() -> Path:
    return Path(os.environ.get("DEN_ALERT_STATE") or Path.home() / ".den_alerts.json")


def load_state() -> dict:
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    state_path().write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def check_once(send=send_telegram, log=print) -> int:
    """Одна проверка всех активов. Возвращает число отправленных сообщений."""
    assets, agree_tfs = _env_list("DEN_ALERT_ASSETS", DEFAULT_ASSETS), _env_list("DEN_ALERT_AGREE", DEFAULT_AGREE)
    state, sent = load_state(), 0
    for sym in assets:
        try:
            results, score, label = analyze_symbol(sym)
        except Exception as e:  # noqa: BLE001
            log(f"{sym}: ошибка, {e}")
            continue
        cur = snapshot(results, score, label, agree_tfs)
        prev = state.get(sym)
        events = detect_events(prev, cur)
        log(f"{datetime.now():%H:%M:%S} {sym}: {label} ({cur['score']:+.2f})"
            + (f", события: {', '.join(EVENT_WORD[e] for e in events)}" if events else "")
            + (", первая проверка, запомнил состояние" if prev is None else ""))
        if events:
            try:
                send(format_message(sym, cur, prev, events, agree_tfs, results))
                sent += 1
            except TelegramError as e:
                log(f"не отправилось: {e}")
                continue  # состояние не обновляем, чтобы попробовать снова
        state[sym] = cur
    save_state(state)
    return sent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m den.alerts", description="Уведомления DEN в Telegram")
    p.add_argument("--once", action="store_true", help="одна проверка и выход")
    p.add_argument("--test", action="store_true", help="отправить тестовое сообщение")
    p.add_argument("--chat-id", action="store_true", help="показать chat id тех, кто написал боту")
    p.add_argument("--interval", type=int, default=int(os.environ.get("DEN_ALERT_INTERVAL") or 300))
    a = p.parse_args(argv)
    try:
        if a.chat_id:
            ids = chat_ids()
            if not ids:
                print("Боту ещё никто не писал. Отправьте ему любое сообщение в Telegram и повторите.")
            for cid, who in ids:
                print(f"{cid}  {who}")
            return 0
        if a.test:
            send_telegram("DEN: тестовое сообщение, уведомления настроены ✅")
            print("Отправлено.")
            return 0
    except TelegramError as e:
        print(e, file=sys.stderr)
        return 1
    if a.once:
        check_once()
        return 0
    print(f"Слежу за {os.environ.get('DEN_ALERT_ASSETS') or DEFAULT_ASSETS} каждые {a.interval} с. "
          "Остановить: Ctrl+C.")
    try:
        while True:
            try:
                check_once()
            except Exception as e:  # noqa: BLE001  сеть пропала и т.п.: не падаем, пробуем в следующий раз
                print(f"ошибка проверки: {e}")
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("Остановлено.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
