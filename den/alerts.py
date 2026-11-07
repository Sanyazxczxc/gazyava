"""Уведомления в Telegram: смена консенсуса и совпадение сигналов на нескольких таймфреймах.

Уведомления выключены, пока их не включить. Включаются и настраиваются в любой момент в боковой панели
дашборда (раздел «Уведомления в Telegram»); настройки сохраняются в ~/.den/alerts.json и подхватываются
на следующей проверке, перезапуск не нужен. Пока открыт дашборд, проверки идут в нём самом.
Чтобы уведомления шли без дашборда, запустите отдельный процесс:
    python -m den.alerts            # следить (берёт настройки из того же файла)
    python -m den.alerts --once     # одна проверка, удобно для cron
    python -m den.alerts --test     # отправить тестовое сообщение
    python -m den.alerts --chat-id  # показать chat id тех, кто написал боту
    python -m den.alerts --on / --off   # включить или выключить без дашборда

Переменные окружения (необязательно, используются, если в файле настроек пусто):
    DEN_TG_TOKEN, DEN_TG_CHAT   токен бота и chat id
    DEN_ALERT_CONFIG            другой путь к файлу настроек
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .analysis import Analysis, analyze, multi_timeframe
from .data import ASSETS, TIMEFRAMES, load_ohlcv
from .derivatives import derivative_votes, load_derivatives
from .signals import DEFAULT_MIN_HIT, DEFAULT_THRESHOLD, HIT_TF, Signal, evaluate

TG_API = "https://api.telegram.org/bot{token}/{method}"
DEFAULT_ASSETS = ["BYBIT:GOLD", "BYBIT:SILVER", "BYBIT:AAPL"]
DEFAULT_AGREE = ["5m", "15m", "1h"]
NAMES = {v: k for k, v in ASSETS.items()}


class TelegramError(RuntimeError):
    pass


def direction(score: float) -> int:
    return 1 if score >= 0.15 else -1 if score <= -0.15 else 0


DIR_WORD = {1: "лонг", -1: "шорт", 0: "нейтрально"}
EVENT_WORD = {"consensus": "смена консенсуса", "agree": "совпадение таймфреймов",
              "signal": "сигнал выше порога уверенности"}
DIR_MARK = {1: "🟢", -1: "🔴", 0: "⚪"}


def _proxies() -> dict | None:
    return {"https": os.environ["DEN_PROXY"]} if os.environ.get("DEN_PROXY") else None


def tg_call(method: str, token: str | None = None, **params) -> dict:
    import requests

    token = (token or load_config().token).strip()
    if not token:
        raise TelegramError("Не задан токен бота. Создайте бота у @BotFather и введите токен "
                            "в настройках уведомлений.")
    try:
        r = requests.post(TG_API.format(token=token, method=method), json=params, timeout=20, proxies=_proxies())
        js = r.json()
    except (requests.RequestException, ValueError) as e:
        # В тексте ошибки requests есть URL, а в нём токен бота: вырезаем, чтобы он не попал в логи и на экран.
        raise TelegramError(f"Telegram недоступен: {str(e).replace(token, '***')}") from None
    if not js.get("ok"):
        raise TelegramError(f"Telegram: {js.get('description', r.status_code)}")
    return js["result"]


def send_telegram(text: str, token: str | None = None, chat_id: str | None = None) -> None:
    chat_id = (chat_id or load_config().chat_id).strip()
    if not chat_id:
        raise TelegramError("Не задан chat id. Напишите боту любое сообщение и нажмите «Найти chat id» "
                            "в настройках уведомлений (или запустите python -m den.alerts --chat-id).")
    tg_call("sendMessage", token, chat_id=chat_id, text=text, disable_web_page_preview=True)


def chat_ids(token: str | None = None) -> list[tuple[str, str]]:
    out = {}
    for u in tg_call("getUpdates", token):
        chat = (u.get("message") or u.get("channel_post") or {}).get("chat")
        if chat:
            out[str(chat["id"])] = chat.get("username") or chat.get("title") or chat.get("first_name") or ""
    return list(out.items())


def analyze_symbol(symbol: str, use_deriv: bool = True,
                   frames: dict | None = None, use_cross: bool = True) -> tuple[dict[str, Analysis], float, str]:
    """Тот же расчёт, что в дашборде: консенсус по всем таймфреймам, с деривативами Bybit и связкой
    золото–серебро. Если передан словарь frames, в него складываются загруженные свечи по таймфреймам."""
    from .cross import cross_votes, partner

    results = {}
    other = partner(symbol) if use_cross else None
    for tf in TIMEFRAMES:
        try:
            df = load_ohlcv(symbol, tf)
        except Exception:  # noqa: BLE001  таймфрейм не загрузился: считаем по остальным
            continue
        if frames is not None:
            frames[tf] = df
        extra = []
        if use_deriv and symbol.startswith("BYBIT:"):
            try:
                extra = derivative_votes(load_derivatives(symbol[6:], tf), df["close"])
            except Exception:  # noqa: BLE001
                pass
        if other:
            try:
                extra += cross_votes(symbol, df, load_ohlcv(other, tf))
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


def snapshot(results: dict[str, Analysis], score: float, label: str, agree_tfs: list[str],
             signal: Signal | None = None) -> dict:
    dirs = {tf: direction(r.score) for tf, r in results.items()}
    agree_dirs = {dirs[tf] for tf in agree_tfs if tf in dirs}
    agree = agree_dirs.pop() if len(agree_dirs) == 1 and all(tf in dirs for tf in agree_tfs) else 0
    price = next(iter(results.values())).price
    snap = {"label": label, "dir": direction(score), "score": round(score, 4), "agree": agree,
            "tf": {tf: r.label for tf, r in results.items()}, "price": price}
    if signal is not None:
        snap["signal"] = signal.direction if signal.passed else 0
        snap["signal_label"] = signal.label
        snap["confidence"] = round(signal.confidence, 1)
        snap["confirm"] = signal.confirmation.label
    return snap


def detect_events(prev: dict | None, cur: dict, on_consensus: bool = True, on_agree: bool = True,
                  on_signal: bool = False) -> list[str]:
    """Что изменилось с прошлой проверки. На первой проверке (prev=None) ничего не шлём."""
    if prev is None:
        return []
    events = []
    if on_consensus and cur["dir"] != prev.get("dir"):
        events.append("consensus")
    if on_agree and cur["agree"] and cur["agree"] != prev.get("agree"):
        events.append("agree")
    if on_signal and cur.get("signal") and cur["signal"] != prev.get("signal"):
        events.append("signal")
    return events


def format_message(symbol: str, cur: dict, prev: dict | None, events: list[str], agree_tfs: list[str],
                   results: dict[str, Analysis] | None = None, risk=None) -> str:
    name = NAMES.get(symbol, symbol)
    lines = []
    if "consensus" in events:
        was = prev.get("label", "?") if prev else "?"
        lines.append(f"{DIR_MARK[cur['dir']]} {name}: консенсус сменился, {was} → {cur['label']}")
    if "agree" in events:
        lines.append(f"{DIR_MARK[cur['agree']]} {name}: все таймфреймы {', '.join(agree_tfs)} "
                     f"за {DIR_WORD[cur['agree']]}")
    if "signal" in events:
        lines.append(f"{DIR_MARK[cur['signal']]} {name}: {cur['signal_label']}, уверенность {cur['confidence']:.0f}%")
    lines.append(f"Цена {cur['price']:,.2f}, за лонг {(cur['score'] + 1) / 2 * 100:.0f}%")
    if "signal_label" in cur:
        lines.append(f"Сигнал: {cur['signal_label']} ({cur['confidence']:.0f}%), {cur['confirm']}")
    lines.append(" · ".join(f"{tf}: {lbl}" for tf, lbl in cur["tf"].items()))
    if results:
        r = results.get("15m") or next(iter(results.values()))
        deriv = [v for v in r.votes if v.name in ("Funding", "Open interest", "Long/Short")]
        lines += [f"{v.name}: {v.reason}" for v in deriv]
        if r.regime:
            lines.append(f"Режим рынка {r.interval}: {r.regime.label.lower()}")
        if r.trade:
            lines.append(f"Ориентир {r.interval}: стоп {r.trade['stop']:,.2f}, цель {r.trade['target1']:,.2f}")
    if risk is not None and risk.elevated:
        lines.append(f"⚠️ Геополитика: {risk.label}, размер позиции уменьшить до {risk.size_mult:.0%} от обычного")
    lines.append(f"DEN · {datetime.now():%d.%m %H:%M} · не инвестиционная рекомендация")
    return "\n".join(lines)


@dataclass
class AlertConfig:
    enabled: bool = False
    token: str = ""
    chat_id: str = ""
    assets: list[str] = field(default_factory=lambda: list(DEFAULT_ASSETS))
    on_consensus: bool = True       # смена общего консенсуса
    on_agree: bool = True           # совпадение сигналов на agree_tfs
    agree_tfs: list[str] = field(default_factory=lambda: list(DEFAULT_AGREE))
    interval: int = 300             # секунд между проверками
    on_signal: bool = True          # итоговый сигнал прошёл порог уверенности
    threshold: float = DEFAULT_THRESHOLD   # порог уверенности, %
    min_hit: float = DEFAULT_MIN_HIT       # минимальная угадываемость на недавней истории


def home_dir() -> Path:
    return Path(os.environ.get("DEN_HOME") or Path.home() / ".den")


def config_path() -> Path:
    return Path(os.environ.get("DEN_ALERT_CONFIG") or home_dir() / "alerts.json")


def state_path() -> Path:
    return Path(os.environ.get("DEN_ALERT_STATE") or home_dir() / "alerts_state.json")


def _read_json(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    """Записать атомарно и только для владельца (0600): в настройках лежит токен бота."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    try:
        os.chmod(tmp, 0o600)  # если файл уже был с другими правами
    except OSError:
        pass
    tmp.replace(path)


def load_config() -> AlertConfig:
    d = _read_json(config_path())
    known = {k: v for k, v in d.items() if k in AlertConfig.__dataclass_fields__}
    cfg = AlertConfig(**known)
    cfg.token = cfg.token or os.environ.get("DEN_TG_TOKEN", "")
    cfg.chat_id = cfg.chat_id or os.environ.get("DEN_TG_CHAT", "")
    cfg.interval = max(int(cfg.interval or 300), 60)
    return cfg


def save_config(cfg: AlertConfig) -> None:
    old_enabled = load_config().enabled
    _write_json(config_path(), asdict(cfg))
    if cfg.enabled and not old_enabled:
        save_state({})  # после включения первая проверка только запоминает состояние, без старых событий


def load_state() -> dict:
    return _read_json(state_path())


def save_state(state: dict) -> None:
    _write_json(state_path(), state)


def check_once(send=send_telegram, log=print, cfg: AlertConfig | None = None) -> int:
    """Одна проверка всех активов. Возвращает число отправленных сообщений. Выключено — ничего не делает."""
    cfg = cfg or load_config()
    if not cfg.enabled:
        return 0
    state, sent = load_state(), 0
    for sym in cfg.assets:
        frames: dict = {}
        try:
            results, score, label = analyze_symbol(sym, frames=frames)
        except Exception as e:  # noqa: BLE001
            log(f"{sym}: ошибка, {e}")
            continue
        try:
            from .news import asset_key

            sig = evaluate(results, frames.get(HIT_TF), threshold=cfg.threshold, min_hit=cfg.min_hit,
                           frames=frames, asset=asset_key(sym))
        except Exception:  # noqa: BLE001
            sig = None
        cur = snapshot(results, score, label, cfg.agree_tfs, sig)
        prev = state.get(sym)
        events = detect_events(prev, cur, cfg.on_consensus, cfg.on_agree, cfg.on_signal)
        log(f"{datetime.now():%H:%M:%S} {sym}: {label} ({cur['score']:+.2f})"
            + (f", события: {', '.join(EVENT_WORD[e] for e in events)}" if events else "")
            + (", первая проверка, запомнил состояние" if prev is None else ""))
        if events:
            try:
                send(format_message(sym, cur, prev, events, cfg.agree_tfs, results, geo_risk(sym)))
                sent += 1
            except TelegramError as e:
                log(f"не отправилось: {e}")
                continue  # состояние не обновляем, чтобы попробовать снова
        state[sym] = cur
    state["_last_check"] = datetime.now().isoformat(timespec="seconds")
    save_state(state)
    return sent


_geo_cache: dict = {}


def geo_risk(symbol: str, ttl: float = 1800):
    """Геополитический фон для сообщения: GPR и словарная разметка новостей, не чаще раза в ttl секунд."""
    from . import geo
    from .news import asset_key

    asset = asset_key(symbol)
    if asset is None:
        return None
    now = time.time()
    if not _geo_cache or now - _geo_cache.get("time", 0) > ttl:
        try:
            gpr = geo.gpr_status(geo.load_gpr())
        except Exception:  # noqa: BLE001
            gpr = None
        try:
            tags = geo.fetch_geo_news()
        except Exception:  # noqa: BLE001
            tags = []
        _geo_cache.update(time=now, gpr=gpr, tags=tags)
    return geo.risk_backdrop(_geo_cache["gpr"], _geo_cache["tags"], asset)


def last_check() -> str | None:
    return load_state().get("_last_check")


# Чтобы дашборд и отдельный процесс не слали одно и то же дважды, проверки ведёт тот, кто первым
# занял «замок»; второй ждёт, пока замок не устареет.
def _lock_path() -> Path:
    return home_dir() / "alerts.lock"


def claim_lock(owner: str, ttl: float) -> bool:
    d = _read_json(_lock_path())
    if d.get("owner") not in (None, owner) and time.time() - float(d.get("time", 0)) < ttl:
        return False
    _write_json(_lock_path(), {"owner": owner, "time": time.time()})
    return True


def run_forever(log=print, stop: threading.Event | None = None, idle_poll: int = 30) -> None:
    """Цикл проверок. Настройки перечитываются каждый раз, поэтому включать, выключать и менять их
    можно на ходу. Пока уведомления выключены, цикл просто ждёт."""
    owner = f"{os.getpid()}:{threading.get_ident()}"
    stop = stop or threading.Event()
    while not stop.is_set():
        cfg = load_config()
        wait = idle_poll
        if cfg.enabled:
            wait = cfg.interval
            if claim_lock(owner, ttl=cfg.interval * 2 + 60):
                try:
                    check_once(log=log, cfg=cfg)
                except Exception as e:  # noqa: BLE001  сеть пропала и т.п.: пробуем в следующий раз
                    log(f"ошибка проверки: {e}")
        # Ждём по кусочкам, чтобы быстро заметить включение или выключение.
        end = time.time() + wait
        while time.time() < end and not stop.is_set():
            if load_config().enabled != cfg.enabled:
                break
            stop.wait(min(idle_poll, max(end - time.time(), 0)))


_worker: threading.Thread | None = None


def start_background(log=lambda *_: None) -> threading.Thread:
    """Фоновая проверка внутри дашборда (одна на процесс)."""
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=run_forever, kwargs={"log": log}, daemon=True, name="den-alerts")
        _worker.start()
    return _worker


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m den.alerts", description="Уведомления DEN в Telegram")
    p.add_argument("--once", action="store_true", help="одна проверка и выход")
    p.add_argument("--test", action="store_true", help="отправить тестовое сообщение")
    p.add_argument("--chat-id", action="store_true", help="показать chat id тех, кто написал боту")
    p.add_argument("--on", action="store_true", help="включить уведомления")
    p.add_argument("--off", action="store_true", help="выключить уведомления")
    a = p.parse_args(argv)
    if a.on or a.off:
        cfg = load_config()
        cfg.enabled = bool(a.on)
        save_config(cfg)
        print("Уведомления " + ("включены." if a.on else "выключены."))
        return 0
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
    cfg = load_config()
    if a.once:
        if not cfg.enabled:
            print("Уведомления выключены. Включите их в дашборде или командой python -m den.alerts --on")
        check_once(cfg=cfg)
        return 0
    print(f"Настройки: {config_path()}. Уведомления сейчас " + ("включены" if cfg.enabled else "выключены")
          + "; включать, выключать и менять их можно на ходу. Остановить: Ctrl+C.")
    try:
        run_forever()
    except KeyboardInterrupt:
        print("Остановлено.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
