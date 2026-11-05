"""Дашборд DEN. Запуск: streamlit run app.py"""
from __future__ import annotations

import os
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from den import ai
from den import indicators as ind
from den.analysis import analyze, multi_timeframe
from den.derivatives import derivative_votes, load_derivatives
from den.data import ASSETS, TIMEFRAMES, load_ohlcv
from den.forecast import HORIZONS, backtest, horizon_bars, price_ranges, score_series, verdict
from den.news import fetch_news, sentiment

st.set_page_config(page_title="DEN: анализ рынка", layout="wide")


@st.cache_data(ttl=60, show_spinner=False)
def get(symbol: str, tf: str):
    return load_ohlcv(symbol, tf)


@st.cache_data(ttl=60, show_spinner=False)
def get_deriv(symbol: str, tf: str):
    if not symbol.startswith("BYBIT:"):
        return None
    return load_derivatives(symbol[6:], tf)


@st.cache_data(ttl=600, show_spinner=False)
def get_news(symbol: str):
    return fetch_news(symbol)


@st.cache_data(ttl=300, show_spinner=False)
def get_scores(symbol: str, tf: str):
    return score_series(get(symbol, tf))


with st.sidebar:
    st.header("Настройки")
    asset = st.selectbox("Актив", list(ASSETS))
    symbol = ASSETS[asset]
    tf = st.selectbox("Таймфрейм графика", list(TIMEFRAMES), index=1)
    bars = st.slider("Свечей на графике", 50, 500, 200, 10)
    show = st.multiselect("Наложить на график", ["EMA 9/21", "EMA 50/200", "Bollinger", "Уровни"],
                          default=["EMA 9/21", "Bollinger", "Уровни"])
    if symbol.startswith("TD:") and not os.environ.get("DEN_TWELVEDATA_KEY"):
        key = st.text_input("Ключ Twelve Data", type="password",
                            help="Бесплатный ключ с twelvedata.com. Чтобы не вводить каждый раз, "
                                 "задайте переменную окружения DEN_TWELVEDATA_KEY.")
        if key:
            os.environ["DEN_TWELVEDATA_KEY"] = key.strip()
            st.cache_data.clear()
    with st.expander("ИИ-анализ (OpenRouter)", expanded=not ai.api_key()):
        if not ai.api_key():
            or_key = st.text_input("Ключ OpenRouter", type="password",
                                   help="Ключ с openrouter.ai/keys. Чтобы не вводить каждый раз, "
                                        "задайте переменную окружения DEN_OPENROUTER_KEY.")
            if or_key:
                os.environ["DEN_OPENROUTER_KEY"] = or_key.strip()
        ai_model = st.text_input("Модель", value=ai.model_name(),
                                 help="openrouter/free выбирает доступную бесплатную модель. Можно указать "
                                      "конкретную, например с суффиксом :free, или платную.")
    use_deriv = st.checkbox("Учитывать деривативы Bybit", value=True,
                            help="Funding, открытый интерес и соотношение лонг/шорт добавляются в консенсус "
                                 "как отдельные голоса. Есть только у фьючерсов Bybit.")
    if st.button("Обновить данные"):
        st.cache_data.clear()
    st.caption("Данные: Bybit, Twelve Data (ключ в DEN_TWELVEDATA_KEY) и Yahoo Finance "
               "(прокси в DEN_PROXY). Не является инвестиционной рекомендацией.")

results, errors, derivs, deriv_err = {}, {}, {}, None
with st.spinner("Загружаю данные…"):
    for t in TIMEFRAMES:
        try:
            candles = get(symbol, t)
        except Exception as e:  # noqa: BLE001
            errors[t] = str(e)
            continue
        extra = []
        if use_deriv:
            try:
                derivs[t] = get_deriv(symbol, t)
                extra = derivative_votes(derivs[t], candles["close"])
            except Exception as e:  # noqa: BLE001
                deriv_err = str(e)
        try:
            results[t] = analyze(candles, symbol, t, extra_votes=extra)
        except Exception as e:  # noqa: BLE001
            errors[t] = str(e)

if tf not in results:
    st.error(f"Не удалось загрузить {symbol} ({tf}): {errors.get(tf)}")
    st.stop()

main = results[tf]
score, label = multi_timeframe(results)

st.title(asset)
_contract = get(symbol, tf).attrs.get("contract")
if _contract:
    st.caption(f"Контракт: {_contract}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Цена", f"{main.price:,.2f}", f"{main.change_pct:+.2f}% за свечу")
c2.metric(f"Консенсус {tf}", main.label, f"лонг {main.long_pct:.0f}%")
c3.metric("Общий консенсус", label, f"лонг {(score + 1) / 2 * 100:.0f}%")
c4.metric("ATR", f"{main.atr:,.2f}", f"{main.atr / main.price * 100:.2f}% от цены", delta_color="off")

# Консенсус по таймфреймам
cols = st.columns(len(results))
for col, (t, r) in zip(cols, results.items()):
    with col:
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=r.long_pct, number={"suffix": "%"},
            title={"text": f"{t}: {r.label}"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#444"},
                   "steps": [{"range": [0, 42.5], "color": "#e45756"},
                             {"range": [42.5, 57.5], "color": "#bab0ac"},
                             {"range": [57.5, 100], "color": "#54a24b"}]}))
        g.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=0))
        st.plotly_chart(g, width="stretch")

# График
df = ind.add_all(get(symbol, tf)).tail(bars)
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
fig.add_trace(go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close, name="Цена"), 1, 1)
if "EMA 9/21" in show:
    fig.add_trace(go.Scatter(x=df.index, y=df.ema9, name="EMA9", line=dict(width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df.ema21, name="EMA21", line=dict(width=1)), 1, 1)
if "EMA 50/200" in show:
    fig.add_trace(go.Scatter(x=df.index, y=df.ema50, name="EMA50", line=dict(width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df.ema200, name="EMA200", line=dict(width=1)), 1, 1)
if "Bollinger" in show:
    for c in ("bb_up", "bb_low"):
        fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c, line=dict(width=1, dash="dot", color="gray")), 1, 1)
if "Уровни" in show:
    for y in main.levels["resistance"]:
        fig.add_hline(y=y, line=dict(color="#e45756", dash="dash", width=1), row=1, col=1)
    for y in main.levels["support"]:
        fig.add_hline(y=y, line=dict(color="#54a24b", dash="dash", width=1), row=1, col=1)
fig.add_trace(go.Bar(x=df.index, y=df["hist"], name="MACD hist",
                     marker_color=["#54a24b" if v >= 0 else "#e45756" for v in df["hist"]]), 2, 1)
fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(width=1)), 2, 1)
fig.add_trace(go.Scatter(x=df.index, y=df["signal"], name="Signal", line=dict(width=1)), 2, 1)
fig.add_trace(go.Scatter(x=df.index, y=df.rsi, name="RSI"), 3, 1)
for y in (30, 70):
    fig.add_hline(y=y, line=dict(color="gray", dash="dot", width=1), row=3, col=1)
fig.update_layout(height=750, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10),
                  legend=dict(orientation="h"))
st.plotly_chart(fig, width="stretch")

# Деривативы
d_main = derivs.get(tf)
if use_deriv and (d_main is not None or deriv_err):
    st.header("Деривативы Bybit")
    if deriv_err:
        st.warning(f"Не удалось загрузить данные деривативов: {deriv_err}")
    if d_main is not None:
        d1, d2, d3 = st.columns(3)
        if d_main.funding is not None:
            d1.metric("Funding (за 8 ч)", f"{d_main.funding * 100:+.4f}%",
                      f"среднее {d_main.funding_avg * 100:+.4f}%" if d_main.funding_avg is not None else None,
                      delta_color="off")
        if len(d_main.oi):
            oi_ch = d_main.oi.iloc[-1] / d_main.oi.iloc[0] - 1 if d_main.oi.iloc[0] else 0.0
            d2.metric("Открытый интерес", f"{d_main.oi.iloc[-1]:,.0f}", f"{oi_ch * 100:+.1f}% за {len(d_main.oi)} периодов")
        if len(d_main.buy_ratio):
            d3.metric("Аккаунтов в лонге", f"{d_main.buy_ratio.iloc[-1]:.0%}",
                      f"в шорте {1 - d_main.buy_ratio.iloc[-1]:.0%}", delta_color="off")
        dfig = make_subplots(specs=[[{"secondary_y": True}]])
        if len(d_main.oi):
            dfig.add_trace(go.Scatter(x=d_main.oi.index, y=d_main.oi, name="Открытый интерес"), secondary_y=False)
        if len(d_main.buy_ratio):
            dfig.add_trace(go.Scatter(x=d_main.buy_ratio.index, y=d_main.buy_ratio * 100, name="% аккаунтов в лонге",
                                      line=dict(dash="dot")), secondary_y=True)
        dfig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
        st.plotly_chart(dfig, width="stretch")
        st.caption("Высокий funding и сильный перекос аккаунтов в одну сторону голосуют против толпы. "
                   "Рост открытого интереса вместе с ценой подтверждает движение.")
elif use_deriv and symbol.startswith("BYBIT:"):
    st.caption("Для этого инструмента Bybit не публикует funding и открытый интерес (это спот-токен).")

# Прогноз
st.header("Прогноз")
news, news_err = [], None
try:
    news = get_news(symbol)
except Exception as e:  # noqa: BLE001
    news_err = str(e)
ns = sentiment(news) if news else None
raw = get(symbol, tf)
scores = get_scores(symbol, tf)
bts = {name: backtest(raw, scores, horizon_bars(raw, m)) for name, m in HORIZONS.items()
       if horizon_bars(raw, m) < len(scores) // 2}
v = verdict(score, ns["score"] if ns else None, bts.get("4 часа") or bts.get("1 час"))
f1, f2 = st.columns([1, 2])
with f1:
    st.metric("Прогнозный вердикт", v.label, f"уверенность: {v.confidence}", delta_color="off")
with f2:
    st.markdown("\n".join(f"- {r}" for r in v.reasons))

st.subheader("Ожидаемый диапазон цены")
rng = price_ranges(raw)
if rng:
    st.dataframe([{k: (f"{x:,.2f}" if isinstance(x, float) and k != "Ход ±%" else
                       f"±{x:.2f}%" if k == "Ход ±%" else x) for k, x in r.items()} for r in rng],
                 hide_index=True, width="stretch")
    st.caption("Цена с вероятностью около 68% (или 95%) останется в диапазоне, если волатильность "
               "будет как в последние дни. Направление этот расчёт не предсказывает.")

st.subheader(f"Как сигналы {tf} срабатывали раньше")
if bts:
    st.dataframe([{"Через": name,
                   "Сигналов лонг": b["long"]["n"], "Лонг угадал": f"{b['long']['hit']:.0%}" if b["long"]["n"] else "–",
                   "Сигналов шорт": b["short"]["n"], "Шорт угадал": f"{b['short']['hit']:.0%}" if b["short"]["n"] else "–",
                   "Рост без сигнала": f"{b['base_up']:.0%}"} for name, b in bts.items()],
                 hide_index=True, width="stretch")
    st.caption(f"Проверка на последних {len(scores)} свечах. Если «угадал» близко к 50%, сигнал на этом активе "
               "сейчас работает не лучше монетки.")
else:
    st.info("Мало истории для проверки сигналов на этом таймфрейме.")

st.subheader("Новости")
if news_err:
    st.warning(f"Не удалось загрузить новости: {news_err}")
elif not news:
    st.info("Новостей не найдено.")
else:
    st.write(f"Тон новостей за 3 дня: {ns['score']:+.2f} (позитивных {ns['positive']}, "
             f"негативных {ns['negative']}, нейтральных {ns['neutral']}).")
    for n in news[:15]:
        mark = "🟢" if n.score > 0.1 else "🔴" if n.score < -0.1 else "⚪"
        st.markdown(f"{mark} [{n.title}]({n.link}) · {n.source} · {n.published:%d.%m %H:%M}")
    st.caption("Тон оценивается по словарю финансовых фраз в заголовках, это грубая оценка.")

st.header("ИИ-анализ")
ai_slot = f"ai::{symbol}"
if st.button("Запросить ИИ-анализ", type="primary", disabled=not ai.api_key(),
             help=None if ai.api_key() else "Сначала задайте ключ OpenRouter в боковой панели."):
    ctx = ai.build_context(asset, _contract, results, score, label, verdict=v, ranges=rng, backtests=bts,
                           news=news, news_sentiment=ns)
    with st.spinner("Модель думает, бесплатные модели иногда отвечают до минуты…"):
        try:
            st.session_state[ai_slot] = (ai.analyze_with_ai(ctx, model=ai_model), datetime.now())
        except ai.AIError as e:
            st.error(str(e))
if not ai.api_key():
    st.info("Добавьте ключ OpenRouter в боковой панели (раздел «ИИ-анализ»), чтобы модель разобрала актив.")
if ai_slot in st.session_state:
    res_ai, when = st.session_state[ai_slot]
    a1, a2 = st.columns([1, 2])
    with a1:
        st.metric("Вывод ИИ", res_ai.direction.capitalize(), f"уверенность: {res_ai.confidence}", delta_color="off")
        st.caption(f"Модель: {res_ai.model} · {when:%H:%M}")
    with a2:
        st.markdown(res_ai.summary)
        if res_ai.levels:
            st.markdown(f"**Уровни:** {res_ai.levels}")
    k1, k2 = st.columns(2)
    with k1:
        if res_ai.key_factors:
            st.markdown("**Ключевые факторы**\n" + "\n".join(f"- {x}" for x in res_ai.key_factors))
    with k2:
        if res_ai.risks:
            st.markdown("**Риски**\n" + "\n".join(f"- {x}" for x in res_ai.risks))
    st.caption("Модель видит только данные этого дашборда и может ошибаться. Не является инвестиционной рекомендацией.")

st.header("Разбор по таймфрейму " + tf)
left, right = st.columns([3, 2])
with left:
    st.subheader("Разбор")
    st.markdown(main.summary)
    if main.trade:
        st.table({k: [f"{v:,.2f}"] for k, v in {"Вход": main.trade["entry"], "Стоп": main.trade["stop"],
                                                  "Цель 1": main.trade["target1"], "Цель 2": main.trade["target2"]}.items()})
    st.write("Сопротивления:", ", ".join(f"{x:,.2f}" for x in main.levels["resistance"]) or "нет")
    st.write("Поддержки:", ", ".join(f"{x:,.2f}" for x in main.levels["support"]) or "нет")
with right:
    st.subheader("Голоса индикаторов")
    st.dataframe(
        [{"Индикатор": v.name, "Голос": "лонг" if v.score > 0.1 else "шорт" if v.score < -0.1 else "нейтр.",
          "Сила": round(v.score, 2), "Почему": v.reason} for v in main.votes],
        hide_index=True, width="stretch")

if errors:
    st.caption("Не загрузились таймфреймы: " + ", ".join(errors))
