"""Read-only dashboard: `streamlit run app.py`. NEVER places trades."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from strategy import SUPPORTED_PAIRS, create_outlook, load_prices

st.set_page_config(page_title="Crypto AI | 7-day forecast", page_icon="📈", layout="wide")
st.title("📈 Crypto AI: 7-day outlook")
st.caption("Daily Kraken spot prices • Random Forest ML • forecasts are uncertain, not guarantees")

pair = st.sidebar.selectbox("Trading pair", SUPPORTED_PAIRS, index=0)
lookback = st.sidebar.slider("Historical chart (days)", 30, 365, 120, 30)
st.sidebar.warning("Read-only dashboard. No orders are sent from this app.")

@st.cache_data(ttl=60 * 60, show_spinner="Retrieving completed Kraken candles...")
def cached_prices(pair_value):
    return load_prices(pair_value)

@st.cache_data(ttl=60 * 60, show_spinner="Training 7 forecasting models and validating...")
def cached_outlook(pair_value, prices):
    return create_outlook(prices, pair_value)

try:
    prices = cached_prices(pair)
    outlook = cached_outlook(pair, prices)
except Exception as error:
    st.error(f"Could not load or calculate data: {error}")
    st.info("Kraken may be temporarily unavailable or rate-limited. Refresh later.")
    st.stop()

q = pair.split("/")[1]
pred7 = float(outlook.forecast.iloc[-1]["forecast"])
change7 = 100 * (pred7 / outlook.last_close - 1)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Last completed daily close", f"{outlook.last_close:,.2f} {q}")
m2.metric("Estimated close in 7 days", f"{pred7:,.2f} {q}", f"{change7:+.2f}%")
m3.metric("Model signal", outlook.signal)
m4.metric("1-day holdout MAE", f"{outlook.validation.loc[1, 'model_mae_pct']:.2f}%")
st.caption(f"Latest completed daily candle: {outlook.last_date:%Y-%m-%d} UTC. Future dates assume one calendar day per bar.")

hist = prices["close"].tail(lookback)
fore = outlook.forecast
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Historical closing price", mode="lines", line=dict(color="#0f766e", width=2)))
fig.add_trace(go.Scatter(x=list(fore.index), y=fore["high"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scatter(x=list(fore.index), y=fore["low"], mode="lines", fill="tonexty", fillcolor="rgba(99,102,241,0.16)", line=dict(width=0), name="Historic residual 10–90% band"))
fig.add_trace(go.Scatter(x=[hist.index[-1]] + list(fore.index), y=[float(hist.iloc[-1])] + list(fore["forecast"]), name="AI forecast", mode="lines+markers", line=dict(color="#6366f1", width=3, dash="dash")))
fig.update_layout(height=490, template="plotly_white", yaxis_title=f"Price ({q})", xaxis_title="Date (UTC)", legend_orientation="h", margin=dict(l=20,r=20,t=20,b=20), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
st.caption("The shaded band uses past holdout residual quantiles. It is NOT a calibrated confidence interval; prices may move far outside it.")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Next 7 completed-day estimates")
    table = fore.reset_index().copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    table = table.rename(columns={"date":"Date (UTC)","forecast":"Predicted close","low":"Low estimate","high":"High estimate","expected_return_pct":"Return vs last close (%)"})
    table = table.drop(columns=["horizon"])
    st.dataframe(table.style.format({"Predicted close":"{:,.2f}", "Low estimate":"{:,.2f}","High estimate":"{:,.2f}","Return vs last close (%)":"{:+.2f}"}), hide_index=True, use_container_width=True)
    st.download_button("Download forecast CSV", fore.to_csv().encode(), file_name=f"{pair.replace('/', '-')}-seven-day-forecast.csv", mime="text/csv")
with right:
    st.subheader("Trading rule")
    st.write(f"**{outlook.signal}** — {outlook.signal_reason}.")
    st.write(f"20-day simple moving average: **{outlook.sma_20:,.2f} {q}**")
    st.subheader("Reality check (holdout)")
    st.write("How far daily return predictions missed on a recent chronological holdout, compared with predicting no price change:")
    v = outlook.validation.loc[[1, 7]].copy()
    v.index = ["1-day", "7-day"]
    st.dataframe(v[["model_mae_pct", "naive_mae_pct"]].rename(columns={"model_mae_pct":"AI MAE (%)","naive_mae_pct":"No-change MAE (%)"}).style.format("{:.2f}"), use_container_width=True)
    if not bool(outlook.validation.loc[1, "model_beats_naive"]):
        st.warning("1-day AI predictions did NOT beat the no-change baseline on this holdout. Do not mistake a chart for an edge.")

st.divider()
st.markdown("**How to paper-trade:** Run `python bot.py --once` locally after installing the requirements. It saves trade records to `paper_state.json`. To run one check every day, use your computer's scheduler or `python bot.py --loop` while your computer stays on.")
st.error("Crypto trading can lose all invested money. This demo excludes taxes, most execution effects and exchange outages; it is not financial advice.")
