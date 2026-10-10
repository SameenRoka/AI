# Crypto AI: 7-day dashboard + spot trading bot

Python **3.11 or 3.12**. Default: BTC/AUD, paper mode, no exchange credentials.

**Risk warning:** An ML forecast is not a reliable way to make money. This demo can lose money, even with stop-loss logic. Backtests are not proof of future performance. Fees, taxes, liquidity, spread, slippage, and outages matter.

## Run it on your computer

1. Download and unzip the project, open a Terminal in the extracted folder.
2. Optionally make a virtual environment: `python -m venv .venv` and activate it.
3. Install dependencies: `python -m pip install -r requirements.txt`
4. Dashboard: `python -m streamlit run app.py` — opens a local interactive chart.
5. Paper bot: `python bot.py --once`. You will see paper_state.json with trades/balance. If the daily signal is HOLD, no order is placed.
6. To keep checking each hour (only one evaluation per *completed* UTC daily candle): `python bot.py --loop` — **the machine must remain running**. Or schedule `python bot.py --once` daily using your OS task scheduler (a scheduler is preferable to an in-process loop).

The dashboard predicts seven calendar days forward each time you load it, using *completed* daily candles only. It doesn't know future prices; the forecast is an estimate. The ML models use features such as moving-average gaps, past returns, volatility, relative volume, and RSI. A rolling-time chronological **90-day holdout** compares return prediction errors with the naive "no price change" baseline. This is not a full trading backtest.

## Where to paste/upload it so you can see the charts online

**Streamlit Community Cloud**: Put `app.py`, `strategy.py`, and `requirements.txt` into a **private GitHub repo**. Go to https://share.streamlit.io, select **Create app**, choose your repository and `app.py`, then deploy. Your interactive chart gets its own URL. Public-market readings require no API key. Do not upload `.env`, `live_state.json`, private API keys, or a real-trading `bot.py` configured to trade. Streamlit Cloud does not provide a persistent trading daemon.

**Replit alternative**: Make a Python Repl, upload these files, install requirements, and run `streamlit run app.py --server.address 0.0.0.0 --server.port 8501`. Host availability and sleeping vary by plan. Neither a web dashboard nor Google Colab alone guarantees 24/7 bot execution.

## Real order execution — opt-in only

This example implements Kraken spot market BUY/SELL orders (AUD or USD as shown in PAIR) through `ccxt`. Real mode requires manual action: copy `.env.example` to `.env`, set `LIVE_TRADING=true`, provide a dedicated Kraken trading-only key, and fill the **exact** `LIVE_ACKNOWLEDGEMENT` string. Keep the file on a trusted, persistent machine. **Never paste API keys in the chat or commit them to GitHub.** Use Kraken permissions to read balances + modify orders ONLY; leave withdrawals and transfers OFF. In-person review and lots of paper testing are strongly recommended before enabling live trades.

Limits: MAX_ORDER_QUOTE defaults to 25 (hard capped 50), MAX_EXPOSURE_QUOTE defaults to 100 (hard capped 200). These values are in the **quote currency**, e.g. AUD for BTC/AUD. No margin/short/leverage. Sells only assets that the bot's local state tracks buying, so deleting state can prevent it selling. Pending orders must be reconciled before further orders; on API timeout the bot deliberately stops rather than blindly retries. One bot process only, one trading pair per local state file. Protect `live_state.json` and back it up. Manual trading or withdrawals can desynchronise the record. This is a teaching example, not production execution software.

The software 5% stop-loss checks the last completed daily close, **not continuously**, and is **not an exchange stop-loss**. It does not guarantee a maximum loss. The bot uses the last close for sizing, but market order execution prices can differ. Paper fees assume 0.4%, which may not match your account; paper balances do not emulate live spread/slippage.

## Troubleshooting

- Kraken data is daily candles; last unfinished UTC candle is intentionally ignored.
- The Kraken endpoint only serves a finite window (up to ~720 candles), not unlimited history.
- Some trade pairs are not permitted in all regions; try BTC/AUD, ETH/AUD, BTC/USD, ETH/USD, SOL/USD where available.
- A HOLD signal is common; the model must beat the no-change baseline for directional signals.
- `live_state.json` `pending` with `id: null` after uncertain API response **must be manually reconciled using Kraken's order history** before running live again. Don't simply delete the file.
- To test the bot again on the SAME completed candle, use synthetic testing rather than deleting real-world state; idempotence is intentional.
