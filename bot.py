"""Spot-only daily trading bot. PAPER by default. Live is two-switch opt-in.

Run `python bot.py --once` daily with a task scheduler, or `python bot.py --loop`
on a computer/server you control. Requires persistent local state; NOT intended
for ephemeral/serverless jobs or public Streamlit deployments.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
from dotenv import load_dotenv
from strategy import SUPPORTED_PAIRS, create_outlook, load_prices

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
PAIR = os.getenv("PAIR", "BTC/AUD")
MAX_ORDER = min(float(os.getenv("MAX_ORDER_QUOTE", "25")), 50.0)  # hard cap 50 in quote currency
MAX_EXPOSURE = min(float(os.getenv("MAX_EXPOSURE_QUOTE", "100")), 200.0)
FEE_ESTIMATE = 0.004  # rough *paper* fee estimate; actual fees vary
PAPER_START = 500.0
REAL_MODE = os.getenv("LIVE_TRADING", "false").lower() == "true"
REQUIRED_ACK = "I_ACCEPT_THAT_LIVE_TRADES_CAN_LOSE_MONEY"
STATE_FILE = ROOT / ("live_state.json" if REAL_MODE else "paper_state.json")


def initial_state():
    return {"pair": PAIR, "cash": PAPER_START, "units": 0.0,
            "average_entry": 0.0, "last_checked_day": "", "pending": None, "orders": []}


def read_state():
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        if state["pair"] != PAIR:
            raise RuntimeError(f"Existing state is for {state['pair']}, not {PAIR}. Use one state file per pair.")
        return state
    return initial_state()


def save_state(state):
    # Atomic replacement reduces corruption risk on shutdown. Don't run two copies at once.
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2))
    os.replace(temp, STATE_FILE)


def log_order(state, side, amount, price, fee, mode, order_id=None):
    state["orders"].append({"utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode, "side": side, "units": amount, "reference_price": price,
        "estimated_fee": fee, "exchange_order_id": order_id})
    state["orders"] = state["orders"][-300:]


def paper_step(state, signal, px):
    quote = PAIR.split("/")[1]
    if signal == "BUY":
        spend = min(MAX_ORDER, MAX_EXPOSURE - state["units"] * px,
                    state["cash"] / (1 + FEE_ESTIMATE))
        if spend >= 5:  # approximate floor; exchange minima differ
            units = spend / px
            fee = spend * FEE_ESTIMATE
            old = state["units"]
            state["average_entry"] = (old * state["average_entry"] + spend + fee) / (old + units)
            state["units"] += units
            state["cash"] -= spend + fee
            log_order(state, "BUY", units, px, fee, "paper")
            print(f"PAPER BUY {units:.8f} units for {spend:.2f} {quote} (+{fee:.2f} estimated fee)")
        else:
            print("PAPER HOLD: no available budget/cash")
    elif signal == "SELL" and state["units"] > 0:
        units = state["units"]
        gross = units * px
        fee = gross * FEE_ESTIMATE
        state["cash"] += gross - fee
        state["units"] = 0.0
        state["average_entry"] = 0.0
        log_order(state, "SELL", units, px, fee, "paper")
        print(f"PAPER SELL {units:.8f} units for {gross - fee:.2f} {quote} after estimated fee")
    else:
        print("PAPER HOLD: no trade")
    print(f"PAPER VALUE ≈ {state['cash'] + state['units'] * px:.2f} {quote}")


def private_exchange():
    key = os.getenv("KRAKEN_API_KEY", "")
    secret = os.getenv("KRAKEN_API_SECRET", "")
    if (os.getenv("LIVE_ACKNOWLEDGEMENT") != REQUIRED_ACK or not key or not secret):
        raise RuntimeError("LIVE mode needs both API keys and exact LIVE_ACKNOWLEDGEMENT in .env")
    exchange = ccxt.kraken({"apiKey": key, "secret": secret,
                           "enableRateLimit": True, "timeout": 20000})
    exchange.load_markets()
    if PAIR not in exchange.markets or not exchange.markets[PAIR].get("spot"):
        raise RuntimeError(f"Spot market {PAIR} not available")
    if exchange.has.get("createMarketOrder") is not True:
        raise RuntimeError("Market orders unsupported")
    return exchange


def reconcile_pending(state, exchange):
    pending = state.get("pending")
    if not pending:
        return True
    order_id = pending.get("id")
    if not order_id:
        print("STOP: uncertain order with no exchange ID; inspect Kraken and live_state.json manually.")
        return False
    order = exchange.fetch_order(order_id, PAIR)
    status = order.get("status")
    if status not in ("closed", "canceled", "expired", "rejected"):
        print(f"STOP: exchange order still pending (status={status})")
        return False
    filled = float(order.get("filled") or 0)
    average = float(order.get("average") or pending["reference_price"])
    if filled > 0:
        if pending["side"] == "BUY":
            old = state["units"]
            state["average_entry"] = (old * state["average_entry"] + filled * average) / (old + filled)
            state["units"] += filled
        else:
            state["units"] = max(0.0, state["units"] - filled)
            if state["units"] < 1e-10:
                state["average_entry"] = 0.0
        log_order(state, pending["side"], filled, average, None, "live", order_id)
    state["pending"] = None
    save_state(state)
    print(f"Reconciled live order {order_id}: {status}, filled={filled:.8f}")
    return True


def live_step(state, signal, px, exchange):
    """Buy from free quote balance; sell ONLY coins bought/tracked by bot."""
    if not reconcile_pending(state, exchange):
        return
    # The strategy uses the last completed close, but LIVE sizing uses a fresh
    # spot quote. A market order may still fill at a different price.
    ticker = exchange.fetch_ticker(PAIR)
    live_px = float(ticker.get("last") or 0)
    if live_px <= 0:
        raise RuntimeError("No valid live market price; will not place an order")
    px = live_px
    balance = exchange.fetch_balance()
    base, quote = PAIR.split("/")
    free_quote = float(balance.get("free", {}).get(quote) or 0)
    free_base = float(balance.get("free", {}).get(base) or 0)
    market = exchange.market(PAIR)
    if signal == "BUY":
        remaining_exposure = max(0.0, MAX_EXPOSURE - state["units"] * px)
        spend = min(MAX_ORDER, remaining_exposure, free_quote * 0.98)
        if spend <= 0:
            print("LIVE HOLD: no free quote currency or exposure limit reached")
            return
        raw_amount = spend / px
        side = "BUY"
    elif signal == "SELL" and state["units"] > 0:
        raw_amount = min(state["units"], free_base)
        side = "SELL"
    else:
        print("LIVE HOLD: no tradable signal or bot-tracked position")
        return

    amount = float(exchange.amount_to_precision(PAIR, raw_amount))
    min_units = float((market.get("limits", {}).get("amount", {}) or {}).get("min") or 0)
    min_cost = float((market.get("limits", {}).get("cost", {}) or {}).get("min") or 0)
    if amount <= 0 or amount < min_units or amount * px < min_cost:
        print("LIVE HOLD: exchange minimum/precision not met")
        return

    # Persist guard BEFORE making the request: avoids a blind retry on timeout.
    state["pending"] = {"id": None, "side": side, "amount": amount,
                        "reference_price": px, "utc": datetime.now(timezone.utc).isoformat()}
    save_state(state)
    print(f"Submitting LIVE {side} market order for {amount:.8f} {base} ({PAIR})")
    try:
        order = exchange.create_order(PAIR, "market", side.lower(), amount)
    except Exception as exc:
        print(f"UNCERTAIN ORDER OUTCOME: {exc}. STOP; check Kraken before clearing pending.")
        return
    state["pending"]["id"] = order.get("id")
    save_state(state)
    if not state["pending"]["id"]:
        print("Order returned without id; inspect exchange before attempting another order.")
        return
    reconcile_pending(state, exchange)


def run_once():
    if PAIR not in SUPPORTED_PAIRS:
        raise RuntimeError(f"PAIR must be one of {SUPPORTED_PAIRS}")
    if MAX_ORDER <= 0 or MAX_EXPOSURE <= 0:
        raise RuntimeError("Order and exposure limits must be positive")
    if REAL_MODE:
        exchange = private_exchange()  # validate authentication even if signal is HOLD
    else:
        exchange = None
    state = read_state()
    if REAL_MODE and state.get("pending"):
        if not reconcile_pending(state, exchange):
            return
    prices = load_prices(PAIR)
    last_day = prices.index[-1].strftime("%Y-%m-%d")
    if state["last_checked_day"] == last_day:
        print(f"Already evaluated closed UTC candle for {last_day}. No repeat trades.")
        return
    outlook = create_outlook(prices, PAIR)
    signal = outlook.signal
    px = outlook.last_close
    # Software stop-loss is checked once each day, not a guaranteed protective order.
    if state["units"] > 0 and state["average_entry"] > 0 and px <= 0.95 * state["average_entry"]:
        signal = "SELL"
        print("5% daily-close software stop-loss condition met")
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC | {PAIR} "
          f"closed {last_day} at {px:,.2f} | Signal: {signal}")
    # Before live request: mark day checked, even if outcome becomes ambiguous.
    state["last_checked_day"] = last_day
    save_state(state)
    if REAL_MODE:
        live_step(state, signal, px, exchange)
    else:
        paper_step(state, signal, px)
        save_state(state)


def main():
    parser = argparse.ArgumentParser(description="Spot-only crypto AI bot")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one evaluation and exit (default)")
    mode.add_argument("--loop", action="store_true", help="Keep running on a persistent machine; checks hourly")
    args = parser.parse_args()
    if args.loop:
        print("Looping hourly; only one evaluation per completed UTC daily candle.")
        while True:
            try:
                run_once()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"Bot error: {exc}")
            time.sleep(3600)
    else:
        run_once()

if __name__ == "__main__":
    main()
