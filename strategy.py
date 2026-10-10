"""Market data, walk-forward-style holdout checks, and 1-7-day forecasts.

All public market information comes from Kraken via ccxt. No API key is needed
for analysis. Predictions are statistical guesses, not trading guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

SUPPORTED_PAIRS = ("BTC/AUD", "ETH/AUD", "BTC/USD", "ETH/USD", "SOL/USD")
HORIZONS = range(1, 8)


def public_exchange():
    return ccxt.kraken({"enableRateLimit": True, "timeout": 20000})


def load_prices(pair: str = "BTC/AUD") -> pd.DataFrame:
    """Load up to 720 daily candles; remove today's unfinished UTC candle."""
    if pair not in SUPPORTED_PAIRS:
        raise ValueError(f"Unsupported pair: {pair}")
    exchange = public_exchange()
    exchange.load_markets()
    if pair not in exchange.markets:
        raise RuntimeError(f"{pair} is currently unavailable on Kraken")
    candles = exchange.fetch_ohlcv(pair, timeframe="1d", limit=720)
    if not candles:
        raise RuntimeError("Kraken returned no OHLCV data")
    prices = pd.DataFrame(
        candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    prices["date"] = pd.to_datetime(prices.pop("timestamp"), unit="ms", utc=True)
    prices = prices.set_index("date").sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    now = pd.Timestamp.now(tz="UTC")
    prices = prices[prices.index < now.normalize()]
    if len(prices) < 250:
        raise RuntimeError(f"Not enough completed daily candles ({len(prices)}): need 250+")
    return prices.astype(float)


def indicators(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["close"].astype(float)
    volume = prices["volume"].astype(float)
    change = close.pct_change()
    feat = pd.DataFrame(index=prices.index)

    for lag in (1, 2, 3, 5, 7, 14):
        feat[f"return_{lag}d"] = close.pct_change(lag)
    for window in (5, 10, 20, 50):
        feat[f"sma_gap_{window}"] = close / close.rolling(window).mean() - 1.0
    for window in (7, 14, 30):
        feat[f"volatility_{window}"] = change.rolling(window).std()
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    feat["rsi_14"] = 100 - (100 / (1 + up / down.replace(0, np.nan)))
    feat["volume_relative_14"] = volume / volume.rolling(14).mean().replace(0, np.nan) - 1.0
    feat["intraday_range"] = (prices["high"] - prices["low"]) / close
    feat["open_close_gap"] = (close - prices["open"]) / prices["open"].replace(0, np.nan)
    return feat.replace([np.inf, -np.inf], np.nan)


@dataclass
class Outlook:
    pair: str
    last_date: pd.Timestamp
    last_close: float
    forecast: pd.DataFrame
    validation: pd.DataFrame
    signal: str
    signal_reason: str
    sma_20: float


def _model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=85,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )


def create_outlook(prices: pd.DataFrame, pair: str = "BTC/AUD") -> Outlook:
    """Predict each horizon directly; evaluate on a separate recent holdout.

    The training set is purged by each horizon to avoid using labels that extend
    into the holdout. The holdout is NOT used to tune hyperparameters.
    """
    feats = indicators(prices)
    today_features = feats.dropna().iloc[[-1]]
    if today_features.index[-1] != prices.index[-1]:
        raise RuntimeError("Latest features unavailable")

    out, metrics = [], []
    for horizon in HORIZONS:
        target = prices["close"].shift(-horizon) / prices["close"] - 1.0
        labeled = feats.join(target.rename("target")).dropna()
        if len(labeled) < 180:
            raise RuntimeError("Not enough historical samples to train model")
        validation_days = min(90, max(40, len(labeled) // 6))
        test_start = len(labeled) - validation_days
        train = labeled.iloc[: test_start - horizon]  # purge future labels
        test = labeled.iloc[test_start:]
        cols = list(feats.columns)
        first_model = _model().fit(train[cols], train["target"])
        observed = test["target"].to_numpy()
        predicted = first_model.predict(test[cols])
        model_mae = float(mean_absolute_error(observed, predicted))
        naive_mae = float(mean_absolute_error(observed, np.zeros(len(observed))))
        residuals = observed - predicted

        # Refit on all data with known labels before forecasting the future.
        production_model = _model().fit(labeled[cols], labeled["target"])
        expected_return = float(production_model.predict(today_features[cols])[0])
        low_r = expected_return + float(np.quantile(residuals, 0.10))
        high_r = expected_return + float(np.quantile(residuals, 0.90))
        last_close = float(prices["close"].iloc[-1])
        out.append({
            "date": prices.index[-1] + pd.Timedelta(days=horizon),
            "horizon": horizon,
            "forecast": last_close * (1 + expected_return),
            "low": max(0, last_close * (1 + low_r)),
            "high": max(0, last_close * (1 + high_r)),
            "expected_return_pct": 100 * expected_return,
        })
        metrics.append({
            "horizon": horizon,
            "model_mae_pct": 100 * model_mae,
            "naive_mae_pct": 100 * naive_mae,
            "model_beats_naive": model_mae < naive_mae,
        })

    forecast = pd.DataFrame(out).set_index("date")
    validation = pd.DataFrame(metrics).set_index("horizon")
    close = float(prices["close"].iloc[-1])
    sma20 = float(prices["close"].tail(20).mean())
    day1_return = float(forecast.iloc[0]["expected_return_pct"])
    # Simple, non-optimised illustration of a conservative spot strategy.
    # SELL means exit only coins acquired by THIS bot; never a short sale.
    if day1_return >= 1.0 and close > sma20 and bool(validation.loc[1, "model_beats_naive"]):
        signal = "BUY"
        reason = "1-day forecast is above +1%, price above 20-day SMA, and model beat holdout baseline"
    elif day1_return <= -1.0 and close < sma20 and bool(validation.loc[1, "model_beats_naive"]):
        signal = "SELL"
        reason = "1-day forecast is below -1%, price below 20-day SMA, and model beat holdout baseline"
    else:
        signal = "HOLD"
        reason = "Signal filters not all met; no new position recommended by the rules"
    return Outlook(pair, prices.index[-1], close, forecast, validation, signal, reason, sma20)
