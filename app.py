import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ---------------------------
# Data loading and utilities
# ---------------------------

def get_price_data(ticker, start_date, end_date):
    """
    Download price data from Yahoo Finance.
    """
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,  # keep raw prices (no dividend / split adjust)
        )
    except Exception:
        # if yfinance fails (bad ticker, network issue, ...)
        return pd.DataFrame()
    return data


def to_series(obj):
    """
    Make sure we are working with a pandas Series.
    If it's a DataFrame, take the first column.
    If it's a scalar, build a 1-element Series.
    """
    if isinstance(obj, pd.DataFrame):
        # take the first column
        return obj.iloc[:, 0]
    if isinstance(obj, pd.Series):
        return obj
    # fallback: scalar -> series of length 1
    return pd.Series([obj])


# ---------------------------
# Strategies
# ---------------------------

def backtest_buy_and_hold(close):
    """
    Very simple buy & hold strategy:
    - invest 100% at the beginning, hold until the end
    Returns:
      equity: normalized portfolio value
      rets: period returns
    """
    close = to_series(close).sort_index()
    equity = close / close.iloc[0]
    rets = equity.pct_change().dropna()
    return equity, rets


def backtest_ma_crossover(close, short_window, long_window):
    """
    Moving average crossover strategy:
    - long when short MA > long MA
    - flat otherwise
    """
    close = to_series(close).sort_index()
    df = pd.DataFrame({"close": close})
    df["ma_short"] = df["close"].rolling(short_window).mean()
    df["ma_long"] = df["close"].rolling(long_window).mean()

    # drop first rows where MAs are not defined
    df = df.dropna()
    if len(df) < 2:
        # not enough points for a real backtest
        eq = pd.Series([1.0], index=df.index[:1])
        rets = eq.pct_change().dropna()
        return eq, rets

    # 1 when short MA > long MA, else 0
    df["position"] = (df["ma_short"] > df["ma_long"]).astype(int)

    # returns of the asset
    asset_rets = df["close"].pct_change().fillna(0.0)

    # strategy returns: position of previous period * asset return
    strat_rets = asset_rets * df["position"].shift(1).fillna(0.0)

    equity = (1.0 + strat_rets).cumprod()
    rets = equity.pct_change().dropna()
    return equity, rets


# ---------------------------
# Performance metrics
# ---------------------------

def compute_metrics(equity, rets):
    """
    Compute basic performance metrics:
    - total return
    - annualized return (using 252 periods per year)
    - annualized volatility
    - Sharpe ratio (rf = 0)
    - max drawdown
    """
    eq = to_series(equity)
    r = to_series(rets)

    total_ret = eq.iloc[-1] - 1.0

    if len(eq.index) > 1:
        nb_days = (eq.index[-1] - eq.index[0]).days
    else:
        nb_days = 0

    if nb_days <= 0 or r.empty:
        ann_ret = np.nan
        ann_vol = np.nan
        sharpe = np.nan
    else:
        mean_per_period = r.mean()
        # we assume 252 trading days even if we resample weekly/monthly
        ann_ret = (1.0 + mean_per_period) ** 252 - 1.0
        vol_per_period = r.std()
        ann_vol = vol_per_period * np.sqrt(252)
        if ann_vol > 0:
            sharpe = ann_ret / ann_vol
        else:
            sharpe = np.nan

    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    max_dd = drawdown.min()

    return {
        "total_return": total_ret,
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


# ---------------------------
# Simple prediction model
# ---------------------------

def forecast_linear_trend(close, horizon, freq_label):
    """
    Very simple prediction model:
    - linear regression of price vs time index
    - extrapolate on future periods
    - build a (rough) 95% confidence band using residual std
    """
    close = to_series(close).dropna().sort_index()
    n = len(close)
    if n < 5:
        raise ValueError("Not enough data to fit a model (need at least 5 points).")

    # time axis: 0, 1, ..., n-1
    t = np.arange(n)
    # matrix for least squares: [t, 1]
    A = np.vstack([t, np.ones(n)]).T

    # least squares fit: price ≈ m * t + c
    m, c = np.linalg.lstsq(A, close.values, rcond=None)[0]

    # in-sample predictions and residuals
    pred_in = m * t + c
    residuals = close.values - pred_in
    sigma = residuals.std(ddof=1)

    # future time points
    t_future = np.arange(n, n + horizon)
    pred_future = m * t_future + c

    # map frequency label to pandas frequency string
    if freq_label == "Daily":
        freq = "D"
    elif freq_label == "Weekly":
        freq = "W"
    else:
        freq = "M"

    # build future dates
    last_date = close.index[-1]
    offset = pd.tseries.frequencies.to_offset(freq)
    start_future = last_date + offset
    future_index = pd.date_range(start=start_future, periods=horizon, freq=freq)

    hist = close.rename("Historical price")
    forecast = pd.Series(pred_future, index=future_index, name="Forecast")
    lower = (forecast - 1.96 * sigma).rename("Lower CI (95%)")
    upper = (forecast + 1.96 * sigma).rename("Upper CI (95%)")

    # combine in one DataFrame (historical + forecast + bands)
    df_hist = pd.DataFrame({"Historical price": hist})
    df_forecast = pd.DataFrame(
        {
            "Forecast": forecast,
            "Lower CI (95%)": lower,
            "Upper CI (95%)": upper,
        }
    )
    combined = pd.concat([df_hist, df_forecast], axis=1)
    return combined