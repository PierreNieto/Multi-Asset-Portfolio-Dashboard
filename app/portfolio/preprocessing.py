#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# Simple returns
def compute_simple_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple percentage returns."""
    return price_df.pct_change().dropna()


# Log returns
def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns: log(Pt / Pt-1)."""
    return np.log(price_df / price_df.shift(1)).dropna()


# Resampling utilities (weekly, monthly…)
def resample_price_data(price_df: pd.DataFrame,
                        freq: str = "M",
                        how: str = "last") -> pd.DataFrame:
    """Resample price data (e.g., daily → monthly)."""
    if how == "last":
        resampled = price_df.resample(freq).last()
    elif how == "first":
        resampled = price_df.resample(freq).first()
    elif how == "mean":
        resampled = price_df.resample(freq).mean()
    elif how == "max":
        resampled = price_df.resample(freq).max()
    elif how == "min":
        resampled = price_df.resample(freq).min()
    else:
        raise ValueError("Unsupported aggregation method.")
    return resampled.dropna()

# Rolling volatility
def rolling_volatility(returns: pd.DataFrame,
                       window: int = 20) -> pd.DataFrame:
    """Compute rolling volatility over a given window."""
    return returns.rolling(window).std()


# Rolling mean (moving average)
def rolling_mean(price_df: pd.DataFrame,
                 window: int = 20) -> pd.DataFrame:
    """Compute rolling mean of price series."""
    return price_df.rolling(window).mean()


# Rolling correlation between two assets
def rolling_correlation(series1: pd.Series,
                        series2: pd.Series,
                        window: int = 20) -> pd.Series:
    """Compute rolling correlation between two price/return series."""
    return series1.rolling(window).corr(series2)
