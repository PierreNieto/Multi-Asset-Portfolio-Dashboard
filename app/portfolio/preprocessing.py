#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 21:16:02 2025

@author: nieto
"""

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




# — Basic cleaning utilities
def clean_price_data(price_df: pd.DataFrame) -> pd.DataFrame:
    """Sort index, drop duplicates and rows with all-NaN values."""
    df = price_df.copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(how="all")


def fill_missing_values(df: pd.DataFrame,
                        method: str = "ffill") -> pd.DataFrame:
    """Fill missing values (default: forward fill, then backfill)."""
    if method == "ffill":
        df = df.ffill().bfill()
    elif method == "bfill":
        df = df.bfill().ffill()
    else:
        raise ValueError("Unsupported fill method.")
    return df


def align_dataframes(dfs: list[pd.DataFrame],
                     how: str = "inner") -> list[pd.DataFrame]:
    """
    Align multiple DataFrames on the same date index.

    Example:
        prices_aligned, macro_aligned = align_dataframes([prices, macro])
    """
    if not dfs:
        return dfs
    common_index = dfs[0].index
    for df in dfs[1:]:
        if how == "inner":
            common_index = common_index.intersection(df.index)
        elif how == "outer":
            common_index = common_index.union(df.index)
        else:
            raise ValueError("Unsupported align method.")
    aligned = [df.reindex(common_index) for df in dfs]
    return aligned
