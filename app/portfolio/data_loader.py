#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:41:06 2025

@author: nieto
"""

import yfinance as yf
import pandas as pd

DEFAULT_TICKERS = [
    "AAPL", "SPY", "ACA.PA", "AIR.PA",
    "BZ=F", "^TNX", "GC=F", "BTC-USD"
]

def load_multi_asset_data(tickers=DEFAULT_TICKERS, start="2015-01-01", end=None):

    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False
    )

    # yfinance returns a MultiIndex for multiple tickers:
    # ('Adj Close', 'AAPL'), ('Adj Close', 'SPY') ...
    if isinstance(data.columns, pd.MultiIndex):
        # Keep only Adj Close level
        if "Adj Close" in data.columns.levels[0]:
            data = data["Adj Close"]
        else:
            # fallback : take Close
            data = data["Close"]

        # Flatten columns
        data.columns = [col for col in data.columns]

    # Ensure DataFrame for single asset
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Clean missing rows (keep full rows only)
    data = data.dropna(how="any")

    return data
