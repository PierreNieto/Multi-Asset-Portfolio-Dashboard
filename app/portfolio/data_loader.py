#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:41:06 2025

@author: nieto
"""

import yfinance as yf
import pandas as pd

DEFAULT_TICKERS = [
    "AAPL",        # Apple - US Tech
    "SPY",         # S&P 500 ETF
    "ACA.PA",      # Crédit Agricole - European Finance
    "AIR.PA",      # Airbus - Europe
    "BZ=F",        # Brent Crude Oil
    "^TNX",        # US 10Y Bond Yield
    "GC=F",        # Gold
    "BTC-USD"      # Bitcoin
]

def load_multi_asset_data(tickers=DEFAULT_TICKERS, start="2015-01-01", end=None):
    """
    Download adjusted close prices for a diversified basket of assets.
    """

    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False
    )

    # ------------------------------------------------------------
    # Why this fallback block is added:
    #
    # Most Yahoo Finance tickers provide an "Adj Close" column,
    # which is the correct price to use for financial analysis
    # (adjusted for dividends and stock splits).
    #
    # However, some tickers do NOT provide "Adj Close":
    # - certain interest rate tickers (e.g., ^TNX)
    # - some indices
    # - some commodities
    #
    # Without this verification, the code would crash.
    #
    # Therefore:
    # - If "Adj Close" exists → we use it (standard case)
    # - If not → we use the first available column as a fallback
    #
    # This makes the data loader more robust and prevents unexpected errors.
    # ------------------------------------------------------------
    if "Adj Close" in data.columns:
        data = data["Adj Close"]
    else:
        data = data.iloc[:, 0]  # fallback to first column

    # ensure DataFrame always
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Remove any date where one of the assets has missing data
    return data.dropna()
