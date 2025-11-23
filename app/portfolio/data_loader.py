#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:41:06 2025

@author: nieto
"""

import yfinance as yf
import pandas as pd

DEFAULT_TICKERS = [
    # Core equity indices / ETFs / US megacaps
    "SPY",          # S&P 500 ETF
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "BRK-B",
    "JPM",
    "V",

    # Non-US large caps
    "TSM",
    "NVO",
    "LLY",
    "ASML.AS",
    "ADYEN.AS",
    "SAN.PA",

    # CAC40 / Euronext leaders
    "ACA.PA",
    "AIR.PA",
    "MC.PA",
    "OR.PA",
    "TTE.PA",

    # China / HK large caps
    "0700.HK",      # Tencent
    "9988.HK",      # Alibaba
    "600519.SS",    # Kweichow Moutai

    # Commodities & crypto
    "BZ=F",         # Brent
    "GC=F",         # Gold
    "BTC-USD",
    "ETH-USD",
]


def load_multi_asset_data(tickers=DEFAULT_TICKERS, start="2015-01-01", end=None):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        interval="1d",
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
