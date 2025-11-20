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
    "ACA.PA",      # Crédit Agricole - Europe Finance
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
    )["Adj Close"]

    return data.dropna()
