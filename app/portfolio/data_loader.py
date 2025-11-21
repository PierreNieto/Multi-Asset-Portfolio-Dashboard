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

    # Rates / sovereign yields (à ajuster selon ce que supporte Yahoo)
    "^TNX",         # US 10Y
    "FR10Y=RR",     # France 10Y (si dispo)
    "IT10Y=RR",     # Italy 10Y
    "GR10Y=RR",     # Greece 10Y
    "BR10Y=RR",     # Brazil 10Y
]



def load_multi_asset_data(tickers, start="2010-01-01"):
    valid_data = {}
    invalid = []

    for t in tickers:
        try:
            df = yf.download(t, start=start)["Adj Close"]
            if df.empty:
                invalid.append(t)
            else:
                valid_data[t] = df
        except Exception:
            invalid.append(t)

    if not valid_data:
        return pd.DataFrame()

    prices = pd.concat(valid_data.values(), axis=1)
    prices.columns = valid_data.keys()

    if invalid:
        st.warning(f"⚠️ Invalid or unavailable tickers ignored: {invalid}")

    return prices

