#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:41:06 2025

@author: nieto
"""

import yfinance as yf
import pandas as pd
import time

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

# -------------------------------------------------------------------
# Helper function: download with auto-retry
# -------------------------------------------------------------------
def _safe_download(ticker, **kwargs):
    """Download ticker with retry logic to avoid API failures."""
    for attempt in range(3):
        try:
            df = yf.download(ticker, progress=False, auto_adjust=False, **kwargs)
            if not df.empty:
                return df
        except Exception:
            pass
        time.sleep(1.0)
    return pd.DataFrame()


# -------------------------------------------------------------------
# Main loader
# -------------------------------------------------------------------
def load_multi_asset_data(
    tickers=DEFAULT_TICKERS,
    start="2015-01-01",
    end=None,
):
    """
    Load multiple assets with mixed frequencies:
      - Crypto (BTC & ETH) -> 5m interval (max 60d range)
      - Other assets -> 1d interval since 2015
    Then resample everything to daily and align cleanly.
    """

    all_series = {}

    for ticker in tickers:

        # ------------------------------------------
        # 1) Choose interval based on asset type
        # ------------------------------------------
        if ticker in ["BTC-USD", "ETH-USD"]:
            df = _safe_download(
                ticker,
                period="60d",
                interval="5m",
            )
        else:
            df = _safe_download(
                ticker,
                start=start,
                end=end,
                interval="1d",
            )

        if df.empty:
            print(f"[WARN] Could not download {ticker}")
            continue

        # ------------------------------------------
        # 2) Extract Adj Close or fallback to Close
        # ------------------------------------------
        if isinstance(df.columns, pd.MultiIndex):
            if "Adj Close" in df.columns.levels[0]:
                s = df["Adj Close"]
            else:
                s = df["Close"]

            if isinstance(s, pd.DataFrame):  # ex: df["Adj Close"] returns DataFrame
                col = s.columns[0]
                series = s[col]
            else:
                series = s

        else:
            if "Adj Close" in df.columns:
                series = df["Adj Close"]
            else:
                series = df["Close"]

        # ------------------------------------------
        # 3) Resample to daily
        # ------------------------------------------
        series = (
            series
            .resample("1D")
            .last()
            .ffill()
        )

        # Force ticker name
        all_series[ticker] = series.rename(ticker)

    # -------------------------------------------------------------------
    # 4) Concatenate and final clean
    # -------------------------------------------------------------------
    if not all_series:
        raise ValueError("ERROR: No assets could be loaded.")

    final_df = pd.concat(all_series.values(), axis=1)
    final_df = final_df.dropna(how="all")

    return final_df