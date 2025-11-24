#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yfinance as yf
import pandas as pd
import numpy as np

DEFAULT_TICKERS = [
    # US
    "SPY", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "BRK-B", "JPM", "V",

    # Non-US
    "TSM", "NVO", "LLY", "ASML.AS", "ADYEN.AS", "SAN.PA",

    # CAC40 / Euronext
    "ACA.PA", "AIR.PA", "MC.PA", "OR.PA", "TTE.PA",

    # China
    "0700.HK", "9988.HK", "600519.SS",

    # Commodities & crypto
    "BZ=F", "GC=F", "BTC-USD", "ETH-USD",
]


def load_multi_asset_data(tickers=DEFAULT_TICKERS, start="2015-01-01", end=None, interval="1d"):
    """Robust multi-asset loader that avoids MultiIndex issues, keeps all assets,
    and prevents data loss due to missing rows."""

    # ----download ----
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as e:
        raise RuntimeError(f"YFinance download failed: {e}")

    # ----extract Close or Adj Close ----
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.levels[0]:
            data = raw["Adj Close"].copy()
        else:
            data = raw["Close"].copy()
    else:
        # Single ticker case
        data = raw.copy()
        if "Adj Close" in data.columns:
            data = data["Adj Close"].to_frame()
        elif "Close" in data.columns:
            data = data["Close"].to_frame()

    # ---- ensure DataFrame ----
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # ---- Remove columns that are fully empty ----
    data = data.dropna(axis=1, how="all")

    # ---- ffill to preserve all dates ----
    data = data.sort_index().ffill()

    # ---- bfill first row only if needed ----
    data = data.bfill(limit=1)

    return data
