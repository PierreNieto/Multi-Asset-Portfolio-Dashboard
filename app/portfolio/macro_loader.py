#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 18:08:09 2025

@author: nieto
"""

import pandas as pd
import pandas_datareader.data as web

def load_cpi(start="2015-01-01"):
    """
    Load US CPI (inflation) from FRED (CPIAUCSL).
    Monthly inflation index.
    """
    try:
        df = web.DataReader("CPIAUCSL", "fred", start)
        df = df.rename(columns={"CPIAUCSL": "US_CPI"})
        return df.dropna()
    except Exception as e:
        print("Error loading CPI from FRED:", e)
        return pd.DataFrame()


def load_real_yield(start="2015-01-01"):
    """
    Load US 10-Year Real Interest Rate from FRED (DFII10).
    Useful to analyze risk-free real returns.
    """
    try:
        df = web.DataReader("DFII10", "fred", start)
        df = df.rename(columns={"DFII10": "US_10Y_Real"})
        return df.dropna()
    except Exception as e:
        print("Error loading Real Yield from FRED:", e)
        return pd.DataFrame()


def load_macro_data(start="2015-01-01"):
    """
    Load all macro indicators used in the portfolio module.
    Returns a dictionary of DataFrames.
    """
    macro_data = {
        "US_CPI": load_cpi(start),
        "US_10Y_Real": load_real_yield(start)
    }
    return macro_data

"test"

from app.portfolio.macro_loader import load_macro_data

macro = load_macro_data()
print(macro["US_CPI"].head())
