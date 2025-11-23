#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Macro Loader Module (Enhanced)
--------------------------------
Loads macroeconomic indicators from FRED using pandas_datareader.

- Robust retry logic (no crash if FRED is down)
- Clean, consistent interface
- Daily or quarterly macro series
"""

import pandas as pd
import time
import pandas_datareader.data as web


# ---------------------------------------------------------
# Safe loader with retry
# ---------------------------------------------------------
def _safe_fred_load(series_code, col_name, start="2015-01-01"):
    """
    Robust loader with retry logic to avoid FRED outages.
    Returns a DataFrame with one column, or empty DataFrame on failure.
    """
    for attempt in range(3):
        try:
            df = web.DataReader(series_code, "fred", start)
            df = df.rename(columns={series_code: col_name})
            return df.dropna()
        except Exception as e:
            print(f"[WARN] Failed loading {col_name} ({series_code}), attempt {attempt+1}: {e}")
            time.sleep(1.0)

    print(f"[ERROR] Could not load {col_name} after retries.")
    return pd.DataFrame()


# ---------------------------------------------------------
# CPI & Real Yield
# ---------------------------------------------------------
def load_cpi(start="2015-01-01"):
    return _safe_fred_load("CPIAUCSL", "US_CPI", start)


def load_real_yield(start="2015-01-01"):
    return _safe_fred_load("DFII10", "US_10Y_Real", start)


# ---------------------------------------------------------
# Sovereign 10-Year Yields
# ---------------------------------------------------------
def load_sovereign_yields(start="2015-01-01"):
    """
    Loads 10Y yields for:
    - United States
    - France
    - Germany
    - Italy
    """
    bonds = {
        "US_10Y": "DGS10",
        "FR_10Y": "IRLTLT01FRM156N",
        "DE_10Y": "IRLTLT01DEM156N",
        "IT_10Y": "IRLTLT01ITM156N",
    }

    series_list = []
    for col_name, code in bonds.items():
        df = _safe_fred_load(code, col_name, start)
        if not df.empty:
            series_list.append(df)

    return pd.concat(series_list, axis=1).dropna(how="all") if series_list else pd.DataFrame()


# ---------------------------------------------------------
# GDP (Quarterly)
# ---------------------------------------------------------
def load_gdp(start="2000-01-01"):
    """
    Quarterly GDP series for:
    - United States
    - France
    """
    gdp_codes = {
        "US_GDP": "GDP",
        "FR_GDP": "CLVMNACSCAB1GQFRA",
    }

    series_list = []
    for col_name, code in gdp_codes.items():
        df = _safe_fred_load(code, col_name, start)
        if not df.empty:
            series_list.append(df)

    return pd.concat(series_list, axis=1).dropna(how="all") if series_list else pd.DataFrame()


# ---------------------------------------------------------
# Global macro loader
# ---------------------------------------------------------
def load_macro_data(start="2015-01-01"):
    """
    Returns a dict of FRED macro indicators:
    {
        "US_CPI": DataFrame,
        "US_10Y_Real": DataFrame,
        "SOVEREIGN_10Y": DataFrame,
        "GDP": DataFrame
    }
    """
    return {
        "US_CPI": load_cpi(start),
        "US_10Y_Real": load_real_yield(start),
        "SOVEREIGN_10Y": load_sovereign_yields(start),
        "GDP": load_gdp("2000-01-01"),
    }


# ---------------------------------------------------------
# Debug test
# ---------------------------------------------------------
if __name__ == "__main__":
    macro = load_macro_data()
    for k, v in macro.items():
        print("\n==========", k, "==========")
        print(v.head())
