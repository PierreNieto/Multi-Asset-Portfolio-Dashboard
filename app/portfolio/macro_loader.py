#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Macro Loader Module
--------------------
Loads macroeconomic indicators from FRED using pandas_datareader.

Includes:
- US CPI inflation
- US Real Yield (10Y)
- Sovereign 10Y Yields (US, France, Germany, Italy)
- GDP (US, France)
"""

import pandas as pd
import pandas_datareader.data as web


# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------
def _load_fred_series(series_code, col_name, start="2015-01-01"):
    """
    Generic loader for a single FRED time series.
    """
    try:
        df = web.DataReader(series_code, "fred", start)
        df = df.rename(columns={series_code: col_name})
        return df.dropna()
    except Exception as e:
        print(f"Error loading {col_name} ({series_code}) from FRED:", e)
        return pd.DataFrame()


# ---------------------------------------------------------
# CPI & Real Yield
# ---------------------------------------------------------
def load_cpi(start="2015-01-01"):
    return _load_fred_series("CPIAUCSL", "US_CPI", start)


def load_real_yield(start="2015-01-01"):
    return _load_fred_series("DFII10", "US_10Y_Real", start)


# ---------------------------------------------------------
# Sovereign Government Bond Yields (10-Year)
# ---------------------------------------------------------
def load_sovereign_yields(start="2015-01-01"):
    """
    10-Year Government Bond Yields for:
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

    dfs = []
    for col_name, fred_code in bonds.items():
        df = _load_fred_series(fred_code, col_name, start)
        if not df.empty:
            dfs.append(df)

    if dfs:
        # Merge on the index (date)
        return pd.concat(dfs, axis=1).dropna(how="all")

    return pd.DataFrame()


# ---------------------------------------------------------
# GDP (Quarterly)
# ---------------------------------------------------------
def load_gdp(start="2000-01-01"):
    """
    Load GDP for:
    - United States
    - France
    """
    gdp_series = {
        "US_GDP": "GDP",
        "FR_GDP": "CLVMNACSCAB1GQFRA",
    }

    dfs = []
    for col_name, fred_code in gdp_series.items():
        df = _load_fred_series(fred_code, col_name, start)
        if not df.empty:
            dfs.append(df)

    if dfs:
        return pd.concat(dfs, axis=1).dropna(how="all")

    return pd.DataFrame()


# ---------------------------------------------------------
# Global macro loader
# ---------------------------------------------------------
def load_macro_data(start="2015-01-01"):
    """
    Returns a dictionary:

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
# Debug Test
# ---------------------------------------------------------
if __name__ == "__main__":
    macro = load_macro_data()
    for k, v in macro.items():
        print("\n==========", k, "==========")
        print(v.head())
