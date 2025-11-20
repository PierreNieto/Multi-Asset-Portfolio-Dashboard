#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def compute_simple_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    returns = price_df.pct_change().dropna()
    return returns
