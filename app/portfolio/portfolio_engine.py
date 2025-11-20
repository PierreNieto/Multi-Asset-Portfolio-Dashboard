#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


# -----------------------------
# Equal-weight portfolio
# -----------------------------
def equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    """Compute equal-weight portfolio returns."""
    n = returns.shape[1]
    weights = np.ones(n) / n
    port_ret = returns.mul(weights).sum(axis=1)
    return port_ret


# -----------------------------
# Custom-weight portfolio
# -----------------------------
def custom_weight_portfolio(returns: pd.DataFrame,
                            weights: list[float]) -> pd.Series:
    """Compute portfolio returns using user-defined weights."""
    w = np.array(weights)
    w = w / w.sum()  # normalize weights
    port_ret = returns.mul(w).sum(axis=1)
    return port_ret

