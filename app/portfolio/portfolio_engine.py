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

# -----------------------------
# Cumulative portfolio performance
# -----------------------------
def cumulative_returns(port_ret: pd.Series) -> pd.Series:
    """Convert daily returns into cumulative performance."""
    return (1 + port_ret).cumprod()


# -----------------------------
# Optional rebalancing (simple)
# -----------------------------
def rebalance_portfolio(returns: pd.DataFrame,
                        weights: list[float],
                        freq: str = "M") -> pd.Series:
    """
    Rebalance portfolio on a fixed frequency (e.g., monthly).
    Recomputes portfolio weights at each rebalancing interval.
    """
    w = np.array(weights) / np.sum(weights)
    
    # group returns by resampling period
    grouped = returns.resample(freq)
    
    port_values = []
    prev_value = 1.0
    
    for _, group in grouped:
        # apply fixed weights inside each rebalancing window
        period_ret = group.mul(w).sum(axis=1)
        cum_period = (1 + period_ret).cumprod() * prev_value
        prev_value = cum_period.iloc[-1]
        port_values.append(cum_period)

    return pd.concat(port_values)