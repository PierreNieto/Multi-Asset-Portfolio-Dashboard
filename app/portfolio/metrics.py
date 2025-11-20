#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


# -----------------------------
# Correlation matrix
# -----------------------------
def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix between assets."""
    return returns.corr()


# -----------------------------
# Annualized metrics
# -----------------------------
def annualized_return(returns: pd.DataFrame | pd.Series,
                      freq: int = 252) -> float | pd.Series:
    """Compute annualized return."""
    if isinstance(returns, pd.DataFrame):
        return (1 + returns.mean() * freq) - 1
    return (1 + returns.mean() * freq) - 1


def annualized_volatility(returns: pd.DataFrame | pd.Series,
                          freq: int = 252) -> float | pd.Series:
    """Compute annualized volatility."""
    if isinstance(returns, pd.DataFrame):
        return returns.std() * np.sqrt(freq)
    return returns.std() * np.sqrt(freq)


# -----------------------------
# Sharpe ratio
# -----------------------------
def sharpe_ratio(returns: pd.DataFrame | pd.Series,
                 risk_free_rate: float = 0.0,
                 freq: int = 252) -> float | pd.Series:
    """Compute Sharpe ratio."""
    excess_ret = returns - risk_free_rate / freq
    ann_ret = annualized_return(excess_ret, freq)
    ann_vol = annualized_volatility(excess_ret, freq)
    return ann_ret / ann_vol


# -----------------------------
# Diversification ratio
# -----------------------------
def diversification_ratio(returns: pd.DataFrame,
                          weights: list[float]) -> float:
    """
    Diversification ratio:
    DR = w^T σ / sqrt(w^T Σ w)
    Higher → better diversification.
    """
    w = np.array(weights) / np.sum(weights)
    stds = returns.std()
    cov = returns.cov()
    portfolio_vol = np.sqrt(w.T @ cov.values @ w)
    weighted_stds = (w * stds).sum()
    return weighted_stds / portfolio_vol
