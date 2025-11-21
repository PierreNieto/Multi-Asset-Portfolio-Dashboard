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
def annualized_return(returns: pd.Series | pd.DataFrame, freq=252):
    if isinstance(returns, pd.DataFrame):
        return (1 + returns).prod() ** (freq / len(returns)) - 1
    else:
        return (1 + returns).prod() ** (freq / len(returns)) - 1



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

def rolling_beta(portfolio_returns, benchmark_returns, window=60):
    cov = portfolio_returns.rolling(window).cov(benchmark_returns)
    var = benchmark_returns.rolling(window).var()
    beta = cov / var
    return beta

def compute_var_cvar(returns, level=5):
    var = np.percentile(returns, level)
    cvar = returns[returns < var].mean()
    return var, cvar

def random_portfolios(returns: pd.DataFrame, cov_matrix: pd.DataFrame, n_portfolios: int = 10000):
    """
    Generate random portfolios for Markowitz Efficient Frontier.

    Returns:
        - results: DataFrame with columns [Volatility, Return, Sharpe]
        - weights_list: list of weight vectors for each portfolio
    """

    n_assets = returns.shape[1]
    mean_returns = returns.mean() * 252
    cov_annual = cov_matrix * 252

    results = []
    weights_list = []

    for _ in range(n_portfolios):
        # Random weights
        weights = np.random.random(n_assets)
        weights /= weights.sum()

        # Portfolio performance
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
        port_sharpe = port_return / port_vol if port_vol > 0 else np.nan

        results.append([port_vol, port_return, port_sharpe])
        weights_list.append(weights)

    df_results = pd.DataFrame(results, columns=["Volatility", "Return", "Sharpe"])
    return df_results, weights_list