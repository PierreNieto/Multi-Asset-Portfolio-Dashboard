#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# =====================================================
# BASIC PLOTS
# =====================================================

# -----------------------------
# Multi-asset price chart
# -----------------------------
def plot_price_series(price_df: pd.DataFrame) -> go.Figure:
    """Plot multi-asset price series."""
    fig = px.line(price_df, title="Asset Price Series")
    fig.update_layout(xaxis_title="Date", yaxis_title="Price")
    return fig


# -----------------------------
# Portfolio cumulative returns
# -----------------------------
def plot_cumulative_returns(cum_series: pd.Series) -> go.Figure:
    """Plot cumulative portfolio performance."""
    fig = px.line(cum_series, title="Portfolio Cumulative Returns")
    fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative Value")
    return fig


# -----------------------------
# Correlation heatmap
# -----------------------------
def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Plot correlation matrix as heatmap."""
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix"
    )
    return fig


# -----------------------------
# Rolling volatility plot 
# -----------------------------
def plot_rolling_volatility(rolling_vol: pd.DataFrame) -> go.Figure:
    """Plot rolling volatility for each asset."""
    fig = px.line(rolling_vol, title="Rolling Volatility")
    fig.update_layout(xaxis_title="Date", yaxis_title="Volatility")
    return fig


# =====================================================
# ADVANCED PRO PLOTS (Mode Pro)
# =====================================================

# -----------------------------
# Rolling Beta
# -----------------------------
def plot_rolling_beta(beta_series: pd.Series, benchmark: str) -> go.Figure:
    """Plot rolling beta vs benchmark."""
    fig = px.line(
        beta_series,
        title=f"Rolling Beta vs {benchmark}"
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Beta",
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black"),
    )
    return fig


# -----------------------------
# Drawdown (Pro version)
# -----------------------------
def plot_drawdown(drawdown: pd.Series) -> go.Figure:
    """Plot portfolio drawdown as an area chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        fill='tozeroy',
        mode='lines',
        name='Drawdown',
        line=dict(color='red')
    ))
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown",
    )
    return fig


# -----------------------------
# Efficient Frontier
# -----------------------------
def plot_efficient_frontier(ef_results: pd.DataFrame,
                            curr_vol: float,
                            curr_ret: float) -> go.Figure:
    """Plot Efficient Frontier and current portfolio point."""
    fig = px.scatter(
        ef_results,
        x="Volatility",
        y="Return",
        color="Sharpe",
        color_continuous_scale="Viridis",
        title="Efficient Frontier — Random Portfolios",
        height=600,
    )

    # Add your portfolio point
    fig.add_scatter(
        x=[curr_vol],
        y=[curr_ret],
        mode="markers",
        marker=dict(color="red", size=14, line=dict(color="black", width=1)),
        name="Your Portfolio",
    )

    return fig


# -----------------------------
# Rolling Correlation
# -----------------------------
def plot_rolling_correlation(returns: pd.DataFrame,
                             asset1: str,
                             asset2: str,
                             window: int = 60) -> go.Figure:
    """Plot rolling correlation between two assets."""
    rolling_corr = returns[asset1].rolling(window).corr(returns[asset2])

    fig = px.line(
        rolling_corr,
        title=f"Rolling Correlation: {asset1} vs {asset2}"
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1, 1]),
    )

    return fig
