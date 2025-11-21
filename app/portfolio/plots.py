#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

DATE_FORMAT = "%d %b %Y"   # Uniform day + month + year for all charts


# =====================================================
# BASIC PLOTS
# =====================================================

def _format_xaxis(fig):
    """Utility: unify date formatting across all charts."""
    fig.update_xaxes(
        dtick="M1",
        tickformat=DATE_FORMAT
    )
    return fig


# -----------------------------
# Multi-asset price chart
# -----------------------------
def plot_price_series(price_df: pd.DataFrame) -> go.Figure:
    """
    Plot multi-asset price series.

    Y-axis:
        - Price / Level (native units of each asset).
    """
    fig = px.line(price_df, title="Asset Price Series")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price / Level (native units)",
        legend_title="Assets",
    )
    return _format_xaxis(fig)


# -----------------------------
# Portfolio cumulative returns
# -----------------------------
def plot_cumulative_returns(cum_series: pd.Series) -> go.Figure:
    """
    Plot cumulative portfolio performance.

    Y-axis:
        - Cumulative value (base = 1.0).
    """
    fig = px.line(cum_series, title="Portfolio Cumulative Returns")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Value (base = 1.0)",
        showlegend=False,
    )
    return _format_xaxis(fig)


# -----------------------------
# Correlation heatmap
# -----------------------------
def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Plot correlation matrix as heatmap (-1 to 1)."""
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Matrix (range: -1 to 1)",
    )
    fig.update_layout(
        xaxis_title="Asset",
        yaxis_title="Asset",
    )
    return fig


# -----------------------------
# Rolling volatility plot 
# -----------------------------
def plot_rolling_volatility(rolling_vol: pd.DataFrame) -> go.Figure:
    """
    Plot rolling volatility for each asset.

    rolling_vol:
        - in decimal (e.g. 0.15)
    Display:
        - converted to %.
    """
    vol_pct = rolling_vol * 100.0
    fig = px.line(vol_pct, title="Rolling Volatility (%)")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        legend_title="Assets",
    )
    return _format_xaxis(fig)


# =====================================================
# ADVANCED PRO PLOTS (Mode Pro)
# =====================================================

# -----------------------------
# Rolling Beta
# -----------------------------
def plot_rolling_beta(beta_series: pd.Series, benchmark: str) -> go.Figure:
    """Plot rolling beta vs benchmark (unitless)."""
    fig = px.line(
        beta_series,
        title=f"Rolling Beta vs {benchmark} (unitless)",
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Beta (unitless)",
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black"),
        showlegend=False,
    )
    return _format_xaxis(fig)


# -----------------------------
# Drawdown (Pro version)
# -----------------------------
def plot_drawdown(drawdown: pd.Series) -> go.Figure:
    """
    Plot portfolio drawdown.

    Input:
        - drawdown in decimal (e.g. -0.20)
    Output:
        - shown as percentage.
    """
    dd_pct = drawdown * 100.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd_pct.index,
            y=dd_pct,
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
            line=dict(color="red"),
        )
    )
    fig.update_layout(
        title="Portfolio Drawdown (%)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
    )
    return _format_xaxis(fig)


# -----------------------------
# Efficient Frontier
# -----------------------------
def plot_efficient_frontier(
    ef_results: pd.DataFrame,
    curr_vol: float,
    curr_ret: float,
) -> go.Figure:
    """
    Plot Efficient Frontier and portfolio point.

    Inputs:
        - ef_results: Volatility (decimal), Return (decimal), Sharpe
        - curr_vol: portfolio volatility (decimal)
        - curr_ret: portfolio return (decimal)

    Display:
        - % scale for both axes.
    """
    df = ef_results.copy()
    df["Volatility (%)"] = df["Volatility"] * 100.0
    df["Return (%)"] = df["Return"] * 100.0

    fig = px.scatter(
        df,
        x="Volatility (%)",
        y="Return (%)",
        color="Sharpe",
        color_continuous_scale="Viridis",
        title="Efficient Frontier — Random Portfolios",
        height=600,
    )

    fig.add_scatter(
        x=[curr_vol * 100.0],
        y=[curr_ret * 100.0],
        mode="markers",
        marker=dict(color="red", size=14, line=dict(color="black", width=2)),
        name="Your Portfolio",
    )

    fig.update_layout(
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
    )

    return fig


# -----------------------------
# Rolling Correlation (optional)
# -----------------------------
def plot_rolling_correlation(
    returns: pd.DataFrame,
    asset1: str,
    asset2: str,
    window: int = 60,
) -> go.Figure:
    """Plot rolling correlation between two assets (-1 to 1)."""
    rolling_corr = returns[asset1].rolling(window).corr(returns[asset2])

    fig = px.line(
        rolling_corr,
        title=f"Rolling Correlation: {asset1} vs {asset2}",
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Correlation ([-1, 1])",
        yaxis=dict(range=[-1, 1]),
        showlegend=False,
    )

    return _format_xaxis(fig)
