#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


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
# Rolling volatility plot (optional)
# -----------------------------
def plot_rolling_volatility(rolling_vol: pd.DataFrame) -> go.Figure:
    """Plot rolling volatility for each asset."""
    fig = px.line(rolling_vol, title="Rolling Volatility")
    fig.update_layout(xaxis_title="Date", yaxis_title="Volatility")
    return fig
