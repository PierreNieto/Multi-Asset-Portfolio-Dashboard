#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Unified date format for the whole dashboard
DATE_FORMAT = "%d %b %Y"  

# Mapping of units for each asset
def _unit_for_asset(ticker: str) -> str:
    """Return unit of measure for the given asset, Amundi-style."""
    if ticker.endswith("-USD"):
        return "$"
    if ticker.endswith(".PA") or ticker.endswith(".AS"):
        return "€"
    if ticker in ["SPY", "AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA", "NVO", "LLY", "BRK-B", "JPM", "V"]:
        return "$"
    if ticker == "GC=F":
        return "$/oz"
    if ticker == "BZ=F":
        return "$/bbl"
    if ticker.startswith("^") or ticker.endswith("=RR"):
        return "%"
    return ""

# ----------------------------------------------------------
#  PRE-IPO = 0 avec vraie data après
# ----------------------------------------------------------
def fill_missing_with_zero_until_first_valid(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each asset column:
    - Before the first REAL price → force to 0
    - After → keep true prices (with ffill for internal gaps only)
    """
    df = df.copy()
    for col in df.columns:
        first_valid = df[col].first_valid_index()

        if first_valid is None:
            df[col] = 0
            continue

        # Before first real price → 0
        df[col].loc[:first_valid] = df[col].loc[:first_valid].fillna(0)

        # After IPO → ffill internal holes
        df[col] = df[col].ffill()

    return df

# Utility for consistent x-axis formatting
def _format_xaxis(fig):
    fig.update_xaxes(
        tickformat=DATE_FORMAT,
        ticks="outside",
        ticklabelmode="period"
    )
    return fig

# =====================================================
# REAL PRICE CHART (with pre-IPO=0 logic)
# =====================================================
def _smart_format(v):
    if v is None:
        return ""
    if abs(v) >= 1_000_000_000:
        return f"{v/1_000_000_000:.2f}B"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:.2f}k"
    return f"{v:.2f}"

def plot_real_prices(price_df, units, title="Real Price Chart"):

    # Apply correct IPO logic
    price_df = fill_missing_with_zero_until_first_valid(price_df)

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    color_map = {col: colors[i % len(colors)] for i, col in enumerate(price_df.columns)}

    for col in price_df.columns:
        unit = units.get(col, "")
        hover_unit = f" {unit}" if unit else ""

        fig.add_trace(
            go.Scatter(
                x=price_df.index,
                y=price_df[col],
                mode="lines",
                name=col,
                line=dict(color=color_map[col], width=2),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Date = %{x}<br>"
                    "Value = %{customdata}" + hover_unit +
                    "<extra></extra>"
                ),
                customdata=[_smart_format(v) for v in price_df[col].values],
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price / Level (native units)",
        legend_title="Assets",
        height=600,
    )

    return fig
