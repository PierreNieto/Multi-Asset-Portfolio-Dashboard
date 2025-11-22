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

#def _unit_for_asset(asset):
#    ASSET_UNITS = {
#        "AAPL": "$",
#        "SPY": "$",
#        "BTC-USD": "$",
#        "BZ=F": "$",
#        "GC=F": "$",
#        "^TNX": "%",  # yield index
#        "ACA.PA": "€",
#        "AIR.PA": "€",
#    }
#    return ASSET_UNITS.get(asset, "")

# Utility for consistent x-axis formatting
def _format_xaxis(fig):
    fig.update_xaxes(
        tickformat=DATE_FORMAT,   # show day + month + year
        ticks="outside",
        ticklabelmode="period"    # prevents label overlap
    )
    return fig


# =====================================================
# PRICE SERIES WITH UNITS IN TOOLTIP
# =====================================================


def _smart_format(v):
    """Format numeric value with K/M/B."""
    if v is None:
        return ""
    if abs(v) >= 1_000_000_000:
        return f"{v/1_000_000_000:.2f}B"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:.2f}k"
    return f"{v:.2f}"

def fill_missing_with_zero_until_first_valid(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each column:
    - Before first valid price → fill with 0
    - After → keep real prices
    This ensures assets with later IPO still appear on charts starting at 0.
    """
    df = df.copy()
    for col in df.columns:
        first_valid = df[col].first_valid_index()
        if first_valid is not None:
            df.loc[:first_valid, col] = df[col].loc[:first_valid].fillna(0)
        else:
            # If column is 100% NaN, fill entire column with 0
            df[col] = 0
    return df


def plot_real_prices(price_df, units, title="Real Price Chart"):
    """
    Graphique PRIX RÉELS avec unités adaptées automatiquement.
    Unité dans la VALUE uniquement.
    Jamais dans le nom de la série.
    """
    # Convert NaN histories to 0 until first valid value
    price_df = fill_missing_with_zero_until_first_valid(price_df)
    fig = go.Figure()

    colors = px.colors.qualitative.Set2
    color_map = {col: colors[i % len(colors)] for i, col in enumerate(price_df.columns)}

    for col in price_df.columns:
        unit = units.get(col, "")

        # Format final du hover (ex : 60.8k $)
        hover_unit = f" {unit}" if unit else ""

        fig.add_trace(
            go.Scatter(
                x=price_df.index,
                y=price_df[col],
                mode="lines",
                name=col,  # AUCUNE unité ici
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


def plot_price_series(price_df: pd.DataFrame) -> go.Figure:
    """
    Plot multi-asset price series with correct currency/tooltips.
    """

    # Reset index and enforce a proper date column name
    df_reset = price_df.reset_index()
    df_reset = df_reset.rename(columns={df_reset.columns[0]: "Date"})

    # Long format needed for custom tooltip
    df_long = df_reset.melt(
        id_vars="Date",
        var_name="Asset",
        value_name="Value",
    )

    df_long["Unit"] = df_long["Asset"].apply(_unit_for_asset)

    fig = px.line(
        df_long,
        x="Date",
        y="Value",
        color="Asset",
        title="Asset Price Series",
        custom_data=["Unit"],
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Date: %{x}<br>"
            "Value: %{y:.2f} %{customdata[0]}<br>"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price / Level (native units)",
        legend_title="Assets",
    )

    return _format_xaxis(fig)



DATE_FORMAT = "%d %b %Y"  # si tu l'as déjà, garde-le

def _format_xaxis(fig):
    fig.update_xaxes(
        tickformat=DATE_FORMAT,
        ticks="outside",
        ticklabelmode="period",
    )
    return fig


def plot_normalized_series(
    price_df: pd.DataFrame,
    title: str = "Normalized Performance (base = 100)",
) -> go.Figure:
    """
    Normalized price comparison (base 100).
    Each asset is normalized on its own first valid price, so:
    - assets with shorter history (IPO later) start when data exists
    - no backward fill is used
    - no fully-NaN first row that would kill the chart.
    """

    def _normalize_column(col: pd.Series) -> pd.Series:
        first_valid = col.first_valid_index()
        if first_valid is None:
            # column is all NaN -> keep it NaN
            return col * np.nan
        base = col.loc[first_valid]
        if base == 0 or pd.isna(base):
            # avoid division by zero or NaN
            return col * np.nan
        return (col / base) * 100.0

    df_norm = price_df.apply(_normalize_column)

    # Reset index for plotly
    df_norm = df_norm.reset_index().rename(columns={"index": "Date"})

    fig = go.Figure()

    for asset in price_df.columns:
        fig.add_trace(
            go.Scatter(
                x=df_norm["Date"],
                y=df_norm[asset],
                mode="lines",
                name=asset,
                hovertemplate=(
                    f"<b>{asset}</b><br>"
                    "Date: %{x|%d %b %Y}<br>"
                    "Performance Index: %{y:.2f}<br>"
                    f"Unit: {_unit_for_asset(asset)}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Performance Index (base = 100)",
        legend_title="Assets",
    )

    return _format_xaxis(fig)

def _unit_for_asset(ticker: str) -> str:
    """Return unit of measure for the asset."""
    if ticker.endswith("-USD"):
        return "$"
    if ticker.endswith(".PA") or ticker.endswith(".AS"):
        return "€"
    if ticker in ["SPY", "AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA", "BRK-B", "JPM", "V", "LLY", "NVO"]:
        return "$"
    if ticker == "GC=F":
        return "$/oz"
    if ticker == "BZ=F":
        return "$/bbl"
    if ticker.startswith("^") or ticker.endswith("=RR"):
        return "%"
    return ""


# =====================================================
# CUMULATIVE RETURNS (unit: base 1.0)
# =====================================================
def plot_cumulative_returns(cum_series: pd.Series) -> go.Figure:
    fig = px.line(cum_series, title="Portfolio Cumulative Returns")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Value (base = 1.0)",
        showlegend=False,
    )
    return _format_xaxis(fig)

# =====================================================
# CORRELATION HEATMAP
# =====================================================
def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Matrix (range: -1 to 1)",
    )
    fig.update_layout(xaxis_title="Asset", yaxis_title="Asset")
    return fig

# =====================================================
# ROLLING VOLATILITY (converted to %)
# =====================================================
def plot_rolling_volatility(rolling_vol: pd.DataFrame) -> go.Figure:
    vol_pct = rolling_vol * 100.0
    fig = px.line(vol_pct, title="Rolling Volatility (%)")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        legend_title="Assets",
    )
    return _format_xaxis(fig)

# =====================================================
# ROLLING BETA (unitless)
# =====================================================
def plot_rolling_beta(beta_series: pd.Series, benchmark: str) -> go.Figure:
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

# =====================================================
# DRAWDOWN (converted to %)
# =====================================================
def plot_drawdown(drawdown: pd.Series) -> go.Figure:
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

# =====================================================
# EFFICIENT FRONTIER (vol & return in %)
# =====================================================
def plot_efficient_frontier(ef_results: pd.DataFrame, curr_vol: float, curr_ret: float) -> go.Figure:
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

# =====================================================
# ROLLING CORRELATION
# =====================================================
def plot_rolling_correlation(returns: pd.DataFrame, asset1: str, asset2: str, window: int = 60) -> go.Figure:
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
