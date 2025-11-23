#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math

# =====================================================
# GLOBAL SETTINGS
# =====================================================

DATE_FORMAT = "%d %b %Y"  # unified date format for the whole dashboard


def _format_xaxis(fig: go.Figure, start_date=None) -> go.Figure:
    """Uniform x-axis formatting for all charts + adaptive date range.

    The adaptive range only considers x-values where the corresponding y
    is not NaN / None, so purely-empty periods (all-NaN) do not extend
    the visible timeline.
    """
    # Standard visual formatting
    fig.update_xaxes(
        tickformat=DATE_FORMAT,
        ticks="outside",
        ticklabelmode="period",
    )

    # Adaptive range based on non-null data points
    xs = []
    for tr in fig.data:
        x = getattr(tr, "x", None)
        y = getattr(tr, "y", None)

        if x is None or y is None:
            continue

        # x and y should have the same length; we only keep dates
        # where y is not None / not NaN.
        for xi, yi in zip(x, y):
            if yi is None:
                continue
            try:
                val = float(yi)
                if math.isnan(val):
                    continue
            except (TypeError, ValueError):
                # Non-numeric (e.g. categorical) y: keep the point
                pass

            xs.append(pd.to_datetime(xi))

    if xs:
        fig.update_xaxes(range=[min(xs), max(xs)])

    # Adaptive range with optional start_date override
    xs=[]
    for tr in fig.data:
        x=getattr(tr,'x',None); y=getattr(tr,'y',None)
        if x is None or y is None: continue
        for xi, yi in zip(x,y):
            if yi is None: continue
            try:
                v=float(yi)
                if math.isnan(v): continue
            except: pass
            xs.append(pd.to_datetime(xi))
    if xs:
        min_x, max_x = min(xs), max(xs)
        if start_date is not None:
            min_x = pd.to_datetime(start_date)
        fig.update_xaxes(range=[min_x, max_x])
    return fig


def _unit_for_asset(ticker: str) -> str:
    """Return unit of measure for the asset (Amundi-style)."""
    if ticker.endswith("-USD"):
        return "$"
    if ticker.endswith(".PA") or ticker.endswith(".AS"):
        return "€"
    if ticker in [
        "SPY", "AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL",
        "TSLA", "BRK-B", "JPM", "V", "LLY", "NVO"
    ]:
        return "$"
    if ticker == "GC=F":
        return "$/oz"
    if ticker == "BZ=F":
        return "$/bbl"
    if ticker.startswith("^") or ticker.endswith("=RR"):
        return "%"
    return ""


# =====================================================
# HELPERS
# =====================================================

def _smart_format(v):
    """Format numeric value with K/M/B for hover labels."""
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
    For each asset column:
    - Before the first REAL price -> 0
    - From the first real price onward -> keep true prices,
      with forward-fill only for internal gaps.
    Assets with full history (non-NaN from the start date)
    are left unchanged (no 0 injected).
    """
    df = df.copy()

    for col in df.columns:
        first_valid = df[col].first_valid_index()

        if first_valid is None:
            # No data at all for this asset -> full column = 0
            df[col] = 0
            continue

        # Mask strictly BEFORE the first valid price
        mask_before = df.index < first_valid
        # Before IPO: NaN -> 0, don't touch anything after
        df.loc[mask_before, col] = df.loc[mask_before, col].fillna(0)

        # From IPO date onward: ffill to smooth internal gaps
        df.loc[~mask_before, col] = df.loc[~mask_before, col].ffill()

    return df


# =====================================================
# REAL PRICE CHART
# =====================================================

def plot_real_prices(price_df: pd.DataFrame, units: dict, title: str = "Real Price Chart") -> go.Figure:
    """
    Real-price chart with units:
    - Full-history assets keep their true prices from the start.
    - Late-IPO assets are displayed at 0 before IPO, then with real prices.
    """
    # Trim leading all-NaN rows for macro series
    price_df = price_df.loc[price_df.notna().any(axis=1)]


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


# =====================================================
# SIMPLE PRICE SERIES (NOT THEMATIC)
# =====================================================

def plot_price_series(price_df: pd.DataFrame) -> go.Figure:
    """
    Plot multi-asset price series with correct currency/tooltips.
    """

    df_reset = price_df.reset_index()
    df_reset = df_reset.rename(columns={df_reset.columns[0]: "Date"})

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


# =====================================================
# NORMALIZED PERFORMANCE (BASE 100)
# =====================================================

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
            return col * np.nan
        base = col.loc[first_valid]
        if base == 0 or pd.isna(base):
            return col * np.nan
        return (col / base) * 100.0

    df_norm = price_df.apply(_normalize_column)

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


# =====================================================
# CUMULATIVE RETURNS (unit: base 1.0)
# =====================================================


def plot_cumulative_returns(
    cum_series: pd.Series,
    bench_cum: pd.Series = None,
    bench_name: str = "Benchmark",
) -> go.Figure:
    """
    Plot cumulative returns with optional benchmark.
    No shading is used to ensure full visibility of both curves.
    """

    fig = go.Figure()

    # ----------------------------------------------------
    # Align benchmark on portfolio index
    # ----------------------------------------------------
    if bench_cum is not None:
        bench_cum = bench_cum.reindex(cum_series.index).ffill()

    # ----------------------------------------------------
    # Portfolio curve (always visible)
    # ----------------------------------------------------
    fig.add_trace(go.Scatter(
        x=cum_series.index,
        y=cum_series.values,
        mode="lines",
        name="Portfolio",
        line=dict(width=3, color="#72b7b2"),
    ))

    # ----------------------------------------------------
    # Benchmark curve (drawn second, dashed)
    # ----------------------------------------------------
    if bench_cum is not None:
        fig.add_trace(go.Scatter(
            x=bench_cum.index,
            y=bench_cum.values,
            mode="lines",
            name=bench_name,
            line=dict(width=2, dash="dash", color="#4c78a8"),
        ))

    # ----------------------------------------------------
    # Base = 1 reference line
    # ----------------------------------------------------
    fig.add_hline(
        y=1,
        line=dict(color="white", width=1, dash="dot"),
        annotation_text="Base = 1.0",
        annotation_position="top left"
    )

    fig.update_layout(
        title="Portfolio vs Benchmark — Cumulative (base = 1)",
        xaxis_title="Date",
        yaxis_title="Cumulative Value",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
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

def plot_efficient_frontier(
    ef_results: pd.DataFrame, curr_vol: float, curr_ret: float
) -> go.Figure:
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

def plot_rolling_correlation(
    returns: pd.DataFrame, asset1: str, asset2: str, window: int = 60
) -> go.Figure:
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
