#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#---------------------------

#I got an issue with the importation with streamlit so i
# add the project root directory to PYTHONPATH.
# Because streamlit executes scripts from a temporary working directory,
# which prevents relative imports such as `from app.portfolio ...`
# from working correctly when running pages directly.

# So, by dynamically adding the project root to sys.path, i ensure
# that the `app` package can always be imported, both when:
# - running this page directly with `streamlit run`,
# - running the full dashboard through the global main.py.

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

#---------------------------

import datetime as dt

import streamlit as st
import pandas as pd

from app.portfolio.data_loader import load_multi_asset_data, DEFAULT_TICKERS
from app.portfolio.preprocessing import (
    compute_simple_returns,
    compute_log_returns,
    resample_price_data,
    rolling_volatility,
    clean_price_data,
    align_dataframes,
)
from app.portfolio.portfolio_engine import (
    equal_weight_portfolio,
    custom_weight_portfolio,
    cumulative_returns,
)
from app.portfolio.metrics import (
    correlation_matrix,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    diversification_ratio,
)
from app.portfolio.plots import (
    plot_price_series,
    plot_cumulative_returns,
    plot_correlation_heatmap,
    plot_rolling_volatility,
)
from app.portfolio.macro_loader import load_macro_data
import numpy as np


def _map_freq_label_to_code(label: str) -> str:
    """Map UI frequency label to pandas resample code."""
    if label == "Daily":
        return "D"
    if label == "Weekly":
        return "W"
    if label == "Monthly":
        return "M"
    return "D"


def run_portfolio_page():
    st.title("Multi-Asset Portfolio Dashboard")
    st.caption(
        "Quant B — Multi-Asset Portfolio Module: "
        "risk–return analysis across equities, rates, commodities and crypto."
    )

    # -----------------------------
    # Sidebar controls
    # -----------------------------
    st.sidebar.header("Portfolio Settings")

    st.sidebar.markdown("### Asset universe")
    selected_tickers = st.sidebar.multiselect(
        "Select assets",
        options=DEFAULT_TICKERS,
        default=DEFAULT_TICKERS,
    )

    start_date = st.sidebar.date_input(
        "Start date",
        value=dt.date(2015, 1, 1),
        min_value=dt.date(2000, 1, 1),
        max_value=dt.date.today(),
    )

    freq_label = st.sidebar.selectbox(
        "Data frequency",
        options=["Daily", "Weekly", "Monthly"],
        index=0,
    )

    return_type = st.sidebar.selectbox(
        "Return type",
        options=["Simple returns", "Log returns"],
        index=0,
    )

    st.sidebar.markdown("### Portfolio strategy")
    strategy = st.sidebar.selectbox(
        "Strategy",
        options=["Equal weight", "Custom weights"],
        index=0,
    )

    # Custom weights controls (only if needed)
    custom_weights = None
    if strategy == "Custom weights" and selected_tickers:
        st.sidebar.markdown("#### Custom weights")
        custom_weights = []
        for ticker in selected_tickers:
            w = st.sidebar.slider(
                f"Weight for {ticker}",
                min_value=0.0,
                max_value=1.0,
                value=1.0 / len(selected_tickers),
                step=0.01,
            )
            custom_weights.append(w)

    st.sidebar.markdown("### Rolling metrics")
    rolling_window = st.sidebar.slider(
        "Rolling window (days)",
        min_value=10,
        max_value=120,
        value=20,
        step=5,
    )

    # -----------------------------
    # Basic validations
    # -----------------------------
    if not selected_tickers:
        st.info("Please select at least one asset in the sidebar to start.")
        return

    # -----------------------------
    # Data loading
    # -----------------------------
    try:
        prices = load_multi_asset_data(
            tickers=selected_tickers,
            start=start_date.isoformat(),
        )
    except Exception as e:
        st.error(f"Failed to load market data: {e}")
        return

    if prices.empty:
        st.warning("No price data available for the selected configuration.")
        return

    prices = clean_price_data(prices)

    # Ensure prices is always a DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    # use the full index and fill missing data
    # prices = prices.reindex(prices.index)   # useless now : guarantee of rectangular structure
    prices = prices.ffill().bfill()         # fill missing values forward/backward


#-------------------
    # Test : Align all asset series on the same index

    # Hard version, big loss of data :
    # prices = prices.dropna(how="any")  # strict align to common dates
    # Alternative softer :
    #prices = prices.ffill().bfill()   # full alignment by filling gaps

    # Remove assets with insufficient data (prevents Plotly crash)
    #prices = prices.loc[:, prices.notna().sum() > 10]

    #if prices.shape[1] == 0:
        #st.error("No assets have enough data to display. Try another date range.")
        #return
#-------------------


    # Maybe resample if needed..?
    freq_code = _map_freq_label_to_code(freq_label)
    if freq_code != "D":
        prices = resample_price_data(prices, freq=freq_code, how="last")

    # -----------------------------
    # Macro data (FRED)
    # -----------------------------
    macro_data = load_macro_data(start=start_date.isoformat())
    macro_series = []
    for key, df in macro_data.items():
        if not df.empty:
            macro_series.append(df)

    # Align macro and prices if we have macro data
    if macro_series:
        aligned_prices, *aligned_macro = align_dataframes([prices] + macro_series)
        prices = aligned_prices
        macro_data_aligned = {
            key: df for key, df in zip(macro_data.keys(), aligned_macro)
        }
    else:
        macro_data_aligned = {}

    # -----------------------------
    # Returns and portfolio
    # -----------------------------
    if return_type == "Simple returns":
        asset_returns = compute_simple_returns(prices)
    else:
        asset_returns = compute_log_returns(prices)

    # FIX: ensure DataFrame even with 1 asset
    if isinstance(asset_returns, pd.Series):
        asset_returns = asset_returns.to_frame()

    if asset_returns.empty:
        st.warning("Not enough data to compute returns.")
        return

    if strategy == "Equal weight":
        port_ret = equal_weight_portfolio(asset_returns)
        weights_used = [1.0 / asset_returns.shape[1]] * asset_returns.shape[1]
    else:
        if not custom_weights or sum(custom_weights) == 0:
            st.error("Custom weights must be strictly positive and not all zero.")
            return
        weights_used = custom_weights
        port_ret = custom_weight_portfolio(asset_returns, weights_used)

    port_cum = cumulative_returns(port_ret)
    corr_mat = correlation_matrix(asset_returns)

    # -----------------------------
    # Metrics
    # -----------------------------
    asset_ann_ret = annualized_return(asset_returns)
    asset_ann_vol = annualized_volatility(asset_returns)
    asset_sharpe = sharpe_ratio(asset_returns)

    port_ann_ret = annualized_return(port_ret)
    port_ann_vol = annualized_volatility(port_ret)
    port_sharpe = sharpe_ratio(port_ret)

    try:
        div_ratio = diversification_ratio(asset_returns, weights_used)
    except Exception:
        div_ratio = None

    # -----------------------------
    # Layout: metrics + charts
    # -----------------------------
    st.subheader("Portfolio summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio annualized return", f"{port_ann_ret * 100:.2f}%")
    col2.metric("Portfolio annualized volatility", f"{port_ann_vol * 100:.2f}%")
    if np.isfinite(port_sharpe):
        col3.metric("Portfolio Sharpe ratio", f"{port_sharpe:.2f}")
    else:
        col3.metric("Portfolio Sharpe ratio", "N/A")

    if div_ratio is not None:
        st.caption(f"Diversification ratio: {div_ratio:.2f}")

    # Asset-level table
    st.markdown("### Asset-level metrics")
    metrics_df = pd.DataFrame(
        {
            "Annualized return": asset_ann_ret,
            "Annualized volatility": asset_ann_vol,
            "Sharpe ratio": asset_sharpe,
        }
    )
    st.dataframe(metrics_df.style.format("{:.4f}"))

    # -----------------------------
    # Charts
    # -----------------------------
    st.markdown("### Price and portfolio charts")

    # Ensure prices is always a DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    # Align all columns on the full index and fill missing values
    prices = prices.reindex(prices.index)
    prices = prices.ffill().bfill()


    price_fig = plot_price_series(prices)
    st.plotly_chart(price_fig, use_container_width=True)

    port_cum_fig = plot_cumulative_returns(port_cum)
    st.plotly_chart(port_cum_fig, use_container_width=True)

    st.markdown("### Correlation matrix")
    corr_fig = plot_correlation_heatmap(corr_mat)
    st.plotly_chart(corr_fig, use_container_width=True)

    # Rolling volatility
    st.markdown("### Rolling volatility")
    roll_vol = rolling_volatility(asset_returns, window=rolling_window)
    if not roll_vol.empty:
        roll_vol_fig = plot_rolling_volatility(roll_vol)
        st.plotly_chart(roll_vol_fig, use_container_width=True)

    # -----------------------------
    # Macro section
    # -----------------------------
    st.markdown("### Macro indicators (FRED)")

    if not macro_data_aligned:
        st.info("No macro data available for the selected start date.")
    else:
        for name, df in macro_data_aligned.items():
            st.line_chart(df, height=200, use_container_width=True)

if __name__ == "__main__":
    run_portfolio_page()
