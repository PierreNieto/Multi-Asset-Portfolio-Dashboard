#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#---------------------------
# I got an issue with the importation with Streamlit so I
# add the project root directory to PYTHONPATH.
# Because Streamlit executes scripts from a temporary working directory,
# which prevents relative imports such as `from app.portfolio ...`
# from working correctly when running pages directly.
#
# So, by dynamically adding the project root to sys.path, I ensure
# that the `app` package can always be imported, both when:
# - running this page directly with `streamlit run`,
# - running the full dashboard through the global main.py.
#---------------------------

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

#---------------------------

import datetime as dt

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
    compute_var_cvar,
    random_portfolios,
)
from app.portfolio.plots import (
    plot_price_series,
    plot_cumulative_returns,
    plot_correlation_heatmap,
    plot_rolling_volatility,
    plot_rolling_beta,       
    plot_drawdown,            
    plot_efficient_frontier, 
)
from app.portfolio.macro_loader import load_macro_data


def _map_freq_label_to_code(label: str) -> str:
    """Map UI frequency label to pandas resample code."""
    if label == "Daily":
        return "D"
    if label == "Weekly":
        return "W"
    if label == "Monthly":
        return "M"
    return "D"


def _compute_rolling_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling beta of portfolio vs benchmark."""
    cov = portfolio_returns.rolling(window).cov(benchmark_returns)
    var = benchmark_returns.rolling(window).var()
    beta = cov / var
    return beta


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
        raw_weights = []

        # Sliders
        for ticker in selected_tickers:
            w = st.sidebar.slider(
                f"Weight for {ticker}",
                min_value=0.0,
                max_value=1.0,
                value=1.0 / len(selected_tickers),
                step=0.01,
            )
            raw_weights.append(w)

        # Safe normalization
        total = sum(raw_weights)
        if total > 0:
            custom_weights = [w / total for w in raw_weights]
        else:
            # fallback equal-weights
            custom_weights = [1 / len(raw_weights)] * len(raw_weights)

        st.sidebar.write(f"**Normalized weights:** {custom_weights}")

    else:
        # If not in custom mode, do nothing
        raw_weights = None



    st.sidebar.markdown("### Rolling metrics")
    rolling_window = st.sidebar.slider(
        "Rolling window (days)",
        min_value=10,
        max_value=120,
        value=20,
        step=5,
    )

    st.sidebar.subheader("Benchmark Selection")
    benchmark = st.sidebar.selectbox(
        "Benchmark (for Rolling Beta)",
        ["SPY", "QQQ", "^FCHI", "^GSPC"],
        index=0,
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

    # Clean and align price data
    prices = clean_price_data(prices)

    # Ensure prices is always a DataFrame (even if only one asset is selected)
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    # Final alignment choice: forward/backward fill for a rectangular structure
    prices = prices.ffill().bfill()

    # Maybe resample if needed
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

    # Ensure DataFrame even with 1 asset
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
        


    # Ensure portfolio returns is a Series
    if isinstance(port_ret, pd.DataFrame):
        port_ret = port_ret.iloc[:, 0]

    # -----------------------------
    # DEBUG DIAGNOSTIC RETURNS
    # -----------------------------
    #st.write("---- DEBUG RETURNS ----")

    #st.write("asset_returns.describe():", asset_returns.describe())
    #st.write("asset_returns max:", asset_returns.max().max())
    #st.write("asset_returns min:", asset_returns.min().min())

    #st.write("port_ret.describe():", port_ret.describe())
    #st.write("Max daily portfolio return:", float(port_ret.max()))
    #st.write("Min daily portfolio return:", float(port_ret.min()))
    #st.write("Mean daily portfolio return:", float(port_ret.mean()))

    #st.write("Length prices:", len(prices))
    #st.write("Length asset_returns:", len(asset_returns))
    #st.write("Length port_ret:", len(port_ret))

    #st.write("Price index sample:", prices.index[:5])
    #st.write("Returns index sample:", asset_returns.index[:5])
    #st.write("Portfolio index sample:", port_ret.index[:5])

    port_cum = cumulative_returns(port_ret)
    corr_mat = correlation_matrix(asset_returns)

   
    # annualization factor

    if freq_code == "D":
        ann_factor = 252
    elif freq_code == "W":
        ann_factor = 52
    elif freq_code == "M":
        ann_factor = 12
    else:
        ann_factor = 252  # fallback

    # -----------------------------
    # Core metrics 
    # -----------------------------
    asset_ann_ret = annualized_return(asset_returns, freq=ann_factor)
    asset_ann_vol = annualized_volatility(asset_returns, freq=ann_factor)
    asset_sharpe  = sharpe_ratio(asset_returns, freq=ann_factor)

    port_ann_ret = annualized_return(port_ret, freq=ann_factor)
    port_ann_vol = annualized_volatility(port_ret, freq=ann_factor)
    port_sharpe  = sharpe_ratio(port_ret, freq=ann_factor)


    # Drawdown & max drawdown (used in Pro / Tail Risk)
    drawdown = port_cum / port_cum.cummax() - 1
    max_drawdown = drawdown.min() if not drawdown.empty else np.nan

    # Tail risk metrics: VaR & CVaR (used in Pro / Tail Risk)
    var_5, cvar_5 = compute_var_cvar(port_ret.dropna(), level=5)

    # Diversification ratio
    try:
        div_ratio = diversification_ratio(asset_returns, weights_used)
    except Exception:
        div_ratio = None

    # -----------------------------
    # Portfolio summary (top KPIs)
    # -----------------------------
    st.subheader("Portfolio summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio annualized return", f"{port_ann_ret * 100:.2f}%")
    col2.metric("Portfolio annualized volatility", f"{port_ann_vol * 100:.2f}%")
    if np.isfinite(port_sharpe):
        col3.metric("Portfolio Sharpe ratio", f"{port_sharpe:.2f}")
    else:
        col3.metric("Portfolio Sharpe ratio", "N/A")
    if np.isfinite(max_drawdown):
        col4.metric("Max drawdown", f"{max_drawdown * 100:.2f}%")
    else:
        col4.metric("Max drawdown", "N/A")

    if div_ratio is not None:
        st.caption(f"Diversification ratio: {div_ratio:.2f}")

    # Asset-level metrics DataFrame
    metrics_df = pd.DataFrame(
        {
            "Annualized return": asset_ann_ret,
            "Annualized volatility": asset_ann_vol,
            "Sharpe ratio": asset_sharpe,
        }
    )

    # -----------------------------
    # Mode selector: Standard vs Pro
    # -----------------------------
    mode = st.radio(
        "Display mode",
        ["Standard", "Pro"],
        index=0,
        horizontal=True,
    )

    # =====================================================
    # MODE STANDARD
    # =====================================================
    if mode == "Standard":
        overview_tab, risk_tab, perf_tab, macro_tab = st.tabs(
            ["Overview", "Risk Analysis", "Performance", "Macro Dashboard"]
        )

        # -----------------------------
        # OVERVIEW TAB (Standard)
        # -----------------------------
        with overview_tab:
            st.subheader("Price and Portfolio Overview")
            price_fig = plot_price_series(prices)
            st.plotly_chart(price_fig, use_container_width=True)

            port_cum_fig = plot_cumulative_returns(port_cum)
            st.plotly_chart(port_cum_fig, use_container_width=True)

        # -----------------------------
        # RISK ANALYSIS TAB (Standard)
        # -----------------------------
        with risk_tab:
            st.subheader("Correlation Matrix")
            corr_fig = plot_correlation_heatmap(corr_mat)
            st.plotly_chart(corr_fig, use_container_width=True)

            st.subheader("Rolling Volatility")
            roll_vol = rolling_volatility(asset_returns, window=rolling_window)
            if not roll_vol.empty:
                roll_vol_fig = plot_rolling_volatility(roll_vol)
                st.plotly_chart(roll_vol_fig, use_container_width=True)
            else:
                st.info("Not enough data to compute rolling volatility.")

        # -----------------------------
        # PERFORMANCE TAB (Standard)
        # -----------------------------
        with perf_tab:
            st.subheader("Asset-level metrics")
            st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

        # -----------------------------
        # MACRO TAB (Standard)
        # -----------------------------
        with macro_tab:
            st.subheader("Macro Indicators (FRED)")

            if not macro_data_aligned:
                st.info("No macro data available for the selected start date.")
            else:
                for name, df in macro_data_aligned.items():
                    st.write(f"### {name}")
                    st.line_chart(df, height=200, use_container_width=True)

    # =====================================================
    # MODE PRO
    # =====================================================
    else:
        pro_overview_tab, pro_risk_tab, pro_beta_tab, pro_tail_tab, pro_frontier_tab, pro_macro_tab = st.tabs(
            [
                "Overview",
                "Risk",
                "Rolling Beta",
                "Tail Risk",
                "Efficient Frontier",
                "Macro Dashboard",
            ]
        )

        # -----------------------------
        # PRE-COMPUTE THINGS USED ONLY IN PRO
        # -----------------------------

        # Benchmark returns for beta
        rolling_beta_series = None
        try:
            bench_prices_raw = load_multi_asset_data(
                tickers=[benchmark],
                start=start_date.isoformat(),
            )
            if isinstance(bench_prices_raw, pd.DataFrame):
                bench_prices = bench_prices_raw.iloc[:, 0]
            else:
                bench_prices = bench_prices_raw

            bench_prices = bench_prices.sort_index().ffill().bfill()

            if freq_code != "D":
                bench_prices = resample_price_data(
                    bench_prices.to_frame(), freq=freq_code, how="last"
                ).iloc[:, 0]

            if return_type == "Simple returns":
                bench_ret_series = compute_simple_returns(
                    bench_prices.to_frame()
                ).iloc[:, 0]
            else:
                bench_ret_series = compute_log_returns(
                    bench_prices.to_frame()
                ).iloc[:, 0]

            aligned_port, aligned_bench = align_dataframes(
                [
                    port_ret.to_frame("Portfolio"),
                    bench_ret_series.to_frame("Benchmark"),
                ]
            )
            port_ret_aligned = aligned_port["Portfolio"]
            bench_ret_aligned = aligned_bench["Benchmark"]

            rolling_beta_series = _compute_rolling_beta(
                port_ret_aligned, bench_ret_aligned, rolling_window
            )
        except Exception:
            rolling_beta_series = None

        # -----------------------------
        # OVERVIEW TAB (Pro)
        # -----------------------------
        with pro_overview_tab:
            st.subheader("Price and Portfolio Overview (Pro)")
            price_fig = plot_price_series(prices)
            st.plotly_chart(price_fig, use_container_width=True)

            port_cum_fig = plot_cumulative_returns(port_cum)
            st.plotly_chart(port_cum_fig, use_container_width=True)

        # -----------------------------
        # RISK TAB (Pro)
        # -----------------------------
        with pro_risk_tab:
            st.subheader("Correlation & Volatility")

            corr_fig = plot_correlation_heatmap(corr_mat)
            st.plotly_chart(corr_fig, use_container_width=True)

            st.subheader("Rolling Volatility (Pro)")
            roll_vol = rolling_volatility(asset_returns, window=rolling_window)
            if not roll_vol.empty:
                roll_vol_fig = plot_rolling_volatility(roll_vol)
                st.plotly_chart(roll_vol_fig, use_container_width=True)
            else:
                st.info("Not enough data to compute rolling volatility.")

        # -----------------------------
        # ROLLING BETA TAB (Pro)
        # -----------------------------
        with pro_beta_tab:
            st.subheader(f"Rolling Beta vs {benchmark}")

            if rolling_beta_series is None or rolling_beta_series.dropna().empty:
                st.info("Not enough data to compute rolling beta for this configuration.")
            else:
                
                beta_fig = plot_rolling_beta(rolling_beta_series, benchmark)
                st.plotly_chart(beta_fig, use_container_width=True)

                # Latest beta KPI
                last_beta = rolling_beta_series.dropna().iloc[-1]
                st.metric("Latest rolling beta", f"{last_beta:.2f}")


        # -----------------------------
        # TAIL RISK TAB (Pro)
        # -----------------------------
        with pro_tail_tab:
            st.subheader("Drawdown & Tail Risk")

            if not drawdown.empty:
                dd_fig = plot_drawdown(drawdown)
                st.plotly_chart(dd_fig, use_container_width=True)
            else:
                st.info("Not enough data to compute drawdown.")

            st.markdown("---")
            st.subheader("VaR & CVaR (5%)")

            colA, colB = st.columns(2)
            colA.metric("Value-at-Risk (5%)", f"{var_5:.2%}")
            colB.metric("Conditional VaR (5%)", f"{cvar_5:.2%}")

        # -----------------------------
        # EFFICIENT FRONTIER TAB (Pro)
        # -----------------------------
        with pro_frontier_tab:
            st.subheader("Efficient Frontier (Markowitz)")

            n_sim = st.slider(
                "Number of simulated portfolios",
                min_value=2000,
                max_value=20000,
                value=5000,
                step=1000,
            )

            cov_matrix = asset_returns.cov()
            ef_results, ef_weights = random_portfolios(asset_returns, cov_matrix, n_sim)

            current_vol = np.sqrt(
                np.dot(weights_used, np.dot(cov_matrix * 252, weights_used))
            )
            current_ret = np.dot(
                weights_used,
                asset_returns.mean() * 252,
            )

            frontier_fig = plot_efficient_frontier(ef_results, current_vol, current_ret)
            st.plotly_chart(frontier_fig, use_container_width=True)

        # -----------------------------
        # MACRO TAB (Pro)
        # -----------------------------
        with pro_macro_tab:
            st.subheader("Macro Indicators (FRED) — Pro View")

            if not macro_data_aligned:
                st.info("No macro data available for the selected start date.")
            else:
                for name, df in macro_data_aligned.items():
                    st.write(f"### {name}")
                    st.line_chart(df, height=200, use_container_width=True)

    # -----------------------------
    # FOOTER — Glossary (small text)
    # -----------------------------
    st.markdown("---")
    st.markdown(
        """
    <div style="font-size:12px; opacity:0.75">
    <strong> Glossary of symbols & abbreviations</strong><br><br>

    <u>Assets in this dashboard</u><br>
    • <strong>AAPL</strong> — Apple Inc. (USD, Nasdaq)<br>
    • <strong>SPY</strong> — S&P 500 ETF (USD)<br>
    • <strong>ACA.PA</strong> — Crédit Agricole SA (EUR, Paris)<br>
    • <strong>AIR.PA</strong> — Airbus SE (EUR, Paris)<br>
    • <strong>BZ=F</strong> — Brent Crude Oil Futures (USD/bbl)<br>
    • <strong>^TNX</strong> — US 10-Year Treasury Yield Index<br>
    • <strong>GC=F</strong> — Gold Futures (USD/oz)<br>
    • <strong>BTC-USD</strong> — Bitcoin (USD)<br><br>

    <u>Main quantities</u><br>
    • <strong>Volatility (%)</strong> — annualized standard deviation of returns<br>
    • <strong>Drawdown (%)</strong> — % drop from running peak<br>
    • <strong>Beta</strong> — sensitivity to the benchmark<br>
    • <strong>VaR 5%</strong> — loss threshold exceeded 5% of the time<br>
    • <strong>CVaR 5%</strong> — average loss beyond the VaR<br>
    </div>
    """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    run_portfolio_page()
