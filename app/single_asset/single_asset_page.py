import datetime as dt
import pandas as pd
import streamlit as st

from app.single_asset.engine import (
    get_price_data,
    to_series,
    backtest_buy_and_hold,
    backtest_ma_crossover,
    compute_metrics,
    forecast_linear_trend,
)

# ---------------------------
# Streamlit app (Quant A)
# ---------------------------

def run_single_asset_page():
    st.title("Multi-Asset Portfolio Dashboard")
    st.subheader("Quant A - Single Asset Module")

    st.write(
        "This page analyses **one asset** with two trading strategies "
        "(Buy & Hold and Moving Average Crossover) plus a simple prediction bonus."
    )

    # -------- Sidebar controls --------
    st.sidebar.header("Asset settings")

    # Official single asset for the project: EUR/USD FX rate
    default_ticker = "EURUSD=X"
    ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", value=default_ticker)
    st.sidebar.caption(
        "Main asset studied in this project: EURUSD=X (EUR/USD FX rate). "
        "You can also test other FX tickers: "
        "GBPUSD=X (GBP/USD), USDJPY=X (USD/JPY), EURJPY=X (EUR/JPY), "
        "BTC-USD (Bitcoin), AAPL (Apple), MSFT (Microsoft)."
    )

    today = dt.date.today()
    default_start = today - dt.timedelta(days=365)

    start_date = st.sidebar.date_input("Start date", value=default_start)
    end_date = st.sidebar.date_input("End date", value=today)

    freq_label = st.sidebar.selectbox(
        "Data frequency",
        ["Daily", "Weekly", "Monthly"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.header("MA crossover strategy")

    # Windows adapted to the chosen frequency
    if freq_label == "Daily":
        short_min, short_max, short_default = 5, 50, 20
        long_min, long_max, long_default = 20, 200, 100
    elif freq_label == "Weekly":
        short_min, short_max, short_default = 2, 20, 5
        long_min, long_max, long_default = 5, 52, 20
    else:  # Monthly
        short_min, short_max, short_default = 2, 12, 3
        long_min, long_max, long_default = 3, 24, 6

    short_window = st.sidebar.slider(
        "Short MA window",
        min_value=short_min,
        max_value=short_max,
        value=short_default,
    )
    long_window = st.sidebar.slider(
        "Long MA window",
        min_value=long_min,
        max_value=long_max,
        value=long_default,
    )

    if short_window >= long_window:
        st.sidebar.error("Short window must be strictly smaller than long window.")

    # -------- Main logic --------
    if start_date >= end_date:
        st.error("Start date must be strictly before end date.")
    else:
        data_raw = get_price_data(ticker, start_date, end_date)

        if data_raw.empty:
            st.warning(
                "No data found for this ticker and date range. "
                "Please check the symbol (for FX, use e.g. EURUSD=X instead of EUR-USD)."
            )
        else:
            data = data_raw.copy()
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()

            # resample depending on chosen frequency
            if freq_label == "Weekly":
                data = data.resample("W").last()
            elif freq_label == "Monthly":
                data = data.resample("M").last()

            if data.empty:
                st.warning("Not enough data after resampling.")
                st.stop()

            if "Close" not in data.columns:
                st.error("Column 'Close' not found in downloaded data.")
                st.stop()

            close = data["Close"].dropna()

            if len(close) < 2:
                st.warning("Not enough points to run a backtest (need at least 2).")
                st.stop()

            # -------- Raw price chart --------
            st.write(f"### {ticker} price ({freq_label.lower()})")
            st.line_chart(close, width="stretch")

            # normalized price for main chart
            price_norm = to_series(close / close.iloc[0])

            # can we run MA strategy?
            # a bit more relaxed condition, but still avoids nonsense
            can_run_ma = len(close) > long_window + 2

            # -------- Backtests --------
            equity_bh, rets_bh = backtest_buy_and_hold(close)

            equity_ma = None
            rets_ma = None
            metrics_ma = None

            if not can_run_ma or short_window >= long_window:
                st.warning(
                    "Not enough data for the MA crossover strategy with the current "
                    "windows. Try reducing the long window or using Daily data."
                )
            else:
                equity_ma, rets_ma = backtest_ma_crossover(close, short_window, long_window)

            # -------- Main chart: price vs chosen strategy --------
            st.write("### Main chart: normalized price vs chosen strategy")

            options = ["Buy & Hold"]
            if equity_ma is not None:
                options.append(f"MA crossover ({short_window}/{long_window})")
                options.append("Both strategies")

            chosen = st.selectbox("Strategy to display", options)

            main_df = price_norm.to_frame(name="Normalized price")

            if chosen == "Buy & Hold":
                main_df["Buy & Hold equity"] = to_series(equity_bh)
            elif chosen.startswith("MA crossover") and equity_ma is not None:
                main_df[f"MA ({short_window}/{long_window}) equity"] = to_series(equity_ma)
            elif chosen == "Both strategies" and equity_ma is not None:
                main_df["Buy & Hold equity"] = to_series(equity_bh)
                main_df[f"MA ({short_window}/{long_window}) equity"] = to_series(equity_ma)

            st.line_chart(main_df, width="stretch")

            # -------- Equity curves chart --------
            st.write("### Strategy equity curves")

            if equity_ma is None:
                equity_df = to_series(equity_bh).to_frame(name="Buy & Hold")
            else:
                equity_df = pd.concat(
                    [
                        to_series(equity_bh).rename("Buy & Hold"),
                        to_series(equity_ma).rename(f"MA ({short_window}/{long_window})"),
                    ],
                    axis=1,
                )

            st.line_chart(equity_df, width="stretch")

            # -------- Performance metrics --------
            st.write("### Performance metrics")

            metrics_bh = compute_metrics(equity_bh, rets_bh)

            col_left, col_right = st.columns(2)

            # Buy & Hold metrics
            with col_left:
                st.write("#### Buy & Hold")
                m1, m2, m3 = st.columns(3)
                m4, m5 = st.columns(2)

                m1.metric(
                    "Total return",
                    f"{metrics_bh['total_return'] * 100:.2f} %",
                    help="Overall gain over the whole period.",
                )
                m2.metric(
                    "Ann. return",
                    f"{metrics_bh['annual_return'] * 100:.2f} %",
                    help="Average yearly growth rate.",
                )
                m3.metric(
                    "Ann. vol",
                    f"{metrics_bh['annual_vol'] * 100:.2f} %",
                    help="Yearly volatility (risk level).",
                )
                m4.metric(
                    "Sharpe",
                    f"{metrics_bh['sharpe']:.2f}",
                    help="Return divided by risk (higher is better).",
                )
                m5.metric(
                    "Max DD",
                    f"{metrics_bh['max_drawdown'] * 100:.2f} %",
                    help="Worst peak-to-trough loss over the period.",
                )

            # MA crossover metrics (if available)
            if equity_ma is not None and rets_ma is not None:
                metrics_ma = compute_metrics(equity_ma, rets_ma)

                with col_right:
                    st.write(f"#### MA crossover ({short_window}/{long_window})")
                    m1, m2, m3 = st.columns(3)
                    m4, m5 = st.columns(2)

                    m1.metric(
                        "Total return",
                        f"{metrics_ma['total_return'] * 100:.2f} %",
                        help="Overall gain over the whole period.",
                    )
                    m2.metric(
                        "Ann. return",
                        f"{metrics_ma['annual_return'] * 100:.2f} %",
                        help="Average yearly growth rate.",
                    )
                    m3.metric(
                        "Ann. vol",
                        f"{metrics_ma['annual_vol'] * 100:.2f} %",
                        help="Yearly volatility (risk level).",
                    )
                    m4.metric(
                        "Sharpe",
                        f"{metrics_ma['sharpe']:.2f}",
                        help="Return divided by risk (higher is better).",
                    )
                    m5.metric(
                        "Max DD",
                        f"{metrics_ma['max_drawdown'] * 100:.2f} %",
                        help="Worst peak-to-trough loss over the period.",
                    )

            # -------- Quick comparison + table --------
            if metrics_ma is not None:
                st.write("### Quick comparison of strategies")

                if metrics_ma["sharpe"] > metrics_bh["sharpe"]:
                    best_sharpe = "MA crossover"
                else:
                    best_sharpe = "Buy & Hold"

                if metrics_ma["total_return"] > metrics_bh["total_return"]:
                    best_return = "MA crossover"
                else:
                    best_return = "Buy & Hold"

                st.write(
                    f"- **Highest Sharpe ratio:** {best_sharpe} "
                    f"(BH = {metrics_bh['sharpe']:.2f}, "
                    f"MA = {metrics_ma['sharpe']:.2f})"
                )
                st.write(
                    f"- **Highest total return:** {best_return} "
                    f"(BH = {metrics_bh['total_return'] * 100:.2f} %, "
                    f"MA = {metrics_ma['total_return'] * 100:.2f} %)"
                )

                summary_df = pd.DataFrame(
                    {
                        "Total return (%)": [
                            metrics_bh["total_return"] * 100,
                            metrics_ma["total_return"] * 100,
                        ],
                        "Ann. return (%)": [
                            metrics_bh["annual_return"] * 100,
                            metrics_ma["annual_return"] * 100,
                        ],
                        "Ann. vol (%)": [
                            metrics_bh["annual_vol"] * 100,
                            metrics_ma["annual_vol"] * 100,
                        ],
                        "Sharpe": [
                            metrics_bh["sharpe"],
                            metrics_ma["sharpe"],
                        ],
                        "Max DD (%)": [
                            metrics_bh["max_drawdown"] * 100,
                            metrics_ma["max_drawdown"] * 100,
                        ],
                    },
                    index=["Buy & Hold", f"MA ({short_window}/{long_window})"],
                )

                st.write("#### Metrics summary table")
                st.dataframe(summary_df.round(2), width="stretch")

            # -------- Bonus: prediction model --------
            with st.expander("Bonus: simple prediction model (linear trend)"):
                show_forecast = st.checkbox(
                    "Show forecast for the selected horizon",
                    value=False,
                    key="show_forecast",
                )
                if show_forecast:
                    horizon = st.slider(
                        "Forecast horizon (number of periods)",
                        min_value=5,
                        max_value=60,
                        value=20,
                    )
                    try:
                        forecast_df = forecast_linear_trend(
                            close, horizon=horizon, freq_label=freq_label
                        )
                        st.line_chart(forecast_df, width="stretch")
                        st.caption(
                            "Forecast based on a simple linear regression of price vs time, "
                            "with an approximate 95% confidence band. "
                            "This is a very basic model, mainly used for illustration."
                        )
                    except Exception as e:
                        st.error(f"Error while computing forecast: {e}")

            # -------- Explanations --------
            with st.expander("Strategy explanations"):
                st.markdown(
                    """
    ### Buy & Hold (simple benchmark)
    - You buy the asset once at the beginning and keep it until the end.
    - The equity curve shows how 1 invested at the start evolves over time.
    - This is a very simple **passive strategy**, used as a benchmark.

    ### Moving Average Crossover (trend following)
    - We compute two moving averages on the closing price:
    - a **short** moving average (reacts faster to new moves),
    - a **long** moving average (smoother, slower).
    - When the short MA is **above** the long MA → we are invested (position = 1).
    - When the short MA is **below** the long MA → we are out of the market (position = 0).
    - Idea: try to
    - **follow upward trends**,
    - and stay out during strong downtrends.

    ### Simple prediction model (bonus)
    - We fit a **linear trend** of the price over time (straight line).
    - We extend this line into the future for a few periods (chosen by the user).
    - The 95% confidence band is built from the typical size of past errors.
    - This model is **very basic** and should not be used for real trading,
    but it shows how we can connect time series and prediction.
                    """
                )

            # -------- Raw data preview --------
            st.write("### Raw data (last rows)")
            st.dataframe(data.tail(), width="stretch")

    st.markdown(
        """
        <div style="font-size:11px; color:#999; text-align:right; margin-top:10px;">
        Lou-anne Peillon — Single Asset Module
        </div>
        """, 
        unsafe_allow_html=True
    )
