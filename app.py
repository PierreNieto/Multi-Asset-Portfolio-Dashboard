import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ---------------------------
# Data loading and utilities
# ---------------------------

def get_price_data(ticker, start_date, end_date):
    """
    Download price data from Yahoo Finance.
    """
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,  # keep raw prices (no dividend / split adjust)
        )
    except Exception:
        # if yfinance fails (bad ticker, network issue, ...)
        return pd.DataFrame()
    return data


def to_series(obj):
    """
    Make sure we are working with a pandas Series.
    If it's a DataFrame, take the first column.
    If it's a scalar, build a 1-element Series.
    """
    if isinstance(obj, pd.DataFrame):
        # take the first column
        return obj.iloc[:, 0]
    if isinstance(obj, pd.Series):
        return obj
    # fallback: scalar -> series of length 1
    return pd.Series([obj])


# ---------------------------
# Strategies
# ---------------------------

def backtest_buy_and_hold(close):
    """
    Very simple buy & hold strategy:
    - invest 100% at the beginning, hold until the end
    Returns:
      equity: normalized portfolio value
      rets: period returns
    """
    close = to_series(close).sort_index()
    equity = close / close.iloc[0]
    rets = equity.pct_change().dropna()
    return equity, rets


def backtest_ma_crossover(close, short_window, long_window):
    """
    Moving average crossover strategy:
    - long when short MA > long MA
    - flat otherwise
    """
    close = to_series(close).sort_index()
    df = pd.DataFrame({"close": close})
    df["ma_short"] = df["close"].rolling(short_window).mean()
    df["ma_long"] = df["close"].rolling(long_window).mean()

    # drop first rows where MAs are not defined
    df = df.dropna()
    if len(df) < 2:
        # not enough points for a real backtest
        eq = pd.Series([1.0], index=df.index[:1])
        rets = eq.pct_change().dropna()
        return eq, rets

    # 1 when short MA > long MA, else 0
    df["position"] = (df["ma_short"] > df["ma_long"]).astype(int)

    # returns of the asset
    asset_rets = df["close"].pct_change().fillna(0.0)

    # strategy returns: position of previous period * asset return
    strat_rets = asset_rets * df["position"].shift(1).fillna(0.0)

    equity = (1.0 + strat_rets).cumprod()
    rets = equity.pct_change().dropna()
    return equity, rets


# ---------------------------
# Performance metrics
# ---------------------------

def compute_metrics(equity, rets):
    """
    Compute basic performance metrics:
    - total return
    - annualized return (using 252 periods per year)
    - annualized volatility
    - Sharpe ratio (rf = 0)
    - max drawdown
    """
    eq = to_series(equity)
    r = to_series(rets)

    total_ret = eq.iloc[-1] - 1.0

    if len(eq.index) > 1:
        nb_days = (eq.index[-1] - eq.index[0]).days
    else:
        nb_days = 0

    if nb_days <= 0 or r.empty:
        ann_ret = np.nan
        ann_vol = np.nan
        sharpe = np.nan
    else:
        mean_per_period = r.mean()
        # we assume 252 trading days even if we resample weekly/monthly
        ann_ret = (1.0 + mean_per_period) ** 252 - 1.0
        vol_per_period = r.std()
        ann_vol = vol_per_period * np.sqrt(252)
        if ann_vol > 0:
            sharpe = ann_ret / ann_vol
        else:
            sharpe = np.nan

    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    max_dd = drawdown.min()

    return {
        "total_return": total_ret,
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


# ---------------------------
# Simple prediction model
# ---------------------------

def forecast_linear_trend(close, horizon, freq_label):
    """
    Very simple prediction model:
    - linear regression of price vs time index
    - extrapolate on future periods
    - build a (rough) 95% confidence band using residual std
    """
    close = to_series(close).dropna().sort_index()
    n = len(close)
    if n < 5:
        raise ValueError("Not enough data to fit a model (need at least 5 points).")

    # time axis: 0, 1, ..., n-1
    t = np.arange(n)
    # matrix for least squares: [t, 1]
    A = np.vstack([t, np.ones(n)]).T

    # least squares fit: price ≈ m * t + c
    m, c = np.linalg.lstsq(A, close.values, rcond=None)[0]

    # in-sample predictions and residuals
    pred_in = m * t + c
    residuals = close.values - pred_in
    sigma = residuals.std(ddof=1)

    # future time points
    t_future = np.arange(n, n + horizon)
    pred_future = m * t_future + c

    # map frequency label to pandas frequency string
    if freq_label == "Daily":
        freq = "D"
    elif freq_label == "Weekly":
        freq = "W"
    else:
        freq = "M"

    # build future dates
    last_date = close.index[-1]
    offset = pd.tseries.frequencies.to_offset(freq)
    start_future = last_date + offset
    future_index = pd.date_range(start=start_future, periods=horizon, freq=freq)

    hist = close.rename("Historical price")
    forecast = pd.Series(pred_future, index=future_index, name="Forecast")
    lower = (forecast - 1.96 * sigma).rename("Lower CI (95%)")
    upper = (forecast + 1.96 * sigma).rename("Upper CI (95%)")

    # combine in one DataFrame (historical + forecast + bands)
    df_hist = pd.DataFrame({"Historical price": hist})
    df_forecast = pd.DataFrame(
        {
            "Forecast": forecast,
            "Lower CI (95%)": lower,
            "Upper CI (95%)": upper,
        }
    )
    combined = pd.concat([df_hist, df_forecast], axis=1)
    return combined


# ---------------------------
# Streamlit app (Quant A)
# ---------------------------

st.title("Multi-Asset Portfolio Dashboard")
st.subheader("Quant A - Single Asset Module")

st.write(
    "This page analyses **one asset** with two trading strategies "
    "(Buy & Hold and Moving Average Crossover) plus a simple prediction bonus."
)

# -------- Sidebar controls --------
st.sidebar.header("Asset settings")

# Official single asset for the project
default_ticker = "BTC-USD"
ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", value=default_ticker)
st.sidebar.caption(
    "Main asset studied: BTC-USD (Bitcoin / USD). "
    "You can also try other valid Yahoo tickers, e.g. AAPL, MSFT, EURUSD=X."
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
    "Short moving average window",
    min_value=short_min,
    max_value=short_max,
    value=short_default,
)
long_window = st.sidebar.slider(
    "Long moving average window",
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

            m1.metric("Total return", f"{metrics_bh['total_return'] * 100:.2f} %")
            m2.metric("Annualized return", f"{metrics_bh['annual_return'] * 100:.2f} %")
            m3.metric("Annualized vol", f"{metrics_bh['annual_vol'] * 100:.2f} %")
            m4.metric("Sharpe (rf = 0)", f"{metrics_bh['sharpe']:.2f}")
            m5.metric("Max drawdown", f"{metrics_bh['max_drawdown'] * 100:.2f} %")

        # MA crossover metrics (if available)
        if equity_ma is not None and rets_ma is not None:
            metrics_ma = compute_metrics(equity_ma, rets_ma)

            with col_right:
                st.write(f"#### MA crossover ({short_window}/{long_window})")
                m1, m2, m3 = st.columns(3)
                m4, m5 = st.columns(2)

                m1.metric("Total return", f"{metrics_ma['total_return'] * 100:.2f} %")
                m2.metric(
                    "Annualized return",
                    f"{metrics_ma['annual_return'] * 100:.2f} %",
                )
                m3.metric(
                    "Annualized vol",
                    f"{metrics_ma['annual_vol'] * 100:.2f} %",
                )
                m4.metric("Sharpe (rf = 0)", f"{metrics_ma['sharpe']:.2f}")
                m5.metric(
                    "Max drawdown",
                    f"{metrics_ma['max_drawdown'] * 100:.2f} %",
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
                    "Annualized return (%)": [
                        metrics_bh["annual_return"] * 100,
                        metrics_ma["annual_return"] * 100,
                    ],
                    "Annualized vol (%)": [
                        metrics_bh["annual_vol"] * 100,
                        metrics_ma["annual_vol"] * 100,
                    ],
                    "Sharpe": [
                        metrics_bh["sharpe"],
                        metrics_ma["sharpe"],
                    ],
                    "Max drawdown (%)": [
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
                        "with an approximate 95% confidence band."
                    )
                except Exception as e:
                    st.error(f"Error while computing forecast: {e}")

        # -------- Explanations --------
        with st.expander("Strategy explanations"):
            st.markdown(
                """
### Buy & Hold
- Invest once at the beginning and keep the position until the end.
- The equity curve shows the portfolio value normalized to 1 at the start.
- This is a **passive benchmark** to compare active strategies.

### Moving Average Crossover
- We compute two moving averages on the closing price:
  - a **short** moving average (reacts faster),
  - a **long** moving average (smoother).
- When the short MA is **above** the long MA → we are invested (position = 1).
- When the short MA is **below** the long MA → we are out of the market (position = 0).
- The goal is to:
  - **capture upward trends**,
  - and reduce exposure during downtrends.

### Simple prediction model (bonus)
- We fit a **linear trend** of the price with respect to time.
- We extend this trend into the future on a user-chosen horizon.
- The 95% confidence band is built from the standard deviation of the residuals.
                """
            )

        # -------- Raw data preview --------
        st.write("### Raw data (last rows)")
        st.dataframe(data.tail(), width="stretch")
