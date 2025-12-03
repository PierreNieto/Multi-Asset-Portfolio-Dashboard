# Multi-Asset-Portfolio-Dashboard

Small Streamlit app for playing with financial time series.

The project is split in two parts:

- **Quant A – Single Asset module (my part)**
- **Quant B – Portfolio module** (handled by my teammate)

For now, this README mainly documents **Quant A**, since this is the part I implemented.

---

## 1. Quant A – Single Asset FX Module

**Main asset studied:** `EURUSD=X` (EUR/USD FX rate on Yahoo Finance).

The idea is to focus on **one asset at a time** and compare simple trading rules.

In the app, the user can:

- choose a **ticker** (default: `EURUSD=X`),
- pick a **date range**,
- choose a **data frequency**: Daily / Weekly / Monthly,
- set the parameters of a **Moving Average crossover** strategy,
- compare it to a simple **Buy & Hold** strategy.

The main page shows:

- the raw price of the asset,
- a chart with **normalized price vs strategy equity**,
- a second chart with **equity curves** only (one line = one strategy).

---

## 2. Strategies

### Buy & Hold (benchmark)

- Buy the asset at the beginning of the period,
- keep the position until the end,
- the equity curve is just the price normalized by its initial value.

This serves as a very simple **benchmark**.

### Moving Average Crossover (trend following)

- We compute two moving averages on the closing price:
  - a **short** moving average (more reactive),
  - a **long** moving average (smoother).
- When `MA_short > MA_long` → we are **in the market** (position = 1).
- When `MA_short <= MA_long` → we are **out of the market** (position = 0).

The short and long windows are controlled by sliders in the sidebar and are
adapted to the chosen frequency (Daily / Weekly / Monthly).

---

## 3. Performance metrics

For each strategy, the app computes and displays:

- **Total return** – overall gain over the whole period.
- **Annualized return** – average yearly growth (assuming 252 trading days).
- **Annualized volatility** – yearly standard deviation of returns (risk).
- **Sharpe ratio** – risk-adjusted performance (return / risk, rf = 0).
- **Max drawdown** – worst peak-to-trough loss.

A short text summary tells which strategy has:

- the **highest Sharpe ratio**, and
- the **highest total return** for the selected period.

A small table with all metrics side by side is also shown.

---

## 4. Bonus: Simple prediction model

As an optional bonus, Quant A also includes a tiny prediction feature:

- we fit a **linear regression** of price vs time (straight line),
- we extend this trend into the future over a user-chosen horizon (5–60 periods),
- we build a rough **95% confidence band** based on past residuals.

The chart displays:

- historical prices,
- the forecast line,
- the confidence interval.

This model is **very basic** and is only here to illustrate how to connect
time series data with a simple prediction method.  
It is **not** meant to be used for real trading.

---

## 5. How to run the app locally (Quant A)

### 5.1. Create and activate the environment

Using conda:

```bash
conda create -n quantproj python=3.11
conda activate quantproj
