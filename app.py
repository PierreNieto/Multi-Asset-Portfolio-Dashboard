import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ---------- Fonctions utiles ----------

def get_price_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Télécharge les prix depuis Yahoo Finance."""
    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
    )
    return data


def backtest_buy_and_hold(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Backtest Buy & Hold.
    - close : série des prix de clôture
    Retourne:
      - equity : valeur cumulée du portefeuille (normalisée à 1 au début)
      - returns : rendements journaliers
    """
    # Si jamais on reçoit un DataFrame, on prend juste la première colonne
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # On s'assure que c'est bien une Series triée par date
    close = close.sort_index()

    equity = close / close.iloc[0]
    returns = equity.pct_change().dropna()
    return equity, returns


def compute_metrics(equity: pd.Series, returns: pd.Series) -> dict:
    """Calcule quelques métriques de performance."""
    total_return = equity.iloc[-1] - 1

    # Nombre de jours entre le début et la fin
    nb_days = (equity.index[-1] - equity.index[0]).days
    if nb_days <= 0:
        annual_return = np.nan
    else:
        mean_daily = returns.mean()
        annual_return = (1 + mean_daily) ** 252 - 1

    vol_daily = returns.std()
    vol_annual = vol_daily * np.sqrt(252)

    if vol_annual > 0:
        sharpe = annual_return / vol_annual
    else:
        sharpe = np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": vol_annual,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


# ---------- Interface Streamlit ----------

st.title("Multi-Asset Portfolio Dashboard")
st.subheader("Module Quant A - Single Asset")

st.write("Ceci est ma toute première version de l'app 😎")

# --- Contrôles utilisateur ---
st.sidebar.header("Paramètres de l'actif")

DEFAULT_TICKER = "BTC-USD"  # tu pourras choisir ton actif officiel ici
ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", value=DEFAULT_TICKER)

today = dt.date.today()
default_start = today - dt.timedelta(days=365)

start_date = st.sidebar.date_input("Date de début", value=default_start)
end_date = st.sidebar.date_input("Date de fin", value=today)

if start_date >= end_date:
    st.error("La date de début doit être avant la date de fin.")
else:
    data = get_price_data(ticker, start_date, end_date)

    if data.empty:
        st.warning("Aucune donnée trouvée pour ce ticker / ces dates.")
    else:
        # On récupère bien la colonne Close, et on enlève les NaN
        if "Close" not in data.columns:
            st.error("La colonne 'Close' est introuvable dans les données téléchargées.")
            st.stop()

        close = data["Close"].dropna()

        if len(close) < 2:
            st.warning("Pas assez de données pour faire un backtest (au moins 2 points).")
            st.stop()

        # -------- Prix ----------
        st.write(f"### Prix de {ticker}")
        st.line_chart(close, use_container_width=True)

        # -------- Backtest Buy & Hold --------
        st.write("### Backtest : Buy & Hold")

        equity, returns = backtest_buy_and_hold(close)

        # Ici equity est garanti comme Series → to_frame fonctionne
        equity_df = equity.to_frame(name="Buy & Hold equity")

        st.line_chart(equity_df, use_container_width=True)

        # -------- Metrics --------
        metrics = compute_metrics(equity, returns)

        col1, col2, col3 = st.columns(3)
        col4, col5 = st.columns(2)

        col1.metric(
            "Rendement total",
            f"{metrics['total_return'] * 100:.2f} %",
        )
        col2.metric(
            "Rendement annualisé",
            f"{metrics['annual_return'] * 100:.2f} %",
        )
        col3.metric(
            "Volatilité annualisée",
            f"{metrics['annual_vol'] * 100:.2f} %",
        )
        col4.metric(
            "Sharpe (rf=0)",
            f"{metrics['sharpe']:.2f}",
        )
        col5.metric(
            "Max drawdown",
            f"{metrics['max_drawdown'] * 100:.2f} %",
        )

        st.write("Aperçu des données :")
        st.dataframe(data.tail(), use_container_width=True)
