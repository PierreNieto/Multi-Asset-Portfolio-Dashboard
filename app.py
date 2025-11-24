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
        auto_adjust=False,  # on fixe le comportement pour éviter le warning
    )
    return data


def ensure_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    """S'assure qu'on travaille avec une Series (pas un DataFrame)."""
    if isinstance(x, pd.DataFrame):
        # on prend la première colonne si DataFrame
        return x.iloc[:, 0]
    return x


def backtest_buy_and_hold(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Backtest Buy & Hold.
    - close : prix de clôture
    Retourne:
      - equity : valeur cumulée du portefeuille (normalisée à 1 au début)
      - returns : rendements journaliers
    """
    close = ensure_series(close).sort_index()

    equity = close / close.iloc[0]
    returns = equity.pct_change().dropna()
    return equity, returns


def backtest_ma_crossover(
    close: pd.Series,
    short_window: int,
    long_window: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Backtest stratégie de croisement de moyennes mobiles.
    - Position = 1 quand MA courte > MA longue, sinon 0.
    """
    close = ensure_series(close).sort_index()

    df = pd.DataFrame({"close": close})
    df["ma_short"] = df["close"].rolling(short_window).mean()
    df["ma_long"] = df["close"].rolling(long_window).mean()

    # On enlève les débuts où les moyennes ne sont pas définies
    df = df.dropna()
    if len(df) < 2:
        # Pas assez de points pour une vraie stratégie, on retourne quelque chose de trivial
        equity = pd.Series([1.0], index=df.index[:1])
        returns = equity.pct_change().dropna()
        return equity, returns

    signals = (df["ma_short"] > df["ma_long"]).astype(int)

    daily_returns = df["close"].pct_change().fillna(0)
    # On applique la position de la veille
    strategy_returns = daily_returns * signals.shift(1).fillna(0)

    equity = (1 + strategy_returns).cumprod()
    returns = equity.pct_change().dropna()

    return equity, returns


def compute_metrics(equity: pd.Series, returns: pd.Series) -> dict:
    """Calcule quelques métriques de performance."""
    equity = ensure_series(equity)

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

st.write(
    "Analyse d'un seul actif avec deux stratégies : "
    "Buy & Hold et croisement de moyennes mobiles."
)

# --- Contrôles utilisateur ---
st.sidebar.header("Paramètres de l'actif")

DEFAULT_TICKER = "BTC-USD"  # tu pourras fixer ton actif officiel ici
ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", value=DEFAULT_TICKER)

today = dt.date.today()
default_start = today - dt.timedelta(days=365)

start_date = st.sidebar.date_input("Date de début", value=default_start)
end_date = st.sidebar.date_input("Date de fin", value=today)

st.sidebar.markdown("---")
st.sidebar.header("Stratégie MA crossover")

short_window = st.sidebar.slider("Moyenne mobile courte", 5, 50, 20)
long_window = st.sidebar.slider("Moyenne mobile longue", 20, 200, 100)

if short_window >= long_window:
    st.sidebar.error("La fenêtre courte doit être strictement plus petite que la longue.")

if start_date >= end_date:
    st.error("La date de début doit être avant la date de fin.")
else:
    data = get_price_data(ticker, start_date, end_date)

    if data.empty:
        st.warning("Aucune donnée trouvée pour ce ticker / ces dates.")
    else:
        if "Close" not in data.columns:
            st.error("La colonne 'Close' est introuvable dans les données téléchargées.")
            st.stop()

        close = data["Close"].dropna()

        if len(close) < 2:
            st.warning("Pas assez de données pour faire un backtest (au moins 2 points).")
            st.stop()

        # -------- Prix ----------
        st.write(f"### Prix de {ticker}")
        st.line_chart(close, width="stretch")

        # On vérifie qu'on a assez de points pour la stratégie MA
        can_run_ma = len(close) >= long_window + 5

        # -------- Backtests --------
        st.write("### Stratégies : Buy & Hold vs MA crossover")

        equity_bh, returns_bh = backtest_buy_and_hold(close)

        equity_ma = None
        returns_ma = None
        metrics_ma = None

        if not can_run_ma or short_window >= long_window:
            st.warning(
                "Pas assez de données pour la stratégie MA crossover "
                "(ou paramètres de fenêtres invalides). "
                "Seule la stratégie Buy & Hold est affichée."
            )
            equity_df = equity_bh.to_frame(name="Buy & Hold")
        else:
            equity_ma, returns_ma = backtest_ma_crossover(
                close,
                short_window=short_window,
                long_window=long_window,
            )

            # DataFrame avec les deux courbes
            equity_df = pd.concat(
                [
                    equity_bh.rename("Buy & Hold"),
                    equity_ma.rename(f"MA {short_window}/{long_window}"),
                ],
                axis=1,
            )

        st.line_chart(equity_df, width="stretch")

        # -------- Metrics --------
        st.write("### Performance des stratégies")

        metrics_bh = compute_metrics(equity_bh, returns_bh)

        col_left, col_right = st.columns(2)

        with col_left:
            st.write("#### Buy & Hold")
            c1, c2, c3 = st.columns(3)
            c4, c5 = st.columns(2)

            c1.metric(
                "Rendement total",
                f"{metrics_bh['total_return'] * 100:.2f} %",
            )
            c2.metric(
                "Rendement annualisé",
                f"{metrics_bh['annual_return'] * 100:.2f} %",
            )
            c3.metric(
                "Volatilité annualisée",
                f"{metrics_bh['annual_vol'] * 100:.2f} %",
            )
            c4.metric(
                "Sharpe (rf=0)",
                f"{metrics_bh['sharpe']:.2f}",
            )
            c5.metric(
                "Max drawdown",
                f"{metrics_bh['max_drawdown'] * 100:.2f} %",
            )

        with col_right:
            if equity_ma is None or returns_ma is None:
                st.write("#### MA crossover")
                st.info("Stratégie MA crossover non disponible avec les paramètres actuels.")
            else:
                metrics_ma = compute_metrics(equity_ma, returns_ma)

                st.write(f"#### MA {short_window}/{long_window}")
                c1, c2, c3 = st.columns(3)
                c4, c5 = st.columns(2)

                c1.metric(
                    "Rendement total",
                    f"{metrics_ma['total_return'] * 100:.2f} %",
                )
                c2.metric(
                    "Rendement annualisé",
                    f"{metrics_ma['annual_return'] * 100:.2f} %",
                )
                c3.metric(
                    "Volatilité annualisée",
                    f"{metrics_ma['annual_vol'] * 100:.2f} %",
                )
                c4.metric(
                    "Sharpe (rf=0)",
                    f"{metrics_ma['sharpe']:.2f}",
                )
                c5.metric(
                    "Max drawdown",
                    f"{metrics_ma['max_drawdown'] * 100:.2f} %",
                )

        # -------- Comparaison rapide + tableau --------
        if metrics_ma is not None:
            st.write("### Comparaison rapide des stratégies")

            better_sharpe = (
                "MA crossover"
                if metrics_ma["sharpe"] > metrics_bh["sharpe"]
                else "Buy & Hold"
            )
            better_return = (
                "MA crossover"
                if metrics_ma["total_return"] > metrics_bh["total_return"]
                else "Buy & Hold"
            )

            st.write(
                f"- **Sharpe le plus élevé :** {better_sharpe} "
                f"(BH = {metrics_bh['sharpe']:.2f}, "
                f"MA = {metrics_ma['sharpe']:.2f})"
            )
            st.write(
                f"- **Meilleur rendement total :** {better_return} "
                f"(BH = {metrics_bh['total_return'] * 100:.2f} %, "
                f"MA = {metrics_ma['total_return'] * 100:.2f} %)"
            )

            # Tableau récapitulatif des métriques
            summary_df = pd.DataFrame(
                {
                    "Rendement total (%)": [
                        metrics_bh["total_return"] * 100,
                        metrics_ma["total_return"] * 100,
                    ],
                    "Rendement annualisé (%)": [
                        metrics_bh["annual_return"] * 100,
                        metrics_ma["annual_return"] * 100,
                    ],
                    "Volatilité annualisée (%)": [
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
                index=["Buy & Hold", f"MA {short_window}/{long_window}"],
            )

            st.write("#### Tableau récapitulatif des métriques")
            st.dataframe(summary_df.round(2), width="stretch")

        # -------- Explications texte --------
        with st.expander("Explications sur les stratégies"):
            st.markdown(
                """
### Stratégie Buy & Hold
- On achète l'actif au début de la période et on le conserve jusqu'à la fin.
- La courbe d'équity représente la valeur du portefeuille normalisée à 1 au début.
- Cette stratégie sert de **référence passive** pour comparer les stratégies actives.

### Stratégie MA crossover
- On calcule deux moyennes mobiles sur le prix de clôture :
  - une **moyenne courte** (réagit vite),
  - une **moyenne longue** (plus lisse).
- Quand la moyenne courte passe **au-dessus** de la longue → on est investi (position = 1).
- Quand elle passe **en-dessous** → on sort du marché (position = 0).
- Cette stratégie cherche à :
  - **capturer les tendances** haussières,
  - tout en réduisant l'exposition pendant les phases baissières.
                """
            )

        st.write("### Aperçu des données brutes")
        st.dataframe(data.tail(), width="stretch")
