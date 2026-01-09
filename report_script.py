import os
import datetime as dt
import pandas as pd
import yfinance as yf
import numpy as np

# --- Configuration ---
# Absolute path 
BASE_DIR = "/home/ubuntu/Multi-Asset-Portfolio-Dashboard"
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# Assets selected from your dashboard files
SINGLE_ASSET_TICKER = "EURUSD=X" 
PORTFOLIO_TICKERS = ["AAPL", "GOOGL", "NVDA", "SPY", "GC=F", "BTC-USD"]

def fetch_market_data(tickers, days=30):
    """Fetch historical data for analysis."""
    try:
        data = yf.download(tickers, start=dt.date.today() - dt.timedelta(days=days), progress=False)
        return data['Close']
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def compute_metrics(prices):
    """Calculate financial metrics."""
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
        
    returns = prices.pct_change().dropna()
    
    # Core calculations
    last_price = prices.iloc[-1]
    daily_ret = (prices.iloc[-1] / prices.iloc[-2] - 1) if len(prices) > 1 else 0
    weekly_ret = (prices.iloc[-1] / prices.iloc[-6] - 1) if len(prices) > 6 else 0
    
    # Annualized Volatility (20-day window)
    vol = returns.tail(20).std() * np.sqrt(252)
    
    return {
        "price": last_price,
        "daily_ret": daily_ret * 100,
        "weekly_ret": weekly_ret * 100,
        "vol": vol * 100
    }

def generate_report():
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    now = dt.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    file_date = now.strftime("%Y-%m-%d")
    
    # 1. Process Single Asset Focus (from single_asset_page.py)
    single_data = fetch_market_data([SINGLE_ASSET_TICKER])
    single_m = compute_metrics(single_data) if not single_data.empty else None

    # 2. Process Multi-Asset Portfolio (from portfolio_page.py)
    port_data = fetch_market_data(PORTFOLIO_TICKERS)
    port_results = {}
    if not port_data.empty:
        for t in PORTFOLIO_TICKERS:
            if t in port_data.columns:
                port_results[t] = compute_metrics(port_data[t])

    # 3. Generate report
    report_name = f"market_report_{file_date}.txt"
    report_path = os.path.join(REPORT_DIR, report_name)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*75 + "\n")
        f.write(f"{'QUANT-B FINANCIAL INTELLIGENCE UNIT':^75}\n")
        f.write(f"{'DAILY MULTI-ASSET PERFORMANCE REPORT':^75}\n")
        f.write("="*75 + "\n")
        f.write(f"Generated on : {timestamp}\n")
        f.write(f"System Status: ONLINE (Oracle Cloud Node)\n")
        f.write("="*75 + "\n\n")

        # SECTION I: SINGLE ASSET MONITORING
        f.write("I. MACRO FOCUS: SINGLE ASSET MONITORING\n")
        f.write("-" * 45 + "\n")
        if single_m:
            f.write(f"Asset Ticker      : {SINGLE_ASSET_TICKER}\n")
            f.write(f"Latest Price      : {single_m['price']:.4f}\n")
            f.write(f"24h Change        : {single_m['daily_ret']:+.2f}%\n")
            f.write(f"7d Performance    : {single_m['weekly_ret']:+.2f}%\n")
            sentiment = "BULLISH" if single_m['weekly_ret'] > 0 else "BEARISH"
            f.write(f"Technical Bias    : {sentiment}\n")
        f.write("\n")

        # SECTION II: PORTFOLIO PERFORMANCE
        f.write("II. MULTI-ASSET PORTFOLIO PERFORMANCE\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'Ticker':<12} | {'Price':>10} | {'1D %':>8} | {'7D %':>8} | {'Ann. Vol':>9}\n")
        f.write("-" * 65 + "\n")
        
        daily_rets = []
        for ticker, m in port_results.items():
            f.write(f"{ticker:<12} | {m['price']:>10.2f} | {m['daily_ret']:>7.2f}% | {m['weekly_ret']:>7.2f}% | {m['vol']:>8.2f}%\n")
            daily_rets.append(m['daily_ret'])
        
        f.write("-" * 65 + "\n")
        avg_ret = np.mean(daily_rets) if daily_rets else 0
        f.write(f"Aggregated Daily Portfolio Return: {avg_ret:+.2f}%\n\n")

        # SECTION III: RISK ANALYTICS
        f.write("III. RISK & VOLATILITY INDICATORS\n")
        f.write("-" * 45 + "\n")
        regime = "STABLE" if abs(avg_ret) < 1.2 else "VOLATILE"
        f.write(f"Market Regime     : {regime}\n")
        f.write(f"Portfolio Health  : ACTIVE\n")
        f.write(f"Storage Path      : {report_path}\n\n")

        f.write("="*75 + "\n")
        f.write(f"{'END OF DAILY MARKET SUMMARY':^75}\n")
        f.write("="*75 + "\n")

    print(f"Professional report successfully generated: {report_path}")

if __name__ == "__main__":
    generate_report()