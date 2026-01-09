MULTI-ASSET PORTFOLIO DASHBOARD  
ESILV Project — Python, Git & Linux for Finance  
(Full project: Single Asset + Multi-Asset)

This project was developed as part of the Python, Git & Linux for Finance course.
The objective is to build an interactive financial platform capable of:

- retrieving real-time financial data
- displaying interactive dashboards
- running quantitative trading strategies
- simulating a multi-asset portfolio
- generating automated daily reports 
- running 24/7 on a Linux virtual machine (Oracle VM)

The project is carried out in pairs, with two distinct modules:
- Quant A — Single Asset Analysis
- Quant B — Multi-Asset Portfolio Analysis

Both modules are integrated into a single Streamlit application.

URL : http://141.253.119.77:8501/

--------------------------------------------------------------------

1) SINGLE ASSET ANALYSIS MODULE (Quant A)  
Folder: /app/single_asset/

Features:
- Analysis of a single asset (AAPL, CAC40, EUR/USD…)
- Dynamic price retrieval (API, yfinance, web scraping)
- Backtesting of at least two strategies
- Metrics: Sharpe ratio, Max Drawdown, Annualized Volatility, Cumulative Returns
- Interactive charts: asset price + strategy performance
- Adjustable parameters in the UI
- Bonus: predictive model (ARIMA, Regression, ML)

--------------------------------------------------------------------

2) MULTI-ASSET PORTFOLIO MODULE (Quant B)  
Folder: /app/portfolio/

Features:
- Multi-asset data loading (at least 3 assets) via yfinance or API
- Daily returns calculation
- Portfolio construction:
  * Equal-weight
  * Custom weights
  * Simple rebalancing
- Performance metrics:
  * Cumulative return
  * Annualized volatility
  * Sharpe ratio
  * Correlation matrix
- Interactive visualizations:
  * Individual asset prices
  * Portfolio cumulative value
  * Asset vs. portfolio comparison
  * Plotly interactive charts

--------------------------------------------------------------------

STREAMLIT APPLICATION

The final dashboard includes:
- A sidebar navigation menu
- A Single Asset page
- A Multi-Asset Portfolio page
- Automatic data refresh every 5 minutes (UTC)
- Robust error handling for API failures

Local execution:
streamlit run app/main.py

--------------------------------------------------------------------

LINUX DEPLOYMENT & CRON AUTOMATION

The project must be deployed on a Linux machine with:
- A Streamlit app running 24/7
- Automatic data updates
- A daily report (generated via cron at 20:00)
- Reports stored in /reports/
- Cron script included in the repository

--------------------------------------------------------------------

TECHNOLOGIES USED

- Python 3.9  
- Streamlit  
- Pandas / NumPy  
- Plotly  
- SciPy  
- yfinance
- altair 
- jinja2
- streamlit-autorefresh 
- pandas-datareader   
- Git / GitHub  
- Linux (VM Oracle, pm2, cron)

--------------------------------------------------------------------

AUTHORS

Lou-anne Peillon — Single Asset Module (Quant A)
Pierre Nieto — Multi-Asset Portfolio Module (Quant B)

--------------------------------------------------------------------

FINAL OBJECTIVE

A robust, professional, interactive financial application deployed on Linux
and running 24/7, capable of analyzing both a single asset and a complete
multi-asset portfolio.
