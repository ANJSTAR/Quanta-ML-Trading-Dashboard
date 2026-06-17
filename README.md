# Quant ML Trading Dashboard

A clean, end-to-end machine learning project that predicts the next-day price movement of major Indian equities (RELIANCE, TCS, INFY, HDFCBANK) using structural market features and a Random Forest classification model.

## Features Built
* **Data Pipeline:** Fetches 5 years of historical data dynamically via Yahoo Finance (`yfinance`).
* **Feature Engineering:** Calculated customized technical indicators like Relative Strength Index (RSI), Moving Average Ratios, and Bollinger Band positions along with fundamental valuation metrics ($P/E$ and $P/B$ proxies).
* **Predictive Modeling:** Enforced strict sequential splitting on a Random Forest Classifier to avoid data leakage, outputting explicit direction probabilities.
* **Paper Trading Ledger:** Integrated a custom performance audit trail that logs trading decisions into a local CSV matrix and computes dynamic live Profit & Loss (P&L) points.

## Tech Stack
* Python
* Streamlit (Dashboard UI)
* Scikit-Learn (Machine Learning)
* Pandas & NumPy (Data Processing)
* YFinance (Market Data API)

## How to Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Launch the server: `streamlit run app.py`
