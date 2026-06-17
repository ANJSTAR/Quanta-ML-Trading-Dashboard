import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Quant Dashboard", layout="wide")

st.title("🚀 Quantamental Machine Learning Trading Dashboard")
st.markdown("Predictive stock movement pipeline using Random Forest Classifier")

st.sidebar.header("Settings")
ticker = st.sidebar.selectbox("Select Stock", ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"])
threshold = st.sidebar.slider("Confidence Threshold (%)", 50, 60, 52) / 100

st.sidebar.write(f"Active Filter: Strategy trades only at >= {threshold*100:.1f}% confidence")

@st.cache_data
def prepare_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="5y")
    
    info = stock.info
    bv = info.get('bookValue', 1)
    eps = info.get('trailingEps', 1)
    
    df['Daily_P_B'] = df['Close'] / bv
    df['Daily_P_E'] = df['Close'] / eps
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_Ratio'] = df['MA_20'] / df['MA_50']
    df['Daily_Return'] = df['Close'].pct_change()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
    df['Lag_Return_1'] = df['Daily_Return'].shift(1)
    df['Lag_Return_2'] = df['Daily_Return'].shift(2)
    
    df['Tomorrow_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow_Close'].values.flatten() > df['Close'].values.flatten()).astype(int)
    df.dropna(inplace=True)
    
    df['Std_Dev_20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA_20'] + (2 * df['Std_Dev_20'])
    df['Lower_Band'] = df['MA_20'] - (2 * df['Std_Dev_20'])
    df['BB_Position'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    df.dropna(inplace=True)
    
    return df

data = prepare_data(ticker)
live_price = data['Close'].iloc[-1]

features = ['MA_Ratio', 'Daily_Return', 'RSI', 'Daily_P_B', 'Daily_P_E', 'Volume_Ratio', 'Lag_Return_1', 'Lag_Return_2', 'BB_Position']
X = data[features]
y = data['Target']

model = RandomForestClassifier(n_estimators=200, min_samples_split=80, min_samples_leaf=25, class_weight='balanced', random_state=42)
model.fit(X.iloc[:-1], y.iloc[:-1])

last_row = X.tail(1)
proba = model.predict_proba(last_row)

st.subheader(f"🔮 Next Session Signal: {ticker}")
up_prob = proba[0][1]

if up_prob >= threshold:
    signal = "BUY"
    conf = up_prob
elif up_prob <= (1 - threshold):
    signal = "SHORT"
    conf = (1 - up_prob)
else:
    signal = "NO TRADE"
    conf = up_prob

col1, col2 = st.columns(2)
with col1:
    if signal == "BUY":
        st.success("### SYSTEM SIGNAL: 🚀 LONG POSITION")
        st.metric(label="Confidence", value=f"{conf*100:.2f}%")
    elif signal == "SHORT":
        st.error("### SYSTEM SIGNAL: 📉 SHORT POSITION")
        st.metric(label="Confidence", value=f"{conf*100:.2f}%")
    else:
        st.warning("### SYSTEM SIGNAL: 💤 NO TRADE")
        st.metric(label="UP Probability", value=f"{conf*100:.2f}%")

with col2:
    st.metric(label=f"Current Price ({ticker})", value=f"₹{live_price:.2f}")


st.subheader("📋 Live Trading Ledger")
log_file = "trading_logs.csv"

if os.path.isfile(log_file):
    logs = pd.read_csv(log_file)
    ticker_logs = logs[logs['Ticker'] == ticker].copy()
    
    if not ticker_logs.empty:
        def get_pnl(row):
            try:
                p_entry = float(row['Current_Price'])
                if row['Signal'] == 'BUY':
                    return live_price - p_entry
                elif row['Signal'] == 'SHORT':
                    return p_entry - live_price
                return 0.0
            except:
                return 0.0
                
        ticker_logs['Live_P&L'] = ticker_logs.apply(get_pnl, axis=1)
        total_pnl = ticker_logs['Live_P&L'].sum()
        
        pnl_col, history_col = st.columns([1, 3])
        with pnl_col:
            if total_pnl > 0:
                st.metric(label="🟢 Strategy P&L", value=f"+₹{total_pnl:.2f}")
            elif total_pnl < 0:
                st.metric(label="🔴 Strategy P&L", value=f"-₹{abs(total_pnl):.2f}")
            else:
                st.metric(label="⚪ Strategy P&L", value=f"₹{total_pnl:.2f}")
        
        with history_col:
            st.write("### Recent Logs")
            st.dataframe(ticker_logs.tail(5), width='stretch')
    else:
        st.info(f"No trades logged for {ticker}.")
else:
    st.info("Ledger file empty. Save a signal below to begin.")

if st.button("📝 Log Today's Position"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([{
        "Timestamp": now,
        "Ticker": ticker,
        "Signal": signal,
        "Confidence": f"{conf*100:.2f}%",
        "Current_Price": round(live_price, 2)
    }])
    
    if not os.path.isfile(log_file):
        new_entry.to_csv(log_file, index=False)
    else:
        new_log = pd.DataFrame(new_entry)
        new_log.to_csv(log_file, mode='a', header=False, index=False)
    st.toast("Position logged successfully!", icon="✅")
    st.rerun()

st.subheader(f"📊 Recent Price Action ({ticker})")
st.line_chart(data['Close'].tail(100))
