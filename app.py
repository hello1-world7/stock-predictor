import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# 1. UI CONFIGURATION
st.set_page_config(page_title="Institutional Stock AI", layout="wide")

# Custom CSS for dark-mode premium feel
st.markdown("""
    <style>
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3em; background-color: #21262d; color: white; border: 1px solid #30363d; }
    div.stButton > button:hover { border-color: #58a6ff; color: #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

st.title("Institutional Stock Predictor")

# 2. QUICK TICKER BUTTONS (Upgrade 1)
st.subheader("Quick Select")
c1, c2, c3, c4, c5 = st.columns(5)
default_ticker = "SPY"

with c1: 
    if st.button(" AAPL"): default_ticker = "AAPL"
with c2: 
    if st.button(" TSLA"): default_ticker = "TSLA"
with c3: 
    if st.button(" NVDA"): default_ticker = "NVDA"
with c4: 
    if st.button(" BTC"): default_ticker = "BTC/USD"
with c5: 
    if st.button(" SPY"): default_ticker = "SPY"

# 3. MAIN SEARCH BOX
SYMBOL = st.text_input("Search Ticker Symbol", value=default_ticker).upper()
st.divider()

# API Setup
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY") or st.secrets.get("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or st.secrets.get("ALPACA_SECRET_KEY")

if SYMBOL:
    try:
        with st.spinner(f"Analyzing {SYMBOL}..."):
            client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
            request_params = StockBarsRequest(symbol_or_symbols=[SYMBOL], timeframe=TimeFrame.Day, start=datetime(2022, 1, 1))
            bars = client.get_stock_bars(request_params)
            df = bars.df.reset_index(level=0, drop=True) if isinstance(bars.df.index, pd.MultiIndex) else bars.df

            # Ridge Logic & Feature Engineering
            df['SMA_10'] = df['close'].rolling(window=10).mean()
            df['SMA_30'] = df['close'].rolling(window=30).mean()
            df['Vol_20'] = df['close'].pct_change().rolling(window=20).std()
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + gain/loss))
            df['Target'] = df['close'].shift(-1)
            df.dropna(inplace=True)

            features = ['close', 'SMA_10', 'SMA_30', 'Vol_20', 'RSI']
            X, y = df[features], df['Target']
            split = int(len(df) * 0.8)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X.iloc[:split])
            X_test = scaler.transform(X.iloc[split:])
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y.iloc[:split])
            predictions = model.predict(X_test)
            
            # Prediction for tomorrow
            latest_scaled = scaler.transform(df[features].iloc[-1:].values)
            next_pred = model.predict(latest_scaled)[0]
            curr_price = df['close'].iloc[-1]

            # 4. TABS & METRICS (Upgrades 2, 3, & 4)
            tab1, tab2 = st.tabs([" AI Forecast", " Market Data"])

            with tab1:
                # Metric Row (Upgrade 2)
                m1, m2, m3 = st.columns(3)
                m1.metric("Current Price", f"${curr_price:.2f}")
                m2.metric("AI Target (Tomorrow)", f"${next_pred:.2f}", f"{((next_pred/curr_price)-1)*100:.2f}%")
                m3.metric("Analysis Range", "250 Days")

                # Styled Chart (Upgrade 3)
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df.index[split:], y.iloc[split:].values, label="Actual Path", color='#58a6ff', linewidth=2)
                ax.plot(df.index[split:], predictions, label="AI Forecast", color='#ff7b72', linestyle='--')
                ax.fill_between(df.index[split:], y.iloc[split:].values, predictions, color='#ff7b72', alpha=0.1)
                ax.set_title(f"{SYMBOL} Forecast vs Reality", fontsize=14, color='white')
                ax.grid(alpha=0.2)
                ax.legend()
                st.pyplot(fig)

            with tab2:
                st.write("### Recent Trading History")
                st.dataframe(df.tail(20), use_container_width=True)

    except Exception as e:
        st.error(f"Error loading {SYMBOL}. Ensure the ticker is correct.")