import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# 1. SETUP & UI CONFIG
st.set_page_config(page_title="Stock Predictor AI", layout="wide")
st.title("📈 Stock Price Predictor")

# Load credentials
load_dotenv()
# For production, use st.secrets; for local, use env
API_KEY = os.getenv("ALPACA_API_KEY") or st.secrets.get("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or st.secrets.get("ALPACA_SECRET_KEY")

# --- MOBILE FRIENDLY UI (Moved from sidebar to main body) ---
col1, col2 = st.columns([3, 1]) # Search bar takes 75%, button takes 25%

with col1:
    SYMBOL = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, NVDA)", value="SPY").upper()

with col2:
    st.write(" ") # Padding to align with input field
    run_button = st.button("Run Prediction", use_container_width=True)

st.divider() # Adds a clean line under search section

if run_button:
    try:
        client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        request_params = StockBarsRequest(symbol_or_symbols=[SYMBOL], timeframe=TimeFrame.Day, start=datetime(2022, 1, 1))
        bars = client.get_stock_bars(request_params)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)

        # 2. FEATURE ENGINEERING
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_30'] = df['close'].rolling(window=30).mean()
        df['Daily_Return'] = df['close'].pct_change()
        df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        df['Target_Next_Close'] = df['close'].shift(-1)
        df.dropna(inplace=True)

        # 3. ML MODEL
        features = ['close', 'SMA_10', 'SMA_30', 'Volatility_20', 'RSI_14']
        X = df[features]
        y = df['Target_Next_Close']
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

        # 4. RESULTS DISPLAY
        latest_features = df[features].iloc[-1].values.reshape(1, -1)
        latest_features_scaled = scaler.transform(latest_features)
        predicted_price = model.predict(latest_features_scaled)[0]

        st.metric(label=f"Predicted Next Close for {SYMBOL}", value=f"${predicted_price:.2f}")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test.index, y_test.values, label="Actual Price", color='#1f77b4')
        ax.plot(y_test.index, predictions, label="Predicted Price", color='#ff7f0e', linestyle='--')
        ax.legend()
        ax.set_title(f"{SYMBOL} Accuracy Chart")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error fetching data for {SYMBOL}. Check the ticker name and your API keys.")