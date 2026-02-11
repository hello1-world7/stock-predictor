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
from supabase import create_client, Client

# 1. UI CONFIGURATION
st.set_page_config(page_title="Institutional Stock AI", layout="wide")

# DATABASE CONNECTION
@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()


# CSS for single-line scrollable buttons and light/dark mode visibility
st.markdown("""
    <style>
    /* Force buttons into a single horizontal scrolling row */
    .stHorizontalBlock {
        display: flex;
        flex-wrap: nowrap;
        overflow-x: auto;
        gap: 10px;
        padding-bottom: 10px;
    }
    .stHorizontalBlock > div {
        flex: 0 0 auto;
        min-width: 90px;
    }
    /* Button Styling for visibility in both modes */
    div.stButton > button {
        border-radius: 8px;
        border: 1px solid #58a6ff;
        font-weight: bold;
    }
    /* Metric card contrast fix */
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

st.title("Institutional Stock Predictor")

# 2. SINGLE-LINE QUICK SELECT (Upgrade)
st.write("### Quick Select")
c1, c2, c3, c4, c5 = st.columns(5)
default_ticker = "SPY"

# Logic for single-line behavior
with c1: 
    if st.button("AAPL"): default_ticker = "AAPL"
with c2: 
    if st.button("TSLA"): default_ticker = "TSLA"
with c3: 
    if st.button("NVDA"): default_ticker = "NVDA"
with c4: 
    if st.button("SPY"): default_ticker = "SPY"

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

            # Ridge Logic
            df['SMA_10'] = df['close'].rolling(window=10).mean()
            df['SMA_30'] = df['close'].rolling(window=30).mean()
            df['Vol_20'] = df['close'].pct_change().rolling(window=20).std()
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + gain/(loss + 1e-9)))
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
            
            latest_scaled = scaler.transform(df[features].iloc[-1:].values)
            next_pred = model.predict(latest_scaled)[0]
            curr_price = df['close'].iloc[-1]

            # 3. TABS & VISIBILITY FIXES
            tab1, tab2 = st.tabs(["AI Forecast", "Market Data"])

            with tab1:
                m1, m2, m3 = st.columns(3)
                m1.metric("Current Price", f"${curr_price:.2f}")
                m2.metric("AI Target", f"${next_pred:.2f}", f"{((next_pred/curr_price)-1)*100:.2f}%")
                m3.metric("Range", "250 Days")

                # Dynamic chart colors based on theme
                fig, ax = plt.subplots(figsize=(12, 5))
                # Set transparent background so it works in light & dark mode
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                
                # Use standard colors that pop on both white and black backgrounds
                ax.plot(df.index[split:], y.iloc[split:].values, label="Actual Path", color='#1f77b4', linewidth=2.5)
                ax.plot(df.index[split:], predictions, label="AI Forecast", color='#d62728', linestyle='--', linewidth=2)
                
                # Fix label colors for light mode visibility
                ax.tick_params(colors='gray', labelsize=10)
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')
                
                ax.legend(facecolor='inherit', framealpha=0.5)
                st.pyplot(fig)

                # --- DATABASE LOGGING ---
            log_data = {
                "ticker": SYMBOL,
                "predicted_price": float(next_pred), # next_pred is your model's target
                "actual_price": float(curr_price),   # curr_price is today's close
                "error_mae": float(abs(next_pred - curr_price)) 
            }

            try:
                # This sends the data to your 'predictions' table
                supabase.table("predictions").insert(log_data).execute()
            except Exception as e:
                st.warning(f"Database sync skipped: {e}")

            with tab2:
                st.dataframe(df.tail(20), use_container_width=True)

    except Exception as e:
        st.error(f"Error loading {SYMBOL}. Verify ticker symbol.")