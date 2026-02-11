import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

# NEW: Linear Model and Scaler for better prediction
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# 1. SETUP
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
SYMBOL = "SPY"  

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
request_params = StockBarsRequest(symbol_or_symbols=[SYMBOL], timeframe=TimeFrame.Day, start=datetime(2022, 1, 1))
bars = client.get_stock_bars(request_params)
df = bars.df

if isinstance(df.index, pd.MultiIndex):
    df = df.reset_index(level=0, drop=True)

# 2. FEATURES
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

# 3. SPLIT & SCALE (Ridge needs scaled data)
features = ['close', 'SMA_10', 'SMA_30', 'Volatility_20', 'RSI_14']
X = df[features]
y = df['Target_Next_Close']

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. TRAIN RIDGE REGRESSION
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# 5. PREDICT
predictions = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, predictions)

print(f"MEAN ABSOLUTE ERROR (MAE): ${mae:.2f}")

latest_features = df[features].iloc[-1].values.reshape(1, -1)
latest_features_scaled = scaler.transform(latest_features)
predicted_tmrw = model.predict(latest_features_scaled)[0]

print(f"PREDICTED PRICE FOR NEXT TRADING DAY: ${predicted_tmrw:.2f}")

# 6. PLOT
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test.values, label="Actual Price", color='blue')
plt.plot(y_test.index, predictions, label="Predicted Price", color='orange', linestyle='--')
plt.title(f"{SYMBOL} Prediction (Ridge Regression)")
plt.legend()
plt.show()