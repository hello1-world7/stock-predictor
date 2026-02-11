# 📈 Stock Price Predictor AI

A professional Machine Learning web application that predicts the next trading day's closing price for any stock ticker. This project transitioned from a basic Random Forest script to a high-accuracy Ridge Regression pipeline with a live interactive dashboard.

## 🚀 Evolution of the Project
* **Initial Build:** Used a Random Forest Regressor. Discovered the "Flatline" issue where the model couldn't predict prices higher than the training data.
* **Final Build:** Swapped to **Ridge Regression** with feature scaling (**StandardScaler**).
* **Accuracy:** Successfully reduced the Mean Absolute Error (MAE) from ~$45.00 to **~$3.88** for tickers like SPY.

## 🛠️ Tech Stack
* **Language:** Python
* **ML Libraries:** Scikit-Learn (Ridge, StandardScaler)
* **Data:** Alpaca Markets API (Historical Stock Bars)
* **UI/UX:** Streamlit
* **Environment:** Python Dotenv (Security)

## 📦 Installation & Setup
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/yourusername/stock-predictor.git](https://github.com/yourusername/stock-predictor.git)

📊 Features
Real-time Data: Fetches live historical data via Alpaca API.

Technical Indicators: Automatically calculates SMA (10/30), Volatility, and RSI (14).

Interactive UI: Users can enter any ticker symbol (AAPL, TSLA, NVDA) to get instant predictions.