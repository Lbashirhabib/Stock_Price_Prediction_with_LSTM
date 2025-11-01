import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Streamlit app configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("🚀 AI Stock Price Predictor")
st.markdown("Predict future stock prices using LSTM neural networks")

# Sidebar for user input
st.sidebar.title("Settings")

# Stock symbol input
symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()

# Model parameters
lookback = st.sidebar.slider("Lookback Days", 30, 90, 60)
future_days = st.sidebar.slider("Future Prediction Days", 7, 90, 30)

# Main function
def main():
    if st.sidebar.button("Predict Stock Prices"):
        with st.spinner("Fetching data and training model..."):
            
            # Fetch data
            stock_data = fetch_stock_data(symbol)
            if stock_data is None:
                st.error(f"Could not fetch data for {symbol}")
                return
            
            # Prepare data
            X, y, scaler = prepare_data(stock_data, lookback)
            
            # Build and train model
            model = build_lstm_model((X.shape[1], 1))
            model.fit(X, y, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
            
            # Make predictions
            predictions = make_predictions(model, stock_data, scaler, lookback)
            
            # Create results
            results = stock_data.iloc[lookback:].copy()
            results['Predicted'] = predictions
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Historical Performance")
                fig_historical = create_historical_plot(results, symbol)
                st.plotly_chart(fig_historical, use_container_width=True)
            
            with col2:
                st.subheader("🔮 Future Predictions")
                future_dates, future_prices = predict_future_prices(
                    model, stock_data, scaler, lookback, future_days
                )
                fig_future = create_future_plot(results, future_dates, future_prices, symbol)
                st.plotly_chart(fig_future, use_container_width=True)
                
                # Show future predictions table
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_prices
                })
                st.dataframe(future_df.style.format({'Predicted Price': '${:.2f}'}))
            
            # Model metrics
            st.subheader("📈 Model Performance")
            metrics = calculate_metrics(results['Close'], results['Predicted'])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"${metrics['MAE']:.2f}")
            col2.metric("RMSE", f"${metrics['RMSE']:.2f}")
            col3.metric("R² Score", f"{metrics['R2']:.4f}")
            col4.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}")

# Helper functions (same as above, adapted for Streamlit)
def fetch_stock_data(symbol, period="2y"):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data if not data.empty else None
    except:
        return None

def prepare_data(data, lookback_days):
    prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    X, y = [], []
    for i in range(lookback_days, len(scaled_prices)):
        X.append(scaled_prices[i-lookback_days:i, 0])
        y.append(scaled_prices[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def make_predictions(model, data, scaler, lookback_days):
    prices = data['Close'].values.reshape(-1, 1)
    scaled_prices = scaler.transform(prices)
    
    X_test = []
    for i in range(lookback_days, len(scaled_prices)):
        X_test.append(scaled_prices[i-lookback_days:i, 0])
    
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    predictions = model.predict(X_test)
    return scaler.inverse_transform(predictions)

def create_historical_plot(results, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results.index, y=results['Close'], name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=results.index, y=results['Predicted'], name='Predicted', line=dict(color='red', dash='dash')))
    fig.update_layout(title=f'{symbol} - Historical Predictions', xaxis_title='Date', yaxis_title='Price ($)')
    return fig

def create_future_plot(results, future_dates, future_prices, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results.index, y=results['Close'], name='Historical', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name='Future Prediction', line=dict(color='green', dash='dot')))
    fig.update_layout(title=f'{symbol} - Future Predictions', xaxis_title='Date', yaxis_title='Price ($)')
    return fig

def calculate_metrics(actual, predicted):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import math
    return {
        'MAE': mean_absolute_error(actual, predicted),
        'RMSE': math.sqrt(mean_squared_error(actual, predicted)),
        'R2': r2_score(actual, predicted)
    }

if __name__ == "__main__":
    main()