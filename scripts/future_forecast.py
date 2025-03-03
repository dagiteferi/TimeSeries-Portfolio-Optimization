"""
forecast.py
This script contains functions for forecasting Tesla stock prices using an LSTM model.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_model_and_data(model_path, data_path):
    """
    Load the trained LSTM model and historical data.
    
    Args:
        model_path (str): Path to the saved Keras model.
        data_path (str): Path to the historical data CSV file.
    
    Returns:
        model: Loaded Keras model.
        data: DataFrame containing historical stock prices.
    """
    logging.info("Loading model and data...")
    model = tf.keras.models.load_model(model_path)
    data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    logging.info(f"Model and data loaded successfully. Data shape: {data.shape}")
    return model, data

def preprocess_data(data, look_back, test_size=0.2):
    """
    Preprocess data for forecasting.
    
    Args:
        data (DataFrame): Historical stock prices.
        look_back (int): Sequence length for LSTM.
        test_size (float): Fraction of data to use as test set.
    
    Returns:
        scaler: Fitted MinMaxScaler.
        input_sequence: Prepared input sequence for forecasting.
    """
    logging.info("Preprocessing data...")
    train_size = int(len(data) * (1 - test_size))
    train_data = data.iloc[:train_size]
    
    # Fit scaler on training data returns
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_returns = train_data['Close'].pct_change().dropna().values.reshape(-1, 1)
    scaler.fit(train_returns)
    
    # Prepare input sequence
    returns = data['Close'].pct_change().dropna().values
    scaled_returns = scaler.transform(returns.reshape(-1, 1))
    input_sequence = scaled_returns[-look_back:].reshape(1, look_back, 1)
    
    logging.info("Data preprocessing complete.")
    return scaler, input_sequence

def forecast_with_ci(model, sequence, scaler, steps=252, n_simulations=100):
    """
    Generate forecasts with confidence intervals.
    
    Args:
        model: Trained LSTM model.
        sequence: Input sequence for forecasting.
        scaler: Fitted MinMaxScaler.
        steps (int): Number of steps to forecast.
        n_simulations (int): Number of Monte Carlo simulations.
    
    Returns:
        forecasts: Array of simulated price paths.
        median_forecast: Median forecast.
        lower_bound: 5th percentile of forecasts.
        upper_bound: 95th percentile of forecasts.
    """
    logging.info("Generating forecasts with confidence intervals...")
    forecasts = []
    last_price = data['Close'].iloc[-1]
    
    for _ in range(n_simulations):
        current_seq = sequence.copy()
        pred_prices = []
        
        for _ in range(steps):
            # Predict scaled return
            scaled_return = model.predict(current_seq, verbose=0)[0][0]
            # Add random noise for uncertainty
            scaled_return += np.random.normal(0, 0.05)  # 5% volatility assumption
            
            # Convert to price
            ret = scaler.inverse_transform([[scaled_return]])[0][0]
            last_price *= (1 + ret)
            pred_prices.append(last_price)
            
            # Update sequence
            current_seq = np.roll(current_seq, -1)
            current_seq[0, -1, 0] = scaled_return
        
        forecasts.append(pred_prices)
    
    forecasts = np.array(forecasts)
    median_forecast = np.median(forecasts, axis=0)
    lower_bound = np.percentile(forecasts, 5, axis=0)
    upper_bound = np.percentile(forecasts, 95, axis=0)
    
    logging.info("Forecast generation complete.")
    return forecasts, median_forecast, lower_bound, upper_bound

def plot_forecast(data, median_forecast, lower_bound, upper_bound, future_dates):
    """
    Plot the forecast alongside historical data.
    
    Args:
        data: Historical stock prices.
        median_forecast: Median forecast.
        lower_bound: 5th percentile of forecasts.
        upper_bound: 95th percentile of forecasts.
        future_dates: Dates for the forecast period.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[-500:], data['Close'].iloc[-500:], label='Historical Prices')
    plt.plot(future_dates, median_forecast, label='12-Month Forecast', color='orange')
    plt.fill_between(future_dates, lower_bound, upper_bound, color='orange', alpha=0.2)
    plt.title('TSLA Stock Price Forecast with 90% Confidence Interval', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_analysis(data, median_forecast, lower_bound, upper_bound, future_dates):
    """
    Generate a summary of the forecast analysis.
    
    Args:
        data: Historical stock prices.
        median_forecast: Median forecast.
        lower_bound: 5th percentile of forecasts.
        upper_bound: 95th percentile of forecasts.
        future_dates: Dates for the forecast period.
    
    Returns:
        analysis: String containing the analysis summary.
    """
    analysis = f"""
** Trend Analysis **
1. Direction: {'Bullish' if median_forecast[-1] > median_forecast[0] else 'Bearish'} 
   - Projected Change: {((median_forecast[-1]/data['Close'].iloc[-1]-1)*100):.1f}%
   - Pattern: {'Consistent Uptrend' if np.all(np.diff(median_forecast) > 0) else 'Volatile Movement'}

** Volatility & Risk **
1. Average CI Width: ${(upper_bound - lower_bound).mean():.2f}
2. Maximum Uncertainty: {future_dates[np.argmax(upper_bound - lower_bound)].strftime('%b %Y')}
   - Range: ${lower_bound.min():.2f} - ${upper_bound.max():.2f}

** Market Opportunities **
1. Potential Entry: ${lower_bound.mean():.2f} (±${(lower_bound.mean() - lower_bound.min()):.2f})
2. Price Targets:
   - Conservative: ${median_forecast.mean():.2f}
   - Optimistic: ${upper_bound.mean():.2f}

** Key Risks **
1. Downside Protection: ${lower_bound.min():.2f} 
2. High Volatility Periods: 
   - {future_dates[0].strftime('%b %Y')} (Initial Forecast Uncertainty)
   - {future_dates[126].strftime('%b %Y')} (Mid-Year Projections)
"""
    return analysis