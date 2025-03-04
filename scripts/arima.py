import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import logging
import os

# Configure logging
LOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(LOG_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "logs", "arima.log")),
        logging.StreamHandler()
    ]
)

def preprocess_data(data):
    """Ensure proper datetime index with business day frequency"""
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    data = data.asfreq('B').ffill()
    return data

def train_arima(data, train_end='2024-12-31'):
    """Train ARIMA model with auto parameter optimization"""
    try:
        logging.info("Starting ARIMA model training")
        
        # Validate and preprocess data
        data = preprocess_data(data)
        if 'Close' not in data.columns:
            raise ValueError("Data missing 'Close' column")
        
        # Split training data
        train_data = data.loc[:train_end]['Close']
        logging.info(f"Training period: {train_data.index[0].date()} to {train_data.index[-1].date()}")
        
        # Optimize and fit ARIMA
        arima_model = auto_arima(
            train_data,
            seasonal=False,
            start_p=1, start_q=1,
            max_p=3, max_q=3,
            d=1,  # Force differencing for non-stationary data
            trace=True,
            error_action='ignore',
            suppress_warnings=True
        )
        
        logging.info(f"Best ARIMA parameters: {arima_model.order}")
        return arima_model, arima_model.order  # Return both model and parameters
    
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise



def forecast_arima(model, test_data):
    """Generate business-day-aligned forecasts"""
    try:
        logging.info(f"Forecasting {len(test_data)} steps")
        
        # Generate forecast using test data index
        forecast = model.predict(n_periods=len(test_data))
        forecast_series = pd.Series(forecast, index=test_data.index)
        
        if forecast_series.isnull().any():
            raise ValueError("Forecast contains NaN values")
        
        logging.info("Forecast completed successfully")
        return forecast_series
    
    except Exception as e:
        logging.error(f"Forecasting failed: {str(e)}")
        raise

def evaluate_arima(actual, forecast):
    """Calculate performance metrics"""
    try:
        # Align data
        aligned_actual, aligned_forecast = actual.align(forecast, join='inner')
        
        mae = mean_absolute_error(aligned_actual, aligned_forecast)
        rmse = np.sqrt(mean_squared_error(aligned_actual, aligned_forecast))
        mape = np.mean(np.abs((aligned_actual - aligned_forecast) / aligned_actual)) * 100
        
        logging.info(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        return mae, rmse, mape
    
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise

def plot_results(actual, forecast):
    """Plot actual vs forecasted prices"""
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label='Actual', color='blue')
    plt.plot(forecast.index, forecast, label='Forecast', linestyle='--', color='red')
    plt.title("TSLA Price Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()

