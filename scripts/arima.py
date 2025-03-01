import pandas as pd
import numpy as np
import logging
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("arima_forecast.log"),
        logging.StreamHandler()
    ]
)

def train_arima(data, train_size=None, train_end='2024-12-31'):
    """
    Train and optimize an ARIMA model using auto_arima for parameter selection.
    
    Parameters:
        data (pd.DataFrame): Time series data with DateTime index and 'Close' column
        train_size (int, optional): Number of samples for training. Default uses date split
        train_end (str): Cutoff date for training data (format: 'YYYY-MM-DD')
    
    Returns:
        tuple: (fitted ARIMA model, (p, d, q) order tuple)
    
    Raises:
        ValueError: If input data is invalid or training fails
    """
    try:
        logging.info("🔧 Starting ARIMA model training...")
        
        # Split data based on date or sample size
        if train_size is None:
            train_data = data[data.index < train_end]['Close']
            logging.info(f"📅 Using date-based split. Training data up to {train_end}")
        else:
            train_data = data[:train_size]['Close']
            logging.info(f"📏 Using sample-based split. Training on first {train_size} samples")

        # Automatically find optimal ARIMA parameters
        logging.info("🔄 Running auto_arima for parameter optimization...")
        arima_model = auto_arima(
            train_data,
            seasonal=False,
            start_p=1, start_q=1,
            max_p=3, max_q=3,
            d=None,
            trace=True,
            error_action='ignore',
            suppress_warnings=True
        )
        
        logging.info(f"✅ ARIMA model trained successfully. Optimal order: {arima_model.order}")
        return arima_model, arima_model.order
        
    except Exception as e:
        logging.error(f"❌ ARIMA training failed: {str(e)}")
        raise ValueError(f"ARIMA training error: {str(e)}")

def forecast_arima(model, steps, start_date):
    """
    Generate future forecasts with date alignment using trained ARIMA model.
    
    Parameters:
        model: Trained pmdarima ARIMA model
        steps (int): Number of periods to forecast
        start_date (str): First date of forecast period ('YYYY-MM-DD')
    
    Returns:
        pd.Series: Forecasted values with DateTime index
    
    Raises:
        ValueError: If forecasting fails due to invalid inputs
    """
    try:
        logging.info(f"🔮 Generating {steps}-step forecast starting from {start_date}...")
        
        # Generate predictions and create date index
        forecast = model.predict(n_periods=steps)
        dates = pd.date_range(start=start_date, periods=steps)
        forecast_series = pd.Series(forecast, index=dates)
        
        logging.info("✨ Forecast generated successfully")
        return forecast_series
        
    except Exception as e:
        logging.error(f"❌ Forecasting failed: {str(e)}")
        raise ValueError(f"Forecasting error: {str(e)}")

def evaluate_arima(actual, forecast):
    """
    Calculate performance metrics between actual and forecasted values.
    
    Parameters:
        actual (pd.Series): Ground truth values with DateTime index
        forecast (pd.Series): Predicted values with DateTime index
    
    Returns:
        tuple: (MAE, RMSE, MAPE) metrics
    
    Raises:
        ValueError: If input series don't align temporally
    """
    try:
        logging.info("📊 Evaluating forecast performance...")
        
        # Ensure temporal alignment
        if not actual.index.equals(forecast.index):
            raise ValueError("Mismatched indices between actual and forecast")
            
        # Calculate metrics
        mae = mean_absolute_error(actual, forecast)
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        
        logging.info(f"📈 Evaluation complete - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        return mae, rmse, mape
        
    except Exception as e:
        logging.error(f"❌ Evaluation failed: {str(e)}")
        raise ValueError(f"Evaluation error: {str(e)}")

def plot_arima(actual, forecast, title="ARIMA Forecast vs Actual Prices"):
    """
    Visual comparison of actual and forecasted values.
    
    Parameters:
        actual (pd.Series): Historical actual values
        forecast (pd.Series): Forecasted values
        title (str): Custom title for the plot
    
    Raises:
        ValueError: If input data cannot be plotted
    """
    try:
        logging.info("🎨 Generating comparison plot...")
        
        plt.figure(figsize=(12, 6))
        plt.plot(actual.index, actual, label='Actual', color='blue', linewidth=2)
        plt.plot(forecast.index, forecast, label='Forecast', 
                linestyle='--', color='red', linewidth=1.5)
        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.4)
        plt.tight_layout()
        
        logging.info("🖼️ Plot displayed successfully")
        plt.show()
        
    except Exception as e:
        logging.error(f"❌ Plotting failed: {str(e)}")
        raise ValueError(f"Plotting error: {str(e)}")

