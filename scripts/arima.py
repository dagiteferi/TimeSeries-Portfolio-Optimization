import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import logging
import os

# Configure logging to write to a file and console
LOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(LOG_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "logs", "arima.log")),  # Save logs to file in project logs/
        logging.StreamHandler()  # Print logs to console
    ]
)

def train_arima(data, train_size=None, train_end='2024-12-31'):
    """
    Train an ARIMA model on TSLA Close price, optimizing parameters using auto_arima.

    This function splits the data into training and testing sets, checks stationarity,
    and uses auto_arima to find the optimal ARIMA parameters (p, d, q) for forecasting TSLA stock prices.

    Parameters:
        data (pd.DataFrame): TSLA data with 'Close' column and Date index.
        train_size (int, optional): Number of training rows. Defaults to date-based split.
        train_end (str, optional): End date for training (default '2024-12-31').

    Returns:
        tuple: (fitted ARIMA model from pmdarima, optimal parameters as (p, d, q) tuple)

    Raises:
        ValueError: If the data lacks a 'Close' column, contains missing values, or if parameter optimization fails.
    """
    try:
        logging.info("Starting ARIMA model training on TSLA Close price data.")

        # Validate input data
        if 'Close' not in data.columns:
            raise ValueError("TSLA DataFrame missing 'Close' column")
        if data['Close'].isnull().any():
            raise ValueError("TSLA Close data contains missing values")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")

        # Split data for training based on train_size or train_end
        if train_size is None:
            train_data = data[data.index < train_end]['Close']
            logging.info(f"Using date-based split for training up to {train_end}. Training data size: {len(train_data)} rows.")
        else:
            train_data = data[:train_size]['Close']
            logging.info(f"Using row-based split for training. Training data size: {train_size} rows.")

        # Check stationarity with ADF test (if p-value > 0.05, data is non-stationary)
        adf_result = adfuller(train_data.dropna())
        logging.info(f"ADF Test - Statistic: {adf_result[0]}, p-value: {adf_result[1]}")
        if adf_result[1] > 0.05:
            logging.warning("Data appears non-stationary (p-value > 0.05). Applying differencing with d=1.")
            train_data = train_data.diff().dropna()  # Apply first differencing

        # Use auto_arima to automatically determine optimal ARIMA parameters
        logging.info("Optimizing ARIMA parameters with auto_arima (seasonal=False, max_p=3, max_q=3)...")
        arima_model = auto_arima(
            train_data, 
            seasonal=False,  # No seasonality based on Task 1 decomposition
            start_p=1, start_q=1, 
            max_p=3, max_q=3,  # Limit parameter search for efficiency
            d=None,  # Automatically detect differencing order (or use d=1 if non-stationary)
            trace=True,  # Show optimization progress for debugging
            error_action='ignore',  # Ignore errors during parameter search
            suppress_warnings=True  # Suppress non-critical warnings
        )
        
        # Log the best parameters found
        logging.info(f"Best ARIMA parameters found: {arima_model.order}")

        # Return the fitted model and its order
        return arima_model, arima_model.order

    except Exception as e:
        logging.error(f"Error training ARIMA model: {str(e)}")
        raise ValueError(f"ARIMA training failed: {str(e)}")

def forecast_arima(model, steps, start_date):
    """
    Forecast future TSLA Close prices using a trained ARIMA model and assign dates.

    This function generates out-of-sample forecasts for a specified number of steps, starting from a given date,
    and returns a pandas Series with a datetime index for alignment with actual data, ensuring business days.

    Parameters:
        model: Fitted ARIMA model from pmdarima.
        steps (int): Number of steps to forecast (e.g., number of days in January 2025).
        start_date (str): Start date for the forecast in 'YYYY-MM-DD' format (e.g., '2025-01-01').

    Returns:
        pd.Series: Forecasted Close prices with a datetime index aligned to business days.

    Raises:
        ValueError: If forecasting fails due to an invalid model, steps, or start_date.
    """
    try:
        logging.info(f"Forecasting {steps} steps starting from {start_date} with ARIMA model...")
        
        # Generate forecast using pmdarima's predict method
        forecast = model.predict(n_periods=steps)
        
        # Generate datetime index for the forecast period, using business days (B) to match stock market days
        dates = pd.date_range(start=start_date, periods=steps, freq='B')  # 'B' for business days
        forecast_series = pd.Series(forecast, index=dates)
        
        # Ensure the forecast values are finite and not NaN
        if forecast_series.isnull().any():
            logging.warning("Forecast contains NaN values. Checking model and data...")
            raise ValueError("Forecast contains NaN values; check model training or data stationarity.")
        
        logging.info("ARIMA forecast completed successfully with datetime index.")
        return forecast_series

    except Exception as e:
        logging.error(f"Error forecasting with ARIMA: {str(e)}")
        raise ValueError(f"ARIMA forecasting failed: {str(e)}")

def evaluate_arima(actual, forecast):
    """
    Evaluate ARIMA performance with MAE, RMSE, and MAPE for model assessment.

    This function compares actual TSLA Close prices with forecasted values to compute key performance metrics,
    ensuring both inputs have compatible datetime indices for alignment.

    Parameters:
        actual (pd.Series): Actual TSLA Close prices with Date index from the test set.
        forecast (pd.Series): Forecasted Close prices with Date index from forecast_arima.

    Returns:
        tuple: (Mean Absolute Error, Root Mean Squared Error, Mean Absolute Percentage Error)

    Raises:
        ValueError: If actual or forecast data is invalid, missing, or misaligned.
    """
    try:
        logging.info("Evaluating ARIMA model performance with MAE, RMSE, and MAPE...")
        
        # Validate inputs
        if not isinstance(actual, pd.Series) or not isinstance(forecast, pd.Series):
            raise ValueError("Actual and forecast must be pandas Series with Date indices")
        if actual.index.dtype != 'datetime64[ns]' or forecast.index.dtype != 'datetime64[ns]':
            raise ValueError("Actual and forecast must have datetime indices")
        if len(actual) != len(forecast):
            raise ValueError("Actual and forecast lengths must match for evaluation")
        if actual.isnull().any() or forecast.isnull().any():
            raise ValueError("Actual or forecast contains NaN values; check data or forecast.")

        # Calculate evaluation metrics
        mae = mean_absolute_error(actual, forecast)
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100

        logging.info(f"ARIMA Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        return mae, rmse, mape

    except Exception as e:
        logging.error(f"Error evaluating ARIMA: {str(e)}")
        raise ValueError(f"ARIMA evaluation failed: {str(e)}")

def plot_arima(actual, forecast, title="ARIMA Forecast vs Actual TSLA Closing Price"):
    """
    Plot ARIMA forecast vs actual values with aligned dates for visualization.

    This function creates a clear, labeled plot comparing actual and forecasted TSLA Close prices,
    ensuring both series have datetime indices for proper alignment.

    Parameters:
        actual (pd.Series): Actual TSLA Close prices with Date index.
        forecast (pd.Series): Forecasted TSLA Close prices with Date index.
        title (str): Title for the plot, defaulting to ARIMA forecast vs actual.

    Raises:
        ValueError: If actual or forecast data is invalid or lacks datetime indices.
    """
    try:
        logging.info("Plotting ARIMA forecast vs actual TSLA Close prices...")
        
        # Validate inputs
        if not isinstance(actual, pd.Series) or not isinstance(forecast, pd.Series):
            raise ValueError("Actual and forecast must be pandas Series with Date indices")
        if actual.index.dtype != 'datetime64[ns]' or forecast.index.dtype != 'datetime64[ns]':
            raise ValueError("Actual and forecast must have datetime indices")

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(actual.index, actual, label='Actual Close Price', color='blue')
        plt.plot(forecast.index, forecast, label='ARIMA Forecast', linestyle='--', color='red')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()

        logging.info("ARIMA plot generated successfully.")

    except Exception as e:
        logging.error(f"Error plotting ARIMA results: {str(e)}")
        raise ValueError(f"ARIMA plotting failed: {str(e)}")