import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def train_arima(data, train_size=None, train_end='2024-12-31'):
    """
    Train an ARIMA model on TSLA Close price, optimizing parameters with auto_arima.

    Parameters:
        data (pd.DataFrame): TSLA data with 'Close' column and Date index.
        train_size (int, optional): Number of training rows. Defaults to date-based split.
        train_end (str, optional): End date for training (default '2024-12-31').

    Returns:
        tuple: (fitted ARIMA model, optimal parameters as (p, d, q) tuple)

    Raises:
        ValueError: If data or parameters are invalid.
    """
    # Split data for training if train_size not provided
    if train_size is None:
        train_data = data[data.index < train_end]['Close']
    else:
        train_data = data[:train_size]['Close']

    # Use auto_arima to find optimal ARIMA parameters
    arima_model = auto_arima(train_data, seasonal=False, start_p=1, start_q=1, max_p=3, max_q=3, d=None, 
                             trace=True, error_action='ignore', suppress_warnings=True)
    
    # Fit the ARIMA model
    arima_fit = arima_model.fit(train_data)
    return arima_fit, arima_model.order

def forecast_arima(model, steps):
    """
    Forecast future TSLA Close prices using a trained ARIMA model.

    Parameters:
        model: Fitted ARIMA model from statsmodels.
        steps (int): Number of steps to forecast.

    Returns:
        np.ndarray: Forecasted Close prices.

    Raises:
        ValueError: If forecasting fails due to invalid model or steps.
    """
    try:
        # Use .forecast() for statsmodels ARIMA (post-0.13.0), ensuring the model is fitted
        forecast = model.get_forecast(steps=steps).predicted_mean  # Use get_forecast for predicted mean
        return forecast
    except Exception as e:
        raise ValueError(f"Error forecasting with ARIMA: {str(e)}")

def evaluate_arima(actual, forecast):
    """
    Evaluate ARIMA performance with MAE, RMSE, and MAPE.

    Parameters:
        actual (pd.Series): Actual Close prices with Date index.
        forecast (np.ndarray): Forecasted Close prices.

    Returns:
        tuple: (MAE, RMSE, MAPE)

    Raises:
        ValueError: If evaluation fails due to invalid data.
    """
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return mae, rmse, mape

def plot_arima(actual, forecast, title="ARIMA Forecast vs Actual TSLA Closing Price"):
    """
    Plot ARIMA forecast vs actual values for visualization.

    Parameters:
        actual (pd.Series): Actual Close prices with Date index.
        forecast (np.ndarray): Forecasted Close prices.
        title (str): Plot title.

    Raises:
        ValueError: If plotting fails due to invalid data.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label='Actual Close Price')
    plt.plot(actual.index, forecast, label='ARIMA Forecast', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid()
    plt.show()