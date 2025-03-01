import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

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
    """
    # Split data for training
    if train_size is None:
        train_data = data[data.index < train_end]['Close']
    else:
        train_data = data[:train_size]['Close']

    # Use auto_arima to find optimal parameters (model is already fitted)
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
    
    # Return the fitted model and its order
    return arima_model, arima_model.order

def forecast_arima(model, steps, start_date):
    """
    Forecast future TSLA Close prices using a trained ARIMA model and assign dates.

    Parameters:
        model: Fitted ARIMA model from pmdarima.
        steps (int): Number of steps to forecast.
        start_date (str): Start date for the forecast (e.g., '2025-01-01').

    Returns:
        pd.Series: Forecasted Close prices with Date index.
    """
    try:
        forecast = model.predict(n_periods=steps)
        # Generate date index for the forecast
        dates = pd.date_range(start=start_date, periods=steps)
        return pd.Series(forecast, index=dates)
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
    Plot ARIMA forecast vs actual values with aligned dates.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label='Actual Close Price')
    plt.plot(forecast.index, forecast, label='ARIMA Forecast', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid()
    plt.show()