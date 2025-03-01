# sarima.py
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
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
        logging.FileHandler(os.path.join(LOG_DIR, "logs", "sarima.log")),
        logging.StreamHandler()
    ]
)

def preprocess_data(data):
    """
    Preprocess time series data with business day alignment and forward filling
    Returns data with proper DatetimeIndex and business day frequency
    """
    try:
        logging.info("Preprocessing data - converting to business day frequency")
        data = data.copy()
        data.index = pd.to_datetime(data.index)
        data = data.asfreq('B')  # Business day frequency
        data = data.ffill()  # Forward fill missing values
        logging.info(f"New date range: {data.index.min()} to {data.index.max()}")
        return data
    except Exception as e:
        logging.error(f"Data preprocessing failed: {str(e)}")
        raise

def check_seasonality(data, period=252):
    """
    Perform seasonal decomposition to identify seasonal patterns
    period: Number of observations per seasonal cycle (252 trading days = 1 year)
    """
    try:
        logging.info("Performing seasonal decomposition")
        decomposition = seasonal_decompose(data['Close'], period=period)
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        plt.show()
        
        # Log seasonal components
        seasonal_strength = decomposition.seasonal.std() / data['Close'].std()
        logging.info(f"Seasonal strength: {seasonal_strength:.2%}")
        return decomposition
    except Exception as e:
        logging.warning(f"Seasonal decomposition failed: {str(e)}")
        return None

def train_sarima(data, train_end='2024-12-31', seasonal_period=63):
    """
    Train SARIMA model with automatic parameter optimization
    seasonal_period: Number of periods in seasonal cycle (63 ≈ quarterly trading days)
    Returns fitted model and parameters (order, seasonal_order)
    """
    try:
        logging.info("Starting SARIMA model training")
        
        # Validate input data
        if 'Close' not in data.columns:
            raise ValueError("Data missing 'Close' column")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

        # Split training data
        train_data = data.loc[:train_end]['Close']
        logging.info(f"Training period: {train_data.index[0].date()} to {train_data.index[-1].date()}")
        logging.info(f"Training samples: {len(train_data)}")

        # Seasonal decomposition check
        decomposition = check_seasonality(data.loc[:train_end])

        # Auto-ARIMA with seasonal components
        logging.info(f"Optimizing SARIMA parameters with seasonal_period={seasonal_period}")
        sarima_model = auto_arima(
            train_data,
            seasonal=True,  # Enable seasonal components
            m=seasonal_period,  # Seasonal cycle length
            start_p=1, start_q=1,
            max_p=3, max_q=3,
            d=None,  # Auto-detect differencing
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            D=None,  # Auto-detect seasonal differencing
            trace=True,
            error_action='ignore',
            suppress_warnings=True
        )

        # Log model parameters
        logging.info(f"Best SARIMA parameters: {sarima_model.order}{sarima_model.seasonal_order}")
        logging.info("Model summary:\n" + str(sarima_model.summary()))
        
        return sarima_model, (sarima_model.order, sarima_model.seasonal_order)

    except Exception as e:
        logging.error(f"SARIMA training failed: {str(e)}")
        raise

def forecast_sarima(model, test_data):
    """
    Generate business-day-aligned forecasts using SARIMA model
    Returns forecast series with same index as test_data
    """
    try:
        logging.info(f"Generating {len(test_data)}-step forecast")
        
        # Generate predictions
        forecast = model.predict(n_periods=len(test_data))
        
        # Create forecast series with test data index
        forecast_series = pd.Series(forecast, index=test_data.index)
        
        # Validate forecast
        if forecast_series.isnull().any():
            raise ValueError("Forecast contains NaN values")
        if len(forecast_series) != len(test_data):
            raise ValueError("Forecast/test length mismatch")

        logging.info("Forecast completed successfully")
        return forecast_series

    except Exception as e:
        logging.error(f"Forecasting failed: {str(e)}")
        raise

def evaluate_sarima(actual, forecast):
    """
    Calculate and log evaluation metrics for SARIMA forecasts
    Returns MAE, RMSE, MAPE tuple
    """
    try:
        logging.info("Evaluating SARIMA model performance")
        
        # Align data
        aligned_actual, aligned_forecast = actual.align(forecast, join='inner')
        
        # Calculate metrics
        mae = mean_absolute_error(aligned_actual, aligned_forecast)
        rmse = np.sqrt(mean_squared_error(aligned_actual, aligned_forecast))
        mape = np.mean(np.abs((aligned_actual - aligned_forecast) / aligned_actual)) * 100

        logging.info(f"SARIMA Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        return mae, rmse, mape

    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise

def plot_sarima_results(actual, forecast, title="SARIMA Forecast vs Actual"):
    """
    Generate comparison plot with actual vs forecasted values
    """
    try:
        logging.info("Generating SARIMA results plot")
        plt.figure(figsize=(12, 6))
        plt.plot(actual.index, actual, label='Actual', color='blue', linewidth=2)
        plt.plot(forecast.index, forecast, label='Forecast', linestyle='--', color='red')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()
        logging.info("Plot generated successfully")
    except Exception as e:
        logging.error(f"Plotting failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Load and preprocess data
        logging.info("SARIMA Forecasting Pipeline Started")
        data = pd.read_csv("TSLA_cleaned.csv", index_col="Date", parse_dates=True)
        data = preprocess_data(data)

        # Train SARIMA model
        sarima_model, params = train_sarima(data, seasonal_period=63)
        logging.info(f"Final SARIMA Parameters: Order={params[0]}, Seasonal Order={params[1]}")

        # Prepare test data
        test_data = data['2025-01-01':'2025-01-31']['Close']
        logging.info(f"Test period: {test_data.index[0].date()} to {test_data.index[-1].date()}")

        # Generate and validate forecast
        forecast = forecast_sarima(sarima_model, test_data)

        # Evaluate and visualize
        evaluate_sarima(test_data, forecast)
        plot_sarima_results(test_data, forecast)

        # Show sample forecasts
        logging.info("Forecast Sample:")
        print(forecast.head())

    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        raise