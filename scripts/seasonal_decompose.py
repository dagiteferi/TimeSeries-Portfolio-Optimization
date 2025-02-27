import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import logging
import os

# Define root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(ROOT_DIR, "logs", "seasonal_decompose.log")),  # Save logs to a file
        logging.StreamHandler()  # Print logs to the console
    ]
)

def decompose_time_series(data, ticker, period=252):
    """
    Decompose the time series into trend, seasonal, and residual components.
    
    Parameters:
        data (pd.DataFrame): The dataset containing the time series.
        ticker (str): The ticker symbol (e.g., "TSLA").
        period (int): The period for seasonal decomposition (default is 252 for daily financial data).
    """
    try:
        logging.info(f"Starting time series decomposition for {ticker}...")
        
        # Perform seasonal decomposition
        logging.info(f"Performing seasonal decomposition for {ticker} with period={period}...")
        decomposition = seasonal_decompose(data["Close"], model="additive", period=period)
        
        # Plot the decomposition
        logging.info(f"Plotting decomposition results for {ticker}...")
        plt.figure(figsize=(12, 8))
        
        # Original Time Series
        plt.subplot(4, 1, 1)
        plt.plot(data.index, data["Close"], label="Original", color="blue")
        plt.legend()
        plt.title(f"{ticker} Time Series Decomposition")
        
        # Trend Component
        plt.subplot(4, 1, 2)
        plt.plot(data.index, decomposition.trend, label="Trend", color="red")
        plt.legend()
        
        # Seasonal Component
        plt.subplot(4, 1, 3)
        plt.plot(data.index, decomposition.seasonal, label="Seasonal", color="green")
        plt.legend()
        
        # Residual Component
        plt.subplot(4, 1, 4)
        plt.plot(data.index, decomposition.resid, label="Residual", color="purple")
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        logging.info(f"Time series decomposition completed for {ticker}.")
    except Exception as e:
        logging.error(f"Error decomposing time series for {ticker}: {e}")