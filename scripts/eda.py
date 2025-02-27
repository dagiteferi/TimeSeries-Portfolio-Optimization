import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("eda.log"),  # Save logs to a file
        logging.StreamHandler()          # Print logs to the console
    ]
)

def visualize_closing_price(data, ticker):
    """
    Visualize the closing price over time.
    
    Parameters:
        data (pd.DataFrame): The dataset containing the closing prices.
        ticker (str): The ticker symbol (e.g., "TSLA").
    """
    try:
        logging.info(f"Visualizing closing price for {ticker}...")
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data["Close"], label="Closing Price", color="blue")
        plt.title(f"{ticker} Closing Price Over Time")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.legend()
        plt.grid()
        plt.show()
        logging.info(f"Closing price visualization completed for {ticker}.")
    except Exception as e:
        logging.error(f"Error visualizing closing price for {ticker}: {e}")

def calculate_daily_returns(data, ticker):
    """
    Calculate and plot the daily percentage change (returns).
    
    Parameters:
        data (pd.DataFrame): The dataset containing the closing prices.
        ticker (str): The ticker symbol (e.g., "TSLA").
    """
    try:
        logging.info(f"Calculating daily returns for {ticker}...")
        data["Daily_Return"] = data["Close"].pct_change() * 100  # Calculate daily returns in percentage

        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data["Daily_Return"], label="Daily Returns", color="green", alpha=0.7)
        plt.title(f"{ticker} Daily Percentage Change (Returns)")
        plt.xlabel("Date")
        plt.ylabel("Daily Return (%)")
        plt.legend()
        plt.grid()
        plt.show()
        logging.info(f"Daily returns calculation and visualization completed for {ticker}.")
    except Exception as e:
        logging.error(f"Error calculating daily returns for {ticker}: {e}")

def analyze_volatility(data, ticker, window=30):
    """
    Analyze volatility using rolling statistics.
    
    Parameters:
        data (pd.DataFrame): The dataset containing the closing prices.
        ticker (str): The ticker symbol (e.g., "TSLA").
        window (int): The rolling window size (default is 30 days).
    """
    try:
        logging.info(f"Analyzing volatility for {ticker} using {window}-day rolling statistics...")
        data["Rolling_Mean"] = data["Close"].rolling(window=window).mean()
        data["Rolling_Std"] = data["Close"].rolling(window=window).std()

        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data["Close"], label="Closing Price", color="blue", alpha=0.7)
        plt.plot(data.index, data["Rolling_Mean"], label=f"{window}-Day Rolling Mean", color="red")
        plt.plot(data.index, data["Rolling_Std"], label=f"{window}-Day Rolling Std", color="orange")
        plt.title(f"{ticker} Volatility Analysis (Rolling Statistics)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        plt.show()
        logging.info(f"Volatility analysis completed for {ticker}.")
    except Exception as e:
        logging.error(f"Error analyzing volatility for {ticker}: {e}")

def detect_outliers(data, ticker, column="Close", threshold=3):
    """
    Detect outliers using the Z-score method.
    
    Parameters:
        data (pd.DataFrame): The dataset.
        ticker (str): The ticker symbol (e.g., "TSLA").
        column (str): The column to analyze (default is "Close").
        threshold (float): The Z-score threshold for detecting outliers (default is 3).
    """
    try:
        logging.info(f"Detecting outliers for {ticker} using Z-score method...")
        data["Z_Score"] = zscore(data[column])
        outliers = data[abs(data["Z_Score"]) > threshold]

        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data[column], label=column, color="blue", alpha=0.7)
        plt.scatter(outliers.index, outliers[column], color="red", label="Outliers")
        plt.title(f"{ticker} Outlier Detection (Z-Score)")
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.legend()
        plt.grid()
        plt.show()
        logging.info(f"Outlier detection completed for {ticker}.")
    except Exception as e:
        logging.error(f"Error detecting outliers for {ticker}: {e}")

def analyze_extreme_returns(data, ticker, threshold=2):
    """
    Analyze days with unusually high or low returns.
    
    Parameters:
        data (pd.DataFrame): The dataset containing daily returns.
        ticker (str): The ticker symbol (e.g., "TSLA").
        threshold (float): The threshold for extreme returns (default is 2%).
    """
    try:
        logging.info(f"Analyzing extreme returns for {ticker}...")
        high_returns = data[data["Daily_Return"] > threshold]
        low_returns = data[data["Daily_Return"] < -threshold]

        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data["Daily_Return"], label="Daily Returns", color="green", alpha=0.7)
        plt.scatter(high_returns.index, high_returns["Daily_Return"], color="red", label="High Returns")
        plt.scatter(low_returns.index, low_returns["Daily_Return"], color="blue", label="Low Returns")
        plt.title(f"{ticker} Extreme Returns Analysis")
        plt.xlabel("Date")
        plt.ylabel("Daily Return (%)")
        plt.legend()
        plt.grid()
        plt.show()
        logging.info(f"Extreme returns analysis completed for {ticker}.")
    except Exception as e:
        logging.error(f"Error analyzing extreme returns for {ticker}: {e}")