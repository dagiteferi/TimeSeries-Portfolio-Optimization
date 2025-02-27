import pandas as pd
import matplotlib.pyplot as plt
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
        logging.FileHandler(os.path.join(ROOT_DIR, "logs", "volatility_analysis.log")),  # Save logs to a file
        logging.StreamHandler()  # Print logs to the console
    ]
)

def analyze_volatility(data, ticker, window=30):
    """
    Analyze volatility using rolling statistics.
    
    Parameters:
        data (pd.DataFrame): The dataset containing the closing prices.
        ticker (str): The ticker symbol (e.g., "TSLA").
        window (int): The rolling window size (default is 30 days).
    
    Returns:
        pd.DataFrame: The dataset with added rolling mean and rolling standard deviation columns.
    """
    try:
        logging.info(f"Analyzing volatility for {ticker} with a {window}-day rolling window...")
        
        # Calculate rolling mean and rolling standard deviation
        data["Rolling_Mean"] = data["Close"].rolling(window=window).mean()
        data["Rolling_Std"] = data["Close"].rolling(window=window).std()
        
        # Plot the results
        logging.info(f"Plotting rolling statistics for {ticker}...")
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
        return data  # Return the dataset with rolling statistics
    except Exception as e:
        logging.error(f"Error analyzing volatility for {ticker}: {e}")
        return None