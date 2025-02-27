import yfinance as yf
import pandas as pd
import logging
import os

# Set up logging
logging.basicConfig(
    filename="logs/fetch_data.log",  # Log file path
    level=logging.INFO,             # Log level (INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    filemode="w"                    # Overwrite the log file each time
)

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)  # For storing data
os.makedirs("logs", exist_ok=True)  # For storing logs

def fetch_asset_data(ticker, start_date, end_date):
    """
    Fetch historical data for a given ticker using yfinance.
    
    Parameters:
        ticker (str): The ticker symbol (e.g., "TSLA").
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
    
    Returns:
        pd.DataFrame: Historical data for the ticker.
    """
    try:
        logging.info(f"Fetching data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        logging.info(f"Successfully fetched data for {ticker}.")
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

def save_data(data, ticker):
    """
    Save the fetched data to a CSV file in the data folder.
    
    Parameters:
        data (pd.DataFrame): The data to save.
        ticker (str): The ticker symbol (e.g., "TSLA").
    """
    try:
        file_path = f"data/{ticker}_data.csv"
        data.to_csv(file_path)
        logging.info(f"Data for {ticker} saved to {file_path}.")
    except Exception as e:
        logging.error(f"Error saving data for {ticker}: {e}")

def main():
    # Define the tickers and time range
    tickers = ["TSLA", "BND", "SPY"]
    start_date = "2015-01-01"
    end_date = "2025-01-31"

    # Fetch and save data for each ticker
    for ticker in tickers:
        data = fetch_asset_data(ticker, start_date, end_date)
        if data is not None:
            save_data(data, ticker)

if __name__ == "__main__":
    main()