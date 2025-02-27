import pandas as pd
import logging
import os
from sklearn.preprocessing import MinMaxScaler

# Define root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create directories if they don't exist
os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)  # For storing logs

# Set up logging
logging.basicConfig(
    filename=os.path.join(ROOT_DIR, "logs", "cleaning.log"),  # Log file path
    level=logging.INFO,             # Log level (INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    filemode="w"                    # Overwrite the log file each time
)

def clean_and_understand_data(data, ticker):
    """
    Clean and understand the data for a given ticker.
    
    Parameters:
        data (pd.DataFrame): The data to clean and analyze.
        ticker (str): The ticker symbol (e.g., "TSLA").
    
    Returns:
        pd.DataFrame: Cleaned and normalized data.
    """
    try:
        logging.info(f"Cleaning and analyzing data for {ticker}...")

        # Check basic statistics
        logging.info(f"Basic statistics for {ticker}:")
        logging.info(data.describe())

        # Check data types and missing values
        logging.info(f"Data types and missing values for {ticker}:")
        logging.info(data.info())

        # Handle missing values
        data.fillna(method="ffill", inplace=True)  # Forward fill missing values
        logging.info(f"Missing values after handling for {ticker}:")
        logging.info(data.isnull().sum())

        # Normalize or scale the data
        scaler = MinMaxScaler()
        data["Close_Normalized"] = scaler.fit_transform(data[["Close"]])
        logging.info(f"Data normalized for {ticker}.")

        logging.info(f"Data cleaning and understanding completed for {ticker}.")
        return data

    except Exception as e:
        logging.error(f"Error cleaning and analyzing data for {ticker}: {e}")
        return None