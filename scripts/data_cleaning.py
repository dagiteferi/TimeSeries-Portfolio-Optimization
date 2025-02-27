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

        # Step 1: Check basic statistics
        logging.info(f"Basic statistics for {ticker}:")
        logging.info(data.describe())

        # Step 2: Ensure all columns have appropriate data types
        logging.info(f"Data types before cleaning for {ticker}:")
        logging.info(data.dtypes)

        # Convert 'Date' column to datetime if it exists
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"])
            logging.info(f"'Date' column converted to datetime for {ticker}.")

        # Step 3: Check for missing values
        logging.info(f"Missing values before handling for {ticker}:")
        logging.info(data.isnull().sum())

        # Handle missing values
        data.fillna(method="ffill", inplace=True)  # Forward fill missing values
        logging.info(f"Missing values after handling for {ticker}:")
        logging.info(data.isnull().sum())

        # Step 4: Normalize or scale the data (if required)
        # Normalize the 'Close' column for machine learning models
        if "Close" in data.columns:
            scaler = MinMaxScaler()
            data["Close_Normalized"] = scaler.fit_transform(data[["Close"]])
            logging.info(f"'Close' column normalized for {ticker}.")
        else:
            logging.warning(f"'Close' column not found in data for {ticker}. Skipping normalization.")

        logging.info(f"Data cleaning and understanding completed for {ticker}.")
        return data

    except Exception as e:
        logging.error(f"Error cleaning and analyzing data for {ticker}: {e}")
        return None