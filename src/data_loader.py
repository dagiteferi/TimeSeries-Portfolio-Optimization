import pandas as pd
import logging
import os
import sys

# Define root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create directories if they don't exist
os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)  # For storing logs

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Log level (INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler(os.path.join(ROOT_DIR, "logs", "loader.log")),  # Save logs to file
        logging.StreamHandler(sys.stdout)  # Print logs to the notebook
    ]
)

def load_data(ticker):
    """
    Load historical data for a given ticker from the data folder.
    
    Parameters:
        ticker (str): The ticker symbol (e.g., "TSLA").
    
    Returns:
        pd.DataFrame: Historical data for the ticker.
    """
    try:
        logging.info(f"Loading data for {ticker}...")
        file_path = os.path.join(ROOT_DIR, "data", f"{ticker}_data.csv")
        data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        logging.info(f"Successfully loaded data for {ticker} from {file_path}.")
        return data
    except Exception as e:
        logging.error(f"Error loading data for {ticker}: {e}")
        return None