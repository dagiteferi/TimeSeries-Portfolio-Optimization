import pandas as pd
import logging
import os

# Define root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(ROOT_DIR, "logs", "loader.log")),
        logging.StreamHandler()
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
        
        # Load the data, skipping the first 3 rows
        data = pd.read_csv(file_path, skiprows=3)
        
        # Print the number of columns for debugging
        print(f"Number of columns in {ticker}_data.csv: {len(data.columns)}")
        
        # Set the first column as the index and parse dates
        data.set_index(data.columns[0], inplace=True)
        data.index = pd.to_datetime(data.index)
        
        # Rename columns based on the number of columns
        if len(data.columns) == 5:
            data.columns = ["Price", "Close", "High", "Low", "Volume"]
        elif len(data.columns) == 6:
            data.columns = ["Price", "Close", "High", "Low", "Open", "Volume"]
        else:
            logging.error(f"Unexpected number of columns in {ticker}_data.csv: {len(data.columns)}")
            return None
        
        logging.info(f"Successfully loaded data for {ticker} from {file_path}.")
        return data
    except Exception as e:
        logging.error(f"Error loading data for {ticker}: {e}")
        return None
