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
        
        # Reset the index to make Date a column and rename it
        data.reset_index(inplace=True)
        data.rename(columns={data.columns[0]: "Date"}, inplace=True)
        
        logging.info(f"Successfully loaded data for {ticker} from {file_path}.")
        return data
    except Exception as e:
        logging.error(f"Error loading data for {ticker}: {e}")
        return None
def load_tsla_data(data_path=None):
    """
    Load and verify the cleaned TSLA data from CSV, ensuring data integrity for forecasting.

    Parameters:
        data_path (str, optional): Path to TSLA_cleaned.csv. Defaults to project data directory.

    Returns:
        pd.DataFrame: Cleaned TSLA data with Date as index.

    Raises:
        Exception: If data loading or verification fails, with details logged.
    """
    try:
        # Determine default data path if not provided
        if data_path is None:
            ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(ROOT_DIR, "data", "TSLA_cleaned.csv")
            logging.info(f"Using default data path: {data_path}")

        # Load the TSLA data from CSV, setting Date as index
        logging.info("Loading TSLA data from CSV...")
        tesla_cleaned = pd.read_csv(data_path, index_col="Date", parse_dates=True)

        # Verify data integrity: check for required column and missing values
        logging.info("Verifying TSLA data integrity...")
        if "Close" not in tesla_cleaned.columns:
            raise ValueError("TSLA DataFrame missing 'Close' column")
        if tesla_cleaned.isnull().any().any():
            raise ValueError("TSLA DataFrame contains missing values")

        # Log successful verification and return data
        logging.info(f"TSLA data loaded successfully. Shape: {tesla_cleaned.shape}, "
                     f"Missing values: {tesla_cleaned.isnull().sum().sum()}")
        return tesla_cleaned

    except Exception as e:
        logging.error(f"Error loading TSLA data: {str(e)}")
        raise