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

def check_basic_statistics(data, ticker):
    """
    Check basic statistics of the data.
    
    Parameters:
        data (pd.DataFrame): The data to analyze.
        ticker (str): The ticker symbol (e.g., "TSLA").
    """
    try:
        logging.info(f"Basic statistics for {ticker}:")
        logging.info(data.describe())
        return data.describe()
    except Exception as e:
        logging.error(f"Error checking basic statistics for {ticker}: {e}")
        return None

def ensure_data_types(data, ticker):
    """
    Ensure all columns have appropriate data types.
    
    Parameters:
        data (pd.DataFrame): The data to clean.
        ticker (str): The ticker symbol (e.g., "TSLA").
    """
    try:
        logging.info(f"Data types before cleaning for {ticker}:")
        logging.info(data.dtypes)

        # Convert 'Date' column to datetime if it exists
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"])
            logging.info(f"'Date' column converted to datetime for {ticker}.")

        logging.info(f"Data types after cleaning for {ticker}:")
        logging.info(data.dtypes)
        return data
    except Exception as e:
        logging.error(f"Error ensuring data types for {ticker}: {e}")
        return None

def check_missing_values(data, ticker):
    """
    Check for missing values in the data.
    
    Parameters:
        data (pd.DataFrame): The data to analyze.
        ticker (str): The ticker symbol (e.g., "TSLA").
    """
    try:
        logging.info(f"Missing values for {ticker}:")
        missing_values = data.isnull().sum()
        logging.info(missing_values)
        return missing_values
    except Exception as e:
        logging.error(f"Error checking missing values for {ticker}: {e}")
        return None

# def handle_missing_values(data, ticker, method="ffill"):
#     """
#     Handle missing values in the data.
    
#     Parameters:
#         data (pd.DataFrame): The data to clean.
#         ticker (str): The ticker symbol (e.g., "TSLA").
#         method (str): Method to handle missing values ("ffill", "bfill", "drop", etc.).
#     """
#     try:
#         logging.info(f"Handling missing values for {ticker} using method: {method}...")
#         if method == "ffill":
#             data.fillna(method="ffill", inplace=True)
#         elif method == "bfill":
#             data.fillna(method="bfill", inplace=True)
#         elif method == "drop":
#             data.dropna(inplace=True)
#         else:
#             logging.warning(f"Unsupported method: {method}. Using 'ffill' by default.")
#             data.fillna(method="ffill", inplace=True)

#         logging.info(f"Missing values after handling for {ticker}:")
#         logging.info(data.isnull().sum())
#         return data
#     except Exception as e:
#         logging.error(f"Error handling missing values for {ticker}: {e}")
#         return None

def normalize_data(data, ticker, column="Close"):
    """
    Normalize or scale a specific column in the data.
    
    Parameters:
        data (pd.DataFrame): The data to normalize.
        ticker (str): The ticker symbol (e.g., "TSLA").
        column (str): The column to normalize (default is "Close").
    """
    try:
        if column in data.columns:
            scaler = MinMaxScaler()
            data[f"{column}_Normalized"] = scaler.fit_transform(data[[column]])
            logging.info(f"'{column}' column normalized for {ticker}.")
        else:
            logging.warning(f"'{column}' column not found in data for {ticker}. Skipping normalization.")
        return data
    except Exception as e:
        logging.error(f"Error normalizing data for {ticker}: {e}")
        return None