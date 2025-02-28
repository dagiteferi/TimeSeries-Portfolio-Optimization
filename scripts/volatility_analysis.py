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
    """
    try:
        logging.info(f"Analyzing volatility for {ticker} with a {window}-day rolling window...")
        
        # Calculate rolling mean and rolling standard deviation
        data["Rolling_Mean"] = data["Close"].rolling(window=window).mean()
        data["Rolling_Std"] = data["Close"].rolling(window=window).std()
        
        # Drop rows with NaN values
        data_cleaned = data.dropna()
        
        # Plot the results
        logging.info(f"Plotting rolling statistics for {ticker}...")
        plt.figure(figsize=(12, 6))
        plt.plot(data_cleaned.index, data_cleaned["Close"], label="Closing Price", color="blue", alpha=0.7)
        plt.plot(data_cleaned.index, data_cleaned["Rolling_Mean"], label=f"{window}-Day Rolling Mean", color="red")
        plt.plot(data_cleaned.index, data_cleaned["Rolling_Std"], label=f"{window}-Day Rolling Std", color="orange")
        plt.title(f"{ticker} Volatility Analysis (Rolling Statistics)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        plt.show()
        
        logging.info(f"Volatility analysis completed for {ticker}.")
        return data_cleaned  # Return the cleaned dataset
    except Exception as e:
        logging.error(f"Error analyzing volatility for {ticker}: {e}")
        return None
def calculate_var_sharpe(data, ticker, confidence_level=0.95, rf_rate=0.02):
    """
    Calculate Value at Risk (VaR) and Sharpe Ratio for a given asset.
    
    Parameters:
        data (pd.DataFrame): The dataset with a 'Close' column and Date as index.
        ticker (str): The ticker symbol (e.g., "TSLA").
        confidence_level (float): Confidence level for VaR (default 0.95).
        rf_rate (float): Risk-free rate (default 0.02 or 2%).
    
    Returns:
        tuple: (VaR as percentage, Sharpe Ratio as float) or (None, None) if error occurs
    """
    try:
        logging.info(f"Calculating VaR and Sharpe Ratio for {ticker}...")
        
        # Ensure data has a 'Close' column and is a DataFrame
        if not isinstance(data, pd.DataFrame) or "Close" not in data.columns:
            raise ValueError(f"Data must be a DataFrame with a 'Close' column for {ticker}")
        
        # Calculate daily returns
        daily_returns = data["Close"].pct_change().dropna()
        if daily_returns.empty:
            raise ValueError(f"No valid daily returns calculated for {ticker}")
        
        # Calculate VaR (historical, percentile method)
        var = np.percentile(daily_returns, (1 - confidence_level) * 100) * 100  # Convert to percentage
        logging.info(f"VaR at {confidence_level*100}% confidence for {ticker}: {var:.2f}%")
        
        # Calculate Sharpe Ratio (annualized)
        mean_return = daily_returns.mean() * 252  # Annualize daily mean return
        std_return = daily_returns.std() * np.sqrt(252)  # Annualize daily standard deviation
        if std_return == 0:  # Avoid division by zero
            sharpe = 0
        else:
            sharpe = (mean_return - rf_rate) / std_return
        logging.info(f"Sharpe Ratio for {ticker}: {sharpe:.2f}")
        
        return var, sharpe
    except Exception as e:
        logging.error(f"Error calculating VaR and Sharpe Ratio for {ticker}: {e}")
        return None, None