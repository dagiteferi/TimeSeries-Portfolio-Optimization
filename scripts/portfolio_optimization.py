"""
portfolio_optimization.py
This script optimizes a portfolio of TSLA, BND, and SPY based on forecasted prices.
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Define the root directory (adjust as needed)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_historical_data(ticker):
    """
    Load historical data for a given ticker from the data folder.
    
    Parameters:
        ticker (str): The ticker symbol (e.g., "TSLA").
    
    Returns:
        pd.DataFrame: Historical data for the ticker.
    """
    try:
        logging.info(f"Loading data for {ticker}...")
        
        # Construct the file path
        if ticker == "TSLA":
            file_path = os.path.join(ROOT_DIR, "data", "cleaned_tesla.csv")  # Use cleaned Tesla data
        else:
            file_path = os.path.join(ROOT_DIR, "data", f"{ticker}_data.csv")
        
        # Load the data
        data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        
        # Rename columns for consistency
        if 'Close' not in data.columns:
            data.rename(columns={'Adj Close': 'Close'}, inplace=True)
        
        logging.info(f"Successfully loaded data for {ticker} from {file_path}.")
        return data
    except Exception as e:
        logging.error(f"Error loading data for {ticker}: {e}")
        return None

def forecast_prices(data, tsla_forecast):
    """
    Forecast prices for BND and SPY using historical average returns.
    
    Args:
        data (DataFrame): Historical prices.
        tsla_forecast (array): Forecasted TSLA prices from Task 3.
    
    Returns:
        forecast_df (DataFrame): Forecasted prices for all assets.
    """
    logging.info("Forecasting prices for BND and SPY...")
    try:
        # Calculate historical average daily returns
        bnd_avg_return = data['BND'].pct_change().mean()
        spy_avg_return = data['SPY'].pct_change().mean()
        
        # Forecast BND and SPY prices
        last_bnd_price = data['BND'].iloc[-1]
        last_spy_price = data['SPY'].iloc[-1]
        
        bnd_forecast = [last_bnd_price * (1 + bnd_avg_return) ** i for i in range(1, 253)]
        spy_forecast = [last_spy_price * (1 + spy_avg_return) ** i for i in range(1, 253)]
        
        # Combine forecasts into a DataFrame
        forecast_df = pd.DataFrame({
            'TSLA': tsla_forecast,
            'BND': bnd_forecast,
            'SPY': spy_forecast
        }, index=pd.date_range(data.index[-1] + pd.offsets.BDay(1), periods=252, freq='B'))
        
        logging.info("Price forecasting complete.")
        return forecast_df
    except Exception as e:
        logging.error(f"Error forecasting prices: {e}")
        raise

def optimize_portfolio(forecast_df):
    """
    Optimize portfolio weights to maximize the Sharpe Ratio.
    
    Args:
        forecast_df (DataFrame): Forecasted prices for all assets.
    
    Returns:
        optimal_weights (array): Optimized portfolio weights.
        portfolio_return (float): Expected annual return.
        portfolio_volatility (float): Annual volatility.
        sharpe_ratio (float): Risk-adjusted return.
        var_95 (float): Value at Risk (95% confidence).
    """
    logging.info("Optimizing portfolio weights...")
    try:
        # Calculate daily returns
        returns = forecast_df.pct_change().dropna()
        
        # Annualized returns and covariance matrix
        annual_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Define negative Sharpe Ratio (to maximize)
        def negative_sharpe(weights, returns, cov_matrix):
            portfolio_return = np.dot(weights, returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -portfolio_return / portfolio_volatility
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(3))
        initial_guess = [0.33, 0.33, 0.33]
        
        # Optimize
        result = minimize(
            negative_sharpe,
            initial_guess,
            args=(annual_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        optimal_weights = result.x
        
        # Portfolio metrics
        portfolio_return = np.dot(optimal_weights, annual_returns)
        portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility
        
        # Value at Risk (95% confidence)
        portfolio_returns = returns.dot(optimal_weights)
        var_95 = np.percentile(portfolio_returns, 5)
        
        logging.info("Portfolio optimization complete.")
        return optimal_weights, portfolio_return, portfolio_volatility, sharpe_ratio, var_95
    except Exception as e:
        logging.error(f"Error optimizing portfolio: {e}")
        raise

def plot_portfolio_performance(forecast_df, optimal_weights):
    """
    Plot cumulative portfolio returns.
    
    Args:
        forecast_df (DataFrame): Forecasted prices.
        optimal_weights (array): Optimized portfolio weights.
    """
    import matplotlib.pyplot as plt
    
    try:
        # Calculate cumulative returns
        returns = forecast_df.pct_change().dropna()
        portfolio_returns = returns.dot(optimal_weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns, label='Portfolio Cumulative Returns')
        plt.title('Portfolio Performance Based on Forecasted Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting portfolio performance: {e}")
        raise