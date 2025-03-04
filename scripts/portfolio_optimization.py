import os
import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
import matplotlib.pyplot as plt

DATA_DIR = "c:/Users/HP/Documents/Dagii/week-11/TimeSeries-Portfolio-Optimization/data"

def load_historical_data(ticker, file_name=None, skiprows=3):
    """
    Load historical data for a given ticker from the data folder.
    
    Parameters:
        ticker (str): The ticker symbol (e.g., "TSLA").
        file_name (str, optional): Custom file name for the ticker. Defaults to None.
        skiprows (int, optional): Number of rows to skip in the CSV file. Defaults to 3.
    
    Returns:
        pd.DataFrame: Historical data for the ticker.
    """
    try:
        logging.info(f"Loading data for {ticker}...")

        if file_name:
            file_path = os.path.join(DATA_DIR, file_name)
        else:
            file_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")

        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return None

        data = pd.read_csv(file_path, skiprows=skiprows)
        print(f"Number of columns in {ticker}_data.csv: {len(data.columns)}")

        data.set_index(data.columns[0], inplace=True)
        data.index = pd.to_datetime(data.index)

        if len(data.columns) == 5:
            data.columns = ["Price", "Close", "High", "Low", "Volume"]
        elif len(data.columns) == 6:
            data.columns = ["Price", "Close", "High", "Low", "Open", "Volume"]
        elif len(data.columns) == 9:
            data.columns = ["Date", "Price", "Close", "High", "Low", "Volume", 
                            "Daily_Return", "Rolling_Mean", "Rolling_Std", "Z_Score"]
        else:
            logging.error(f"Unexpected number of columns in {ticker}_data.csv: {len(data.columns)}")
            return None

        data.reset_index(inplace=True)
        data.rename(columns={data.columns[0]: "Date"}, inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        logging.info(f"Successfully loaded data for {ticker} from {file_path}.")
        return data
    except Exception as e:
        logging.error(f"Error loading data for {ticker}: {e}")
        return None

def forecast_prices(data, tsla_forecast, bnd_growth_factor=1.5, spy_growth_factor=1.0):
    """
    Forecast prices for BND and SPY using historical average returns with adjustable growth factors.
    
    Args:
        data (DataFrame): Historical prices with a DatetimeIndex.
        tsla_forecast (array): Forecasted TSLA prices from Task 3.
        bnd_growth_factor (float): Multiplier for BND growth rate (default: 1.5).
        spy_growth_factor (float): Multiplier for SPY growth rate (default: 1.0).
    
    Returns:
        forecast_df (DataFrame): Forecasted prices for all assets.
    """
    logging.info("Forecasting prices for BND and SPY...")
    try:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("The input DataFrame must have a DatetimeIndex.")

        bnd_avg_return = data['BND'].pct_change().mean() * bnd_growth_factor
        spy_avg_return = data['SPY'].pct_change().mean() * spy_growth_factor
        
        last_bnd_price = data['BND'].iloc[-1]
        last_spy_price = data['SPY'].iloc[-1]
        
        bnd_forecast = [last_bnd_price * (1 + bnd_avg_return) ** i for i in range(1, 253)]
        spy_forecast = [last_spy_price * (1 + spy_avg_return) ** i for i in range(1, 253)]
        
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

def optimize_portfolio(forecast_df, min_allocation=0.1, concentration_penalty=0.05):
    """
    Optimize portfolio weights to maximize the Sharpe Ratio with constraints.
    
    Args:
        forecast_df (DataFrame): Forecasted prices for all assets.
        min_allocation (float): Minimum allocation for each asset (default: 10%).
        concentration_penalty (float): Penalty for over-concentration (default: 0.05).
    
    Returns:
        optimal_weights (array), portfolio_return (float), portfolio_volatility (float), 
        sharpe_ratio (float), var_95 (float)
    """
    logging.info("Optimizing portfolio weights...")
    try:
        returns = forecast_df.pct_change().dropna()
        annual_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        print("Annualized Returns:\n", annual_returns)
        print("Covariance Matrix:\n", cov_matrix)
        
        def negative_sharpe(weights, returns, cov_matrix):
            portfolio_return = np.dot(weights, returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = portfolio_return / portfolio_volatility
            concentration = np.sum(weights**2)  # Penalize high concentration
            return -sharpe + concentration_penalty * concentration
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((min_allocation, 1) for _ in range(3))  # Minimum allocation constraint
        initial_guess = [0.33, 0.33, 0.33]
        
        result = minimize(
            negative_sharpe,
            initial_guess,
            args=(annual_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        optimal_weights = result.x
        
        portfolio_return = np.dot(optimal_weights, annual_returns)
        portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility
        
        portfolio_returns = returns.dot(optimal_weights)
        var_95 = np.percentile(portfolio_returns, 5)
        
        logging.info("Portfolio optimization complete.")
        return optimal_weights, portfolio_return, portfolio_volatility, sharpe_ratio, var_95
    except Exception as e:
        logging.error(f"Error optimizing portfolio: {e}")
        raise

def plot_portfolio_performance(forecast_df, optimal_weights):
    returns = forecast_df.pct_change().dropna()
    portfolio_returns = returns.dot(optimal_weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Calculate cumulative returns for individual assets
    tsla_cumulative = (1 + returns['TSLA']).cumprod()
    bnd_cumulative = (1 + returns['BND']).cumprod()
    spy_cumulative = (1 + returns['SPY']).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns, label='Portfolio Cumulative Returns', linewidth=2)
    plt.plot(tsla_cumulative, label='TSLA Cumulative Returns', linestyle='--', alpha=0.5)
    plt.plot(bnd_cumulative, label='BND Cumulative Returns', linestyle='--', alpha=0.5)
    plt.plot(spy_cumulative, label='SPY Cumulative Returns', linestyle='--', alpha=0.5)
    plt.title('Portfolio and Asset Performance Based on Forecasted Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

