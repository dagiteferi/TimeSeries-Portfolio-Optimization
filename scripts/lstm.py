# lstm.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

import tensorflow as tf  
import logging
import os

# Configure logging
LOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(LOG_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "logs", "lstm_optimized.log")),
        logging.StreamHandler()
    ]
)

def preprocess_data(data, look_back=60, test_size=0.2):
    """Preprocess data with robust column checks"""
    try:
        logging.info("Starting data preprocessing")
        
        # Validate required columns
        if 'Close' not in data.columns:
            raise KeyError("DataFrame must contain 'Close' column")
        
        # Clean data
        data = data.asfreq('B').ffill()
        logging.info(f"Original data range: {data.index.min()} to {data.index.max()}")

        # Calculate returns and clean NaNs
        data['Returns'] = data['Close'].pct_change()
        data = data.dropna(subset=['Returns'])  # Remove rows with NaN returns
        
        # Split data
        train_size = int(len(data) * (1 - test_size))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        # Scale returns to [-1, 1]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_scaled = scaler.fit_transform(train_data[['Returns']])
        test_scaled = scaler.transform(test_data[['Returns']])

        # Create sequences
        def create_sequences(data, look_back):
            X, y = [], []
            for i in range(look_back, len(data)):
                X.append(data[i-look_back:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_scaled, look_back)
        X_test, y_test = create_sequences(test_scaled, look_back)

        # Reshape for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        return X_train, y_train, X_test, y_test, scaler

    except Exception as e:
        logging.error(f"Data preprocessing failed: {str(e)}")
        raise

def build_lstm_model(input_shape):
    """Construct LSTM model with TensorFlow imports"""
    model = tf.keras.Sequential([  # Explicitly use tf.keras
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    
    # Use TensorFlow's Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error'
    )
    return model

def train_lstm(model, X_train, y_train):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    ]
    
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    return model, history

def forecast_lstm(model, X_test, scaler, original_data):
    try:
        logging.info("Generating forecasts")
        
        # Generate predictions
        scaled_predictions = model.predict(X_test)
        print("Scaled predictions sample:", scaled_predictions[:5])  # Check first predictions
        
        # Inverse transform
        predictions = scaler.inverse_transform(scaled_predictions)
        print("Inverse transformed sample:", predictions[:5])
        
        # Calculate indices
        start_idx = len(original_data) - len(X_test) - 1
        end_idx = start_idx + len(predictions)
        print(f"Index range: {start_idx} to {end_idx}")
        
        # Get prices
        last_prices = original_data['Close'].iloc[start_idx:end_idx].values
        print("Last prices sample:", last_prices[:5])
        
        # Calculate final prices
        predicted_prices = last_prices * (1 + predictions.flatten())
        print("Final prices sample:", predicted_prices[:5])
        
        return predicted_prices
    except Exception as e:
        logging.error(f"Forecasting failed: {str(e)}")
        raise

def evaluate_lstm(actual, predicted):
    """Calculate metrics with NaN safety checks"""
    # Align indices and drop NaNs pairwise
    df = pd.DataFrame({'Actual': actual, 'Predicted': predicted}).dropna()
    
    if len(df) == 0:
        raise ValueError("No valid data for evaluation after NaN removal")
    
    mae = mean_absolute_error(df['Actual'], df['Predicted'])
    rmse = np.sqrt(mean_squared_error(df['Actual'], df['Predicted']))
    mape = np.mean(np.abs((df['Actual'] - df['Predicted']) / df['Actual'])) * 100
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    return mae, rmse, mape


def plot_lstm_results(actual, predicted, dates):
    """Visualize actual vs predicted prices"""
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(dates, actual, label='Actual Prices', color='blue')
        plt.plot(dates, predicted, label='Predicted Prices', color='red', linestyle='--')
        plt.title("TSLA Price Forecast vs Actual")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        logging.error(f"Plotting failed: {str(e)}")
        raise