# lstm_optimized.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
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

def preprocess_data(data_path, look_back=60, test_size=0.2):
    """Preprocess data with proper time-series validation and scaling"""
    try:
        logging.info("Starting data preprocessing")
        
        # Load and clean data
        data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        data = data.asfreq('B').ffill()
        logging.info(f"Original data range: {data.index.min()} to {data.index.max()}")

        # Create percentage returns instead of raw prices
        data['Returns'] = data['Close'].pct_change().dropna()
        
        # Split data before scaling
        train_size = int(len(data) * (1 - test_size))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        # Scale data to [-1, 1] range
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_scaled = scaler.fit_transform(train_data[['Returns']])
        test_scaled = scaler.transform(test_data[['Returns']])

        # Create time-step sequences
        def create_sequences(data, look_back):
            X, y = [], []
            for i in range(look_back, len(data)):
                X.append(data[i-look_back:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_scaled, look_back)
        X_test, y_test = create_sequences(test_scaled, look_back)

        # Reshape for LSTM [samples, timesteps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        logging.info(f"Training sequences: {X_train.shape[0]}")
        logging.info(f"Test sequences: {X_test.shape[0]}")
        
        return X_train, y_train, X_test, y_test, scaler, data

    except Exception as e:
        logging.error(f"Data preprocessing failed: {str(e)}")
        raise

def build_lstm_model(input_shape):
    """Construct optimized LSTM architecture"""
    try:
        logging.info("Building LSTM model")
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape,
                recurrent_dropout=0.2),
            Dropout(0.3),
            LSTM(64, recurrent_dropout=0.2),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )

        logging.info("Model summary:\n" + str(model.summary()))
        return model

    except Exception as e:
        logging.error(f"Model construction failed: {str(e)}")
        raise

def train_model(model, X_train, y_train):
    """Train model with early stopping and learning rate scheduling"""
    try:
        logging.info("Starting model training")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
        ]

        history = model.fit(
            X_train,
            y_train,
            epochs=200,
            batch_size=64,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        logging.info(f"Training stopped at epoch {len(history.history['loss'])}")
        return model, history

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

def forecast_and_evaluate(model, X_test, y_test, scaler, original_data):
    """Generate forecasts and calculate metrics"""
    try:
        logging.info("Generating forecasts")
        
        # Generate predictions
        scaled_predictions = model.predict(X_test)
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(scaled_predictions)
        actual_returns = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Convert returns back to prices
        last_prices = original_data['Close'].iloc[-len(predictions)-1:-1].values
        predicted_prices = last_prices * (1 + predictions.flatten())
        actual_prices = original_data['Close'].iloc[-len(predictions):].values

        # Calculate metrics
        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

        logging.info(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return predicted_prices, actual_prices

    except Exception as e:
        logging.error(f"Forecasting failed: {str(e)}")
        raise

def plot_results(actual, predicted, dates):
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
        logging.info("Plot generated successfully")
    except Exception as e:
        logging.error(f"Plotting failed: {str(e)}")
        raise

