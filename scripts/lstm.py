# lstm.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
        logging.FileHandler(os.path.join(LOG_DIR, "logs", "lstm.log")),
        logging.StreamHandler()
    ]
)

def preprocess_data(data, look_back=60):
    """
    Prepare LSTM-ready data with time steps
    Returns scaled data, scaler, and reshaped datasets
    """
    try:
        logging.info("Preprocessing data for LSTM")
        
        # 1. Select and scale Close prices
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['Close']])
        
        # 2. Create time-step sequences
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
            
        X = np.array(X)
        y = np.array(y)
        
        # 3. Reshape for LSTM input [samples, timesteps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        logging.info(f"Created {len(X)} sequences with {look_back} time steps")
        return X, y, scaler
    
    except Exception as e:
        logging.error(f"Data preprocessing failed: {str(e)}")
        raise

def build_lstm_model(input_shape, units=50, dropout=0.2):
    """
    Construct LSTM model architecture
    Returns compiled Keras model
    """
    try:
        logging.info("Building LSTM model")
        
        model = Sequential([
            LSTM(units=units, return_sequences=True, input_shape=input_shape),
            LSTM(units=units, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        logging.info("Model summary:\n" + str(model.summary()))
        return model
    
    except Exception as e:
        logging.error(f"Model construction failed: {str(e)}")
        raise

def train_lstm(model, X_train, y_train, epochs=100, batch_size=32):
    """
    Train LSTM model with early stopping
    Returns trained model and training history
    """
    try:
        logging.info("Starting model training")
        
        early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        logging.info(f"Training stopped at epoch {len(history.history['loss'])}")
        return model, history
    
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

def forecast_lstm(model, data, scaler, look_back=60):
    """
    Generate forecasts using trained LSTM model
    Returns inverse-transformed predictions
    """
    try:
        logging.info("Generating forecasts")
        
        # 1. Get last sequence from training data
        inputs = data[-look_back:]
        inputs = scaler.transform(inputs)
        
        # 2. Generate multi-step forecast
        predictions = []
        current_batch = inputs.reshape((1, look_back, 1))
        
        for _ in range(len(data) - look_back):
            current_pred = model.predict(current_batch)[0]
            predictions.append(current_pred)
            current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)
            
        # 3. Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        logging.info(f"Generated {len(predictions)} forecasts")
        return predictions.flatten()
    
    except Exception as e:
        logging.error(f"Forecasting failed: {str(e)}")
        raise

def evaluate_lstm(actual, predictions):
    """
    Calculate and log evaluation metrics
    Returns MAE, RMSE, MAPE tuple
    """
    try:
        logging.info("Evaluating LSTM performance")
        
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        logging.info(f"LSTM Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        return mae, rmse, mape
    
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise

def plot_lstm_results(actual, predictions, title="LSTM Forecast vs Actual"):
    """
    Generate comparison plot
    """
    try:
        logging.info("Generating results plot")
        
        plt.figure(figsize=(12, 6))
        plt.plot(actual.index, actual, label='Actual Price', color='blue')
        plt.plot(actual.index[len(actual)-len(predictions):], predictions, 
                 label='LSTM Forecast', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        logging.info("Plot generated successfully")
    
    except Exception as e:
        logging.error(f"Plotting failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logging.info("LSTM Forecasting Pipeline Started")
        
        # 1. Load data
        data = pd.read_csv("TSLA_cleaned.csv", index_col="Date", parse_dates=True)
        data = data.asfreq('B').ffill()
        
        # 2. Split data
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # 3. Preprocess
        X_train, y_train, scaler = preprocess_data(train_data, look_back=60)
        
        # 4. Build model
        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # 5. Train model
        trained_model, history = train_lstm(model, X_train, y_train)
        
        # 6. Generate forecasts
        predictions = forecast_lstm(trained_model, train_data[['Close']], scaler)
        
        # 7. Prepare test data
        valid = test_data.copy()
        valid['Predictions'] = np.nan
        valid.iloc[-len(predictions):, -1] = predictions[-len(valid):]
        
        # 8. Evaluate & plot
        evaluate_lstm(valid['Close'], valid['Predictions'].dropna())
        plot_lstm_results(valid['Close'], valid['Predictions'].dropna())
        
        logging.info("Forecast Sample:")
        print(valid[['Close', 'Predictions']].tail())

    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        raise