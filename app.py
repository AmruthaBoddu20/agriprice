# Step 1: Import Libraries
from datetime import datetime

import numpy as np
# Step 2: Load the Dataset
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# Step 2: Load the Dataset
data = pd.read_csv('C:/Users/amrut/OneDrive/Documents/new_dset.csv')  # Replace with your dataset path
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)  # Convert 'Date' to datetime
data.set_index('Date', inplace=True)  # Set 'Date' as index
data.fillna(method='ffill', inplace=True)  # Forward fill for missing values

# Step 3: Fit the ARIMA Model
def fit_arima_model(data, commodity):
    train_size = int(len(data) * 0.8)  # 80% for training
    train = data[:train_size]
    model_arima = ARIMA(train[f'{commodity}_price'], order=(5, 1, 0))  # Fit ARIMA model
    model_fit = model_arima.fit()  # Fit the model
    return model_fit

# Step 4: Build and Train the LSTM Model
def build_lstm_model(data, commodity):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize scaler
    scaled_data = scaler.fit_transform(data[f'{commodity}_price'].values.reshape(-1, 1))  # Scale data

    train_size = int(len(data) * 0.8)
    train_data = scaled_data[:train_size]

    # Function to create datasets for LSTM
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Build the LSTM model
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(50, return_sequences=False))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(1))

    model_lstm.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model
    model_lstm.fit(X_train, y_train, epochs=5, batch_size=32)  # Train the model

    return model_lstm, scaler

# Step 5: Create Lagged Features
def create_lagged_features(data, commodity, lag=1):
    for i in range(1, lag + 1):
        data[f'{commodity}lag{i}'] = data[f'{commodity}_price'].shift(i)  # Create lagged features
    return data

# Step 6: Make Predictions
def make_predictions(commodity, date, centre_name, avg_temperature, avg_precipitation, model_arima, model_lstm, data, scaler):
    lagged_data = create_lagged_features(data, commodity, lag=3)  # Create lagged features
    last_prices = lagged_data[lagged_data.index < date].iloc[-1]  # Get the last available prices

    # Exclude non-numeric columns before scaling
    numerical_columns = [col for col in last_prices.index if col.startswith(f'{commodity}') and 'lag' in col] # Only include lagged features
    last_prices_numerical = last_prices[numerical_columns]

    scaled_input = scaler.transform(last_prices_numerical.values.reshape(-1, 1))  # Scale input for LSTM
    final_input = scaled_input.reshape(1, 3, 1)  # Reshape for LSTM input

    predicted_price_arima = model_arima.forecast(steps=1)  # ARIMA prediction
    predicted_price_lstm = model_lstm.predict(final_input)  # LSTM prediction

    return predicted_price_arima, scaler.inverse_transform(predicted_price_lstm)  # Return predictions

def get_predictions(commodity, date, centre_name, avg_temperature, avg_precipitation, model_arima, model_lstm, data, scaler):
    predicted_price_arima, predicted_price_lstm = make_predictions(commodity, date, centre_name, avg_temperature, avg_precipitation, model_arima, model_lstm, data, scaler)

    return {'predicted_price_arima': predicted_price_arima.values[0],
            'predicted_price_lstm': predicted_price_lstm[0][0]}

# Step 7: Train Models Before User Input
commodities = ['onion', 'potato', 'wheat']
models_arima = {}
models_lstm = {}
scalers = {}

for commodity in commodities:
    models_arima[commodity] = fit_arima_model(data, commodity)  # Fit ARIMA model
    models_lstm[commodity], scalers[commodity] = build_lstm_model(data, commodity)  # Fit LSTM model

# Step 8: Streamlit User Interface
st.title("Commodity Price Predictor")

# User Input
state = st.text_input("Enter State:")
city = st.text_input("Enter City:")
date_input = st.date_input("Select Date:", datetime.today())
commodity = st.selectbox("Select Commodity:", commodities)

# Button to Predict Price
if st.button("Predict Price"):
    # Convert date to string for processing
    date_str = date_input.strftime('%Y-%m-%d')
    
    # Assuming average temperature and precipitation are provided by the user
    avg_temperature = st.number_input("Enter Average Temperature:", value=25.0)  # Default value
    avg_precipitation = st.number_input("Enter Average Precipitation:", value=10.0)  # Default value

    # Get the trained models and scalers for the specified commodity
    model_arima = models_arima[commodity]
    model_lstm = models_lstm[commodity]
    scaler = scalers[commodity]

    # Get predictions
    predictions = get_predictions(commodity, date_str, state, avg_temperature, avg_precipitation,model_arima, model_lstm, data, scaler)

    # Display Predictions
    st.success(f"Predicted Price (ARIMA): {predictions['predicted_price_arima']:.2f}")
    st.success(f"Predicted Price (LSTM): {predictions['predicted_price_lstm']:.2f}")
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)  # Convert 'Date' to datetime
    data.set_index('Date', inplace=True)  # Set 'Date' as index
    data.fillna(method='ffill', inplace=True)  # Forward fill for missing values

# Step 3: Fit the ARIMA Model
def fit_arima_model(data, commodity):
    train_size = int(len(data) * 0.8)  # 80% for training
    train = data[:train_size]
    model_arima = ARIMA(train[f'{commodity}_price'], order=(5, 1, 0))  # Fit ARIMA model
    model_fit = model_arima.fit()  # Fit the model
    return model_fit

# Step 4: Build and Train the LSTM Model
def build_lstm_model(data, commodity):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize scaler
    scaled_data = scaler.fit_transform(data[f'{commodity}_price'].values.reshape(-1, 1))  # Scale data

    train_size = int(len(data) * 0.8)
    train_data = scaled_data[:train_size]

    # Function to create datasets for LSTM
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Build the LSTM model
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(50, return_sequences=False))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(1))

    model_lstm.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model
    model_lstm.fit(X_train, y_train, epochs=5, batch_size=32)  # Train the model

    return model_lstm, scaler

# Step 5: Create Lagged Features
def create_lagged_features(data, commodity, lag=1):
    for i in range(1, lag + 1):
        data[f'{commodity}lag{i}'] = data[f'{commodity}_price'].shift(i)  # Create lagged features
    return data

# Step 6: Make Predictions
def make_predictions(commodity, date, centre_name, avg_temperature, avg_precipitation, model_arima, model_lstm, data, scaler):
    lagged_data = create_lagged_features(data, commodity, lag=3)  # Create lagged features
    last_prices = lagged_data[lagged_data.index < date].iloc[-1]  # Get the last available prices

    # Exclude non-numeric columns before scaling
    numerical_columns = [col for col in last_prices.index if col.startswith(f'{commodity}') and 'lag' in col] # Only include lagged features
    last_prices_numerical = last_prices[numerical_columns]

    scaled_input = scaler.transform(last_prices_numerical.values.reshape(-1, 1))  # Scale input for LSTM
    final_input = scaled_input.reshape(1, 3, 1)  # Reshape for LSTM input

    predicted_price_arima = model_arima.forecast(steps=1)  # ARIMA prediction
    predicted_price_lstm = model_lstm.predict(final_input)  # LSTM prediction

    return predicted_price_arima, scaler.inverse_transform(predicted_price_lstm)  # Return predictions

def get_predictions(commodity, date, centre_name, avg_temperature, avg_precipitation, model_arima, model_lstm, data, scaler):
    predicted_price_arima, predicted_price_lstm = make_predictions(commodity, date, centre_name, avg_temperature, avg_precipitation, model_arima, model_lstm, data, scaler)

    return {'predicted_price_arima': predicted_price_arima.values[0],
            'predicted_price_lstm': predicted_price_lstm[0][0]}

# Step 7: Train Models Before User Input
commodities = ['onion', 'potato', 'wheat']
models_arima = {}
models_lstm = {}
scalers = {}

for commodity in commodities:
    models_arima[commodity] = fit_arima_model(data, commodity)  # Fit ARIMA model
    models_lstm[commodity], scalers[commodity] = build_lstm_model(data, commodity)  # Fit LSTM model

# Step 8: Streamlit User Interface
st.title("Commodity Price Predictor")

# User Input
state = st.text_input("Enter State:")
city = st.text_input("Enter City:")
date_input = st.date_input("Select Date:", datetime.today())
commodity = st.selectbox("Select Commodity:", commodities)

# Button to Predict Price
if st.button("Predict Price"):
    # Convert date to string for processing
    date_str = date_input.strftime('%Y-%m-%d')
    
    # Assuming average temperature and precipitation are provided by the user
    avg_temperature = st.number_input("Enter Average Temperature:", value=25.0)  # Default value
    avg_precipitation = st.number_input("Enter Average Precipitation:", value=10.0)  # Default value

    # Get the trained models and scalers for the specified commodity
    model_arima = models_arima[commodity]
    model_lstm = models_lstm[commodity]
    scaler = scalers[commodity]

    # Get predictions
    predictions = get_predictions(commodity, date_str, state, avg_temperature, avg_precipitation, model_arima, model_lstm, data, scaler)

    # Display Predictions
    st.success(f"Predicted Price (ARIMA): {predictions['predicted_price_arima']:.2f}")
    st.success(f"Predicted Price (LSTM): {predictions['predicted_price_lstm']:.2f}")