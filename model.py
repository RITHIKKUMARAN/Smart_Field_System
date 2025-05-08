import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta

dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq='D')  # 2 years of daily data
np.random.seed(42)

# Simulate temperature with seasonal variation (warmer in summer, cooler in winter)
days_of_year = np.array([d.timetuple().tm_yday for d in dates])
temperature = 20 + 10 * np.sin(2 * np.pi * days_of_year / 365) + np.random.normal(0, 2, len(dates))

# Simulate humidity (higher in rainy seasons, correlated with rainfall)
humidity = 60 + 20 * np.sin(2 * np.pi * (days_of_year + 30) / 365) + np.random.normal(0, 5, len(dates))

# Simulate rainfall (more rain in monsoon season)
rainfall = np.maximum(0, 5 * np.sin(2 * np.pi * (days_of_year + 60) / 365) + np.random.exponential(2, len(dates)))

# Simulate pest risk (increases with high humidity and temperature, decreases with heavy rain)
pest_risk = np.clip(
    0.5 + 0.3 * (temperature - 20) / 10 + 0.4 * (humidity - 60) / 20 - 0.2 * (rainfall / 5),
    0, 1
) + np.random.normal(0, 0.05, len(dates))
pest_risk = np.clip(pest_risk, 0, 1)

# Create a DataFrame
data = pd.DataFrame({
    'Date': dates,
    'Temperature': temperature,
    'Humidity': humidity,
    'Rainfall': rainfall,
    'Pest Risk': pest_risk
})
data.set_index('Date', inplace=True)

# Save the dataset for use in the app (to simulate historical data)
data.to_csv('historical_weather_pest.csv')

# Step 2: Prepare the data for LSTM
features = ['Temperature', 'Humidity', 'Rainfall']
target = 'Pest Risk'
data_for_lstm = data[features + [target]].values

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_for_lstm)

# Create sequences for LSTM (use 5 days to predict the next day's pest risk)
time_steps = 5
X, y = [], []
for i in range(len(scaled_data) - time_steps):
    X.append(scaled_data[i:i + time_steps])
    y.append(scaled_data[i + time_steps, -1])  # Pest risk is the last column
X = np.array(X)
y = np.array(y)

# Step 3: Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, len(features) + 1)))
model.add(Dropout(0.2))  # Add dropout to prevent overfitting
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Step 5: Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=1)

# Step 6: Save the trained model and scaler
model.save('pest_lstm_model.h5')
joblib.dump(scaler, 'pest_scaler.pkl')

# Step 7: Evaluate the model
val_loss = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
