import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import requests
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import joblib

# Enhanced CSS with more animations, gradients, and emoji styling
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    margin: 0;
    padding: 0;
    height: 100vh;
    overflow: auto;
}
.stApp {
    max-width: 100% !important;
    padding: 20px;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    animation: fadeIn 1s ease-in;
}
@keyframes fadeIn {
    0% { opacity: 0; transform: scale(0.9); }
    100% { opacity: 1; transform: scale(1); }
}
.welcome-banner {
    background: linear-gradient(45deg, #28a745, #ffeb3b);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: white;
    font-size: 1.5em;
    font-weight: bold;
    animation: slideDown 1s ease-in-out;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}
@keyframes slideDown {
    0% { opacity: 0; transform: translateY(-20px); }
    100% { opacity: 1; transform: translateY(0); }
}
.card {
    background: rgba(255, 255, 255, 0.95);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    margin-bottom: 25px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border: 2px solid #28a745;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    border-color: #ffeb3b;
}
.alert {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
    animation: slideIn 0.5s ease, pulse 2s infinite;
    font-weight: bold;
}
@keyframes slideIn {
    0% { opacity: 0; transform: translateX(-20px); }
    100% { opacity: 1; transform: translateX(0); }
}
@keyframes pulse {
    0% { box-shadow: 0 0 5px rgba(0, 0, 0, 0.1); }
    50% { box-shadow: 0 0 15px rgba(0, 255, 0, 0.5); }
    100% { box-shadow: 0 0 5px rgba(0, 0, 0, 0.1); }
}
.alert-warning { background-color: #fff3cd; color: #856404; }
.alert-info { background-color: #d1ecf1; color: #0c5460; }
.alert-success { background-color: #d4edda; color: #155724; }
.stButton>button {
    background: linear-gradient(45deg, #ff6f61, #6b48ff);
    color: white;
    border: none;
    padding: 12px 25px;
    border-radius: 25px;
    font-weight: bold;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(45deg, #6b48ff, #ff6f61);
    transform: scale(1.05);
    box-shadow: 0 5px 15px rgba(107, 72, 255, 0.4);
}
.weather-icon {
    width: 60px;
    height: 60px;
    animation: bounce 1.5s infinite;
}
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}
.progress-bar {
    background-color: #e0e0e0;
    border-radius: 10px;
    height: 25px;
    width: 100%;
    overflow: hidden;
}
.progress {
    background: linear-gradient(45deg, #ff6f61, #6b48ff);
    height: 100%;
    border-radius: 10px;
    text-align: center;
    color: white;
    font-weight: bold;
    animation: fillProgress 1s ease-in-out;
}
@keyframes fillProgress {
    0% { width: 0%; }
    100% { width: inherit; }
}
h1 { color: #ffffff; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); }
h2 { color: #2a5298; animation: fadeInText 1.5s ease-in; }
@keyframes fadeInText {
    0% { opacity: 0; }
    100% { opacity: 1; }
}
.fancy-divider {
    height: 3px;
    background: linear-gradient(to right, #ff6f61, #6b48ff);
    border: none;
    margin: 20px 0;
    border-radius: 2px;
    animation: gradientShift 3s infinite;
}
@keyframes gradientShift {
    0% { background-position: 0%; }
    100% { background-position: 200%; }
}
.weather-animation {
    animation: rotate 2s linear infinite;
}
@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.badge {
    display: inline-block;
    padding: 5px 10px;
    background: #28a745;
    color: white;
    border-radius: 15px;
    margin: 5px;
    font-size: 0.9em;
    animation: pop 0.5s ease;
}
@keyframes pop {
    0% { transform: scale(0); }
    80% { transform: scale(1.2); }
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# Adding a JavaScript snippet for confetti animation when badges are earned
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
<script>
function triggerConfetti() {
    confetti({
        particleCount: 100,
        spread: 70,
        origin: { y: 0.6 }
    });
}
</script>
""", unsafe_allow_html=True)

# Load the pre-trained LSTM model and scaler
try:
    pest_model = load_model('pest_lstm_model.h5')
    pest_scaler = joblib.load('pest_scaler.pkl')
except Exception as e:
    st.error(f"Error loading the pest prediction model: {e}")
    pest_model = None
    pest_scaler = None


# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'historical_water' not in st.session_state:
    st.session_state.historical_water = []
if 'points' not in st.session_state:
    st.session_state.points = 0
if 'badges' not in st.session_state:
    st.session_state.badges = []
if 'selected_crop' not in st.session_state:
    st.session_state.selected_crop = None
if 'live_moisture' not in st.session_state:
    st.session_state.live_moisture = 0.0
if 'moisture_history' not in st.session_state:
    st.session_state.moisture_history = []
if 'last_moisture_update' not in st.session_state:
    st.session_state.last_moisture_update = 0
if 'new_badge' not in st.session_state:
    st.session_state.new_badge = False
if 'energy_usage' not in st.session_state:
    st.session_state.energy_usage = []  # To store historical energy usage

def load_sensor_data():
    try:
        with open("sensor_data.json", 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def fetch_coordinates(city, api_key):
    geocoding_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
    try:
        response = requests.get(geocoding_url)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]['lat'], data[0]['lon']
        else:
            st.error(f"City '{city}' not found in OpenWeatherMap database. 📍")
            return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching coordinates for {city}: {str(e)} 📍")
        return None, None

def fetch_current_weather(city, api_key):
    lat, lon = fetch_coordinates(city, api_key)
    if lat is None or lon is None:
        return None

    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    air_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    
    try:
        weather_response = requests.get(weather_url)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        
        air_response = requests.get(air_url)
        air_response.raise_for_status()
        air_data = air_response.json()
        
        temp = round(weather_data['main']['temp'], 1)
        humidity = round(weather_data['main']['humidity'], 1)
        rain_chance = weather_data.get('rain', {}).get('1h', 0) * 100
        icon = weather_data['weather'][0]['icon']
        aqi = air_data['list'][0]['main']['aqi']
        aqi_label = {1: "Good 🌟", 2: "Fair 🌼", 3: "Moderate ⚠️", 4: "Poor 😷", 5: "Very Poor 🚨"}.get(aqi, "Unknown ❓")
        
        return {
            'temp': temp,
            'humidity': humidity,
            'rain_chance': rain_chance,
            'icon': icon,
            'aqi': aqi_label,
            'aqi_value': aqi
        }
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error fetching weather data: {http_err.response.status_code} - {http_err.response.text} 🌦️")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {str(e)} 🌦️")
        return None

def fetch_weather_forecast(city, api_key):
    lat, lon = fetch_coordinates(city, api_key)
    if lat is None or lon is None:
        return []

    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        forecast_data = []
        current_date = datetime.utcnow().date()
        for i in range(1, 4):
            target_date = current_date + timedelta(days=i)
            daily_data = [entry for entry in data['list'] if datetime.fromtimestamp(entry['dt']).date() == target_date]
            if daily_data:
                avg_temp = np.mean([entry['main']['temp'] for entry in daily_data])
                avg_humidity = np.mean([entry['main']['humidity'] for entry in daily_data])
                avg_pop = np.mean([entry.get('pop', 0) for entry in daily_data]) * 100
                rain_predicted = avg_pop > 30
                icon = daily_data[0]['weather'][0]['icon']
                forecast_data.append({
                    'date': target_date.strftime("%Y-%m-%d"),
                    'temp': round(avg_temp, 1),
                    'humidity': round(avg_humidity, 1),
                    'rain_predicted': rain_predicted,
                    'icon': icon,
                    'pop': avg_pop
                })
        return forecast_data
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error fetching forecast data: {http_err.response.status_code} - {http_err.response.text} 🌧️")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching forecast data: {str(e)} 🌧️")
        return []

soil_adjustments = {
    "Red Soil": 1.20,
    "Black Soil": 0.85,
    "Alluvial Soil": 1.0,
    "Laterite Soil": 1.10,
    "Sandy Soil": 1.30
}

irrigation_efficiency = {
    "Red Soil": {"Drip": 0.90, "Flood": 0.60},
    "Black Soil": {"Drip": 0.85, "Flood": 0.70},
    "Alluvial Soil": {"Drip": 0.88, "Flood": 0.65},
    "Laterite Soil": {"Drip": 0.87, "Flood": 0.62},
    "Sandy Soil": {"Drip": 0.92, "Flood": 0.55}
}

crop_water_needs = {
    "Wheat": 800,
    "Rice": 1500,
    "Maize": 1000,
    "Soybean": 900
}

crop_details = {
    "Wheat": {
        "temp_range": (15, 25),
        "humidity_range": (50, 70),
        "water_need": 800,
        "tip": "Plant in well-drained soil, water early morning. Avoid overwatering during rainy seasons! 🌱"
    },
    "Rice": {
        "temp_range": (20, 30),
        "humidity_range": (70, 90),
        "water_need": 1500,
        "tip": "Requires flooded fields; ensure consistent water supply and high humidity. 💧"
    },
    "Maize": {
        "temp_range": (20, 30),
        "humidity_range": (50, 70),
        "water_need": 1000,
        "tip": "Thrives in warm weather; water deeply but allow soil to dry between sessions. 🌽"
    },
    "Soybean": {
        "temp_range": (20, 30),
        "humidity_range": (50, 70),
        "water_need": 900,
        "tip": "Prefers moderate moisture; rotate crops to maintain soil health. 🌾"
    }
}

def calculate_water_requirement(temperature, humidity, rain_predicted, acres, soil_type, crop, live_moisture, min_moisture, max_moisture):
    base_water_per_acre = crop_water_needs.get(crop, 1000)
    if live_moisture > max_moisture:
        base_water_per_acre *= 0.7
    elif live_moisture < min_moisture:
        base_water_per_acre *= 1.3
    if temperature > 25:
        temp_adjustment = ((temperature - 25) // 5) * 0.1
        base_water_per_acre *= (1 + temp_adjustment)
    if humidity < 50:
        humidity_adjustment = ((50 - humidity) // 10) * 0.05
        base_water_per_acre *= (1 + humidity_adjustment)
    if rain_predicted:
        base_water_per_acre *= 0.5
    soil_factor = soil_adjustments.get(soil_type, 1.0)
    base_water_per_acre *= soil_factor
    total_water = base_water_per_acre * acres
    return round(total_water, 2)

def schedule_irrigation(temperature, humidity, forecast_data):
    # Base irrigation time on current conditions
    if temperature > 30:
        base_time = "Early Morning (6 AM) 🌅"
    elif humidity < 40:
        base_time = "Evening (6 PM) 🌙"
    else:
        base_time = "Mid-Morning (9 AM) ☀️"
    
    # Adjust based on forecast (next 3 days)
    schedule = []
    for day in forecast_data:
        date = day['date']
        temp = day['temp']
        hum = day['humidity']
        pop = day['pop']
        
        # Adjust time based on forecast conditions
        if pop > 50:  # High chance of rain
            scheduled_time = "Skip (Rain Expected) 🌧️"
        elif temp > 30:
            scheduled_time = "Early Morning (6 AM) 🌅"
        elif hum < 40:
            scheduled_time = "Evening (6 PM) 🌙"
        else:
            scheduled_time = "Mid-Morning (9 AM) ☀️"
        
        schedule.append({"date": date, "time": scheduled_time})
    
    return base_time, schedule

def calculate_irrigation_cost_and_energy(water_needed, acres, soil_type, irrigation_method="Drip", water_price=0.01, energy_rate=7.0, regional_factor=1.0, pump_power=5.0):
    base_cost = water_needed * water_price
    efficiency = irrigation_efficiency.get(soil_type, {"Drip": 0.90}).get(irrigation_method, 0.90)
    effective_water = water_needed / efficiency
    
    # Energy consumption calculation
    # Assume pump power in kW, runtime in hours = (liters / flow rate), flow rate = 1000 liters/hour
    flow_rate = 1000  # liters per hour
    runtime_hours = effective_water / flow_rate  # hours
    energy_consumed = pump_power * runtime_hours  # kWh
    energy_cost = energy_consumed * energy_rate
    
    crop_factor = {"Rice": 1.2, "Maize": 1.0, "Wheat": 0.9, "Soybean": 0.95}.get(recommended_crop, 1.0)
    operational_cost = effective_water * water_price * crop_factor
    maintenance_cost = acres * 10
    total_cost = (base_cost + energy_cost + operational_cost + maintenance_cost) * regional_factor
    
    # Store energy usage
    st.session_state.energy_usage.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "energy_consumed": round(energy_consumed, 2)
    })
    st.session_state.energy_usage = st.session_state.energy_usage[-100:]  # Keep last 100 records
    
    return round(total_cost, 2), round(energy_consumed, 2)

def calculate_stress_index(temperature, humidity, rain_predicted):
    temp_stress = (temperature - 25) * 1  # Reduced scaling factor from 2 to 1
    humidity_stress = max((50 - humidity) * 0.5, -5)  # Reduced scaling factor from 1.5 to 0.5, cap from -10 to -5
    rain_adjustment = -2 if rain_predicted else 0  # Reduced from -5 to -2
    baseline_stress = 20  # Increased from 5 to 20 to ensure some inherent stress
    stress = temp_stress + humidity_stress + rain_adjustment + baseline_stress
    return min(max(stress, 0), 100)

def calculate_farm_health(moisture, temperature, stress_index):
    moisture_score = min(100, max(0, 100 - abs(55 - moisture) * 2))
    temp_score = min(100, max(0, 100 - abs(25 - temperature) * 4))
    stress_score = 100 - stress_index
    return round((moisture_score + temp_score + stress_score) / 3, 1), moisture_score

def calculate_efficiency_and_points(water_needed, optimal_water, acres, live_moisture, min_moisture, max_moisture):
    optimal_total = optimal_water * acres
    efficiency = max(0, (optimal_total - water_needed) / optimal_total) * 100
    points_earned = int(efficiency * 10)
    new_badge_earned = False
    if min_moisture <= live_moisture <= max_moisture:
        points_earned += 50
        st.session_state.points += 50
        if st.session_state.points >= 2000 and "Moisture Master 💧" not in st.session_state.badges:
            st.session_state.badges.append("Moisture Master 💧")
            new_badge_earned = True
    st.session_state.points += points_earned
    if st.session_state.points >= 1000 and "Water Saver 🌊" not in st.session_state.badges:
        st.session_state.badges.append("Water Saver 🌊")
        new_badge_earned = True
    if st.session_state.points >= 5000 and "Eco Champion 🌍" not in st.session_state.badges:
        st.session_state.badges.append("Eco Champion 🌍")
        new_badge_earned = True
    st.session_state.new_badge = new_badge_earned
    return efficiency, points_earned

def forecast_water_and_yield(historical_data, recommended_crop, forecast_data, min_moisture, max_moisture):
    if not forecast_data:
        return [], 0
    next_day_data = forecast_data[0]
    temp = next_day_data['temp']
    humidity = next_day_data['humidity']
    rain_predicted = next_day_data['rain_predicted']
    acres = st.session_state.get('land_area', 1.0)
    soil_type = st.session_state.get('soil_type', 'Alluvial Soil')
    live_moisture = st.session_state.live_moisture
    water_needed = calculate_water_requirement(temp, humidity, rain_predicted, acres, soil_type, recommended_crop, live_moisture, min_moisture, max_moisture)
    
    crop_detail = crop_details.get(recommended_crop, {"temp_range": (20.0, 30.0), "humidity_range": (50.0, 70.0)})
    optimal_temp_range = (float(crop_detail["temp_range"][0]), float(crop_detail["temp_range"][1]))
    optimal_humidity_range = (float(crop_detail["humidity_range"][0]), float(crop_detail["humidity_range"][1]))
    temp_score = 1.0 if optimal_temp_range[0] <= temp <= optimal_temp_range[1] else 0.8
    humidity_score = 1.0 if optimal_humidity_range[0] <= humidity <= optimal_humidity_range[1] else 0.9
    yield_potential = 0.95 * temp_score * humidity_score
    
    forecast = [{"date": next_day_data['date'], "water_needed": round(water_needed, 2)}]
    return forecast, yield_potential

def train_crop_model():
    X = np.array([[60, 25], [70, 30], [50, 20], [65, 28]])
    y = ['Wheat', 'Rice', 'Maize', 'Soybean']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_crop(model, moisture, temp):
    return model.predict([[moisture * 1.2, temp]])[0]

def assess_crop_threats(current_weather, recommended_crop):
    threats = []
    recommendations = []
    crop_info = crop_details.get(recommended_crop, {})
    optimal_temp = crop_info.get('temp_range', (20, 30))
    optimal_humidity = crop_info.get('humidity_range', (50, 70))

    aqi_value = current_weather.get('aqi_value', 1)
    aqi_label = current_weather.get('aqi', 'Unknown ❓')
    if aqi_value >= 3:
        threats.append(f"Poor Air Quality (AQI: {aqi_label}) - High pollutant levels may reduce photosynthesis and increase disease susceptibility. 😷")
        recommendations.append("Monitor crop health closely for signs of stress or disease. Consider using organic sprays to protect plants. 🌿")

    temp = current_weather.get('temp', 25)
    if temp < optimal_temp[0]:
        threats.append(f"Low Temperature ({temp}°C) - Below optimal range for {recommended_crop} ({optimal_temp[0]}°C to {optimal_temp[1]}°C), may slow growth. 🥶")
        recommendations.append("Use mulching or row covers to retain soil warmth. 🧣")
    elif temp > optimal_temp[1]:
        threats.append(f"High Temperature ({temp}°C) - Above optimal range for {recommended_crop} ({optimal_temp[0]}°C to {optimal_temp[1]}°C), may cause heat stress. 🔥")
        recommendations.append("Increase irrigation frequency and provide shade if possible. ☂️")

    humidity = current_weather.get('humidity', 50)
    if humidity < optimal_humidity[0]:
        threats.append(f"Low Humidity ({humidity}%) - Below optimal range for {recommended_crop} ({optimal_humidity[0]}% to {optimal_humidity[1]}%), may cause dehydration. 🌵")
        recommendations.append("Increase irrigation to maintain soil moisture. 💧")
    elif humidity > optimal_humidity[1]:
        threats.append(f"High Humidity ({humidity}%) - Above optimal range for {recommended_crop} ({optimal_humidity[0]}% to {optimal_humidity[1]}%), may increase disease risk. 🦠")
        recommendations.append("Ensure proper plant spacing and ventilation to reduce fungal diseases. 🌬️")

    rain_chance = current_weather.get('rain_chance', 0)
    if rain_chance > 50:
        threats.append(f"High Rain Chance ({rain_chance}%) - Excessive rain may lead to waterlogging and root rot. 🌧️")
        recommendations.append("Ensure proper drainage systems are in place. 🚿")
    elif rain_chance < 10:
        threats.append(f"Low Rain Chance ({rain_chance}%) - Insufficient rain may lead to drought stress. 🏜️")
        recommendations.append("Plan for supplemental irrigation if dry conditions persist. 💦")

    return threats, recommendations

# AI-Powered Pest Prediction Functions
def predict_pest_outbreak(current_weather, forecast_data, model, scaler):
    """
    Predict pest outbreak risk for the next 3 days using a pre-trained LSTM model.
    
    Args:
        current_weather (dict): Current weather data (temp, humidity, rain_chance).
        forecast_data (list): Weather forecast for the next 3 days.
        model: Pre-trained LSTM model.
        scaler: Pre-trained scaler for normalizing data.
    
    Returns:
        list: Predicted pest risks for the next 3 days.
    """
    if model is None or scaler is None:
        return []

    # Load historical data to get the last 5 days
    try:
        historical_data = pd.read_csv('historical_weather_pest.csv', parse_dates=['Date'])
        time_steps = 5
        recent_data = historical_data.tail(time_steps)[['Temperature', 'Humidity', 'Rainfall', 'Pest Risk']].values
        
        # If we don't have enough historical data, use current weather as a fallback
        if len(recent_data) < time_steps:
            recent_data = []
            for i in range(time_steps):
                recent_data.append([
                    current_weather['temp'] + np.random.normal(0, 1),
                    current_weather['humidity'] + np.random.normal(0, 2),
                    (current_weather['rain_chance'] / 100 * 5) + np.random.normal(0, 0.5),
                    0.3  # Placeholder pest risk
                ])
            recent_data = np.array(recent_data)
    except:
        # Fallback if historical data is unavailable
        time_steps = 5
        recent_data = []
        for i in range(time_steps):
            recent_data.append([
                current_weather['temp'] + np.random.normal(0, 1),
                current_weather['humidity'] + np.random.normal(0, 2),
                (current_weather['rain_chance'] / 100 * 5) + np.random.normal(0, 0.5),
                0.3  # Placeholder pest risk
            ])
        recent_data = np.array(recent_data)

    # Normalize the recent data using the pre-trained scaler
    last_sequence = scaler.transform(recent_data)

    # Predict pest risk for the next 3 days
    predictions = []
    for day in forecast_data:
        # Reshape the sequence for the LSTM model
        last_sequence_input = last_sequence.reshape((1, time_steps, last_sequence.shape[1]))
        
        # Predict the pest risk
        predicted_risk = model.predict(last_sequence_input, verbose=0)
        
        # Inverse transform to get the pest risk in the original scale
        last_features = last_sequence[-1, :-1].reshape(1, -1)
        combined = np.concatenate([last_features, predicted_risk], axis=1)
        predicted_risk_value = scaler.inverse_transform(combined)[-1, -1]
        
        # Ensure the predicted risk is between 0 and 1
        predicted_risk_value = np.clip(predicted_risk_value, 0, 1)
        
        # Store the prediction
        predictions.append({"date": day['date'], "pest_risk": predicted_risk_value})
        
        # Update the sequence for the next prediction
        new_data = np.array([[day['temp'], day['humidity'], day['pop'] / 100 * 5, predicted_risk_value]])
        last_sequence = np.vstack([last_sequence[1:], new_data])
    
    return predictions

def generate_pdf_report(current_weather, threats, recommendations, water_needed, irrigation_time, cost, land_area, soil_type, recommended_crop):
    pdf_file = "farm_report.pdf"
    current_date = datetime.now().strftime("%Y-%m-%d")
    temp = current_weather.get('temp', 25)
    humidity = current_weather.get('humidity', 50)
    rain_chance = current_weather.get('rain_chance', 0)
    aqi_label = current_weather.get('aqi', 'Unknown ❓')

    # Calculate stress level
    latest_data = st.session_state.data[-1]
    stress_index = calculate_stress_index(latest_data['temperature'], latest_data['humidity'], latest_data['rain_predicted'])

    # Generate graphs as images
    graph_files = []
    if st.session_state.data:
        df = pd.DataFrame(st.session_state.data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

        # Temperature and Humidity Trends
        fig_temp_humidity = go.Figure()
        fig_temp_humidity.add_trace(go.Scatter(
            x=df['timestamp'], y=df['temperature'], name='Temperature (°C) 🌡️',
            line=dict(color='#ff6f61', width=2.5), mode='lines+markers', marker=dict(size=8, color='#ff6f61'), yaxis='y1'
        ))
        fig_temp_humidity.add_trace(go.Scatter(
            x=df['timestamp'], y=df['humidity'], name='Humidity (%) 🌫️',
            line=dict(color='#AAFF00', width=2.5, dash='dash'), mode='lines+markers', marker=dict(size=8, color='#AAFF00'), yaxis='y2'
        ))
        fig_temp_humidity.update_layout(
            title='Temperature and Humidity Trends 🌡️🌫️', xaxis_title='Time ⏰',
            yaxis1=dict(title='Temperature (°C)', titlefont=dict(color='#ff6f61'), tickfont=dict(color='#ff6f61'),
                        range=[min(df['temperature'].min(), 0), max(df['temperature'].max(), 40)], side='left'),
            yaxis2=dict(title='Humidity (%)', titlefont=dict(color='#AAFF00'), tickfont=dict(color='#AAFF00'),
                        range=[0, 100], side='right', overlaying='y1'),
            plot_bgcolor='white', paper_bgcolor='white', font=dict(size=12), showlegend=True
        )
        temp_humidity_file = "temp_humidity_trend.png"
        fig_temp_humidity.write_image(temp_humidity_file, width=600, height=400)
        graph_files.append(temp_humidity_file)

        # Soil Moisture Trend
        if st.session_state.moisture_history:
            moisture_df = pd.DataFrame(st.session_state.moisture_history)
            moisture_df['timestamp'] = pd.to_datetime(moisture_df['timestamp'])
            fig_moisture = go.Figure()
            fig_moisture.add_trace(go.Scatter(
                x=moisture_df['timestamp'], y=moisture_df['moisture'], name='Moisture (%) 💧',
                line=dict(color='#1e90ff', width=2.5), mode='lines+markers', marker=dict(size=8, color='#1e90ff')
            ))
            fig_moisture.update_layout(
                title='Soil Moisture Trend 💧', xaxis_title='Time ⏰', yaxis_title='Moisture (%)',
                plot_bgcolor='white', paper_bgcolor='white', font=dict(size=12)
            )
            moisture_file = "moisture_trend.png"
            fig_moisture.write_image(moisture_file, width=600, height=400)
            graph_files.append(moisture_file)

        # Water Requirement Trend
        if st.session_state.historical_water:
            water_df = pd.DataFrame(st.session_state.historical_water)
            water_df['timestamp'] = pd.to_datetime(water_df['timestamp'])
            fig_water = go.Figure()
            fig_water.add_trace(go.Scatter(
                x=water_df['timestamp'], y=water_df['water_needed'], name='Water Needed (Liters) 💦',
                line=dict(color='#00cc99', width=2.5), mode='lines+markers', marker=dict(size=8, color='#00cc99')
            ))
            fig_water.update_layout(
                title='Water Requirement Trend 💧', xaxis_title='Time ⏰', yaxis_title='Water Needed (Liters)',
                plot_bgcolor='white', paper_bgcolor='white', font=dict(size=12)
            )
            water_file = "water_trend.png"
            fig_water.write_image(water_file, width=600, height=400)
            graph_files.append(water_file)

    try:
        # Set up the PDF document
        doc = SimpleDocTemplate(pdf_file, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        styles = getSampleStyleSheet()
        
        # Define custom styles
        title_style = ParagraphStyle(
            name='Title',
            parent=styles['Title'],
            fontSize=18,
            spaceAfter=12,
            alignment=1,  # Center
            textColor=colors.navy
        )
        subtitle_style = ParagraphStyle(
            name='Subtitle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            alignment=1,  # Center
            textColor=colors.grey
        )
        heading_style = ParagraphStyle(
            name='Heading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.blue
        )
        normal_style = ParagraphStyle(
            name='Normal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=4
        )
        bullet_style = ParagraphStyle(
            name='Bullet',
            parent=styles['Normal'],
            fontSize=10,
            leftIndent=20,
            bulletIndent=10,
            spaceAfter=4
        )

        # Build the PDF content
        elements = []

        # Title and Date
        elements.append(Paragraph("AgriNexus Farm Report", title_style))
        elements.append(Paragraph(f"Date: {current_date}", subtitle_style))
        elements.append(Spacer(1, 24))

        # Farm Details
        elements.append(Paragraph("Farm Details", heading_style))
        elements.append(Paragraph(f"Land Area: {land_area} acres", normal_style))
        elements.append(Paragraph(f"Soil Type: {soil_type}", normal_style))
        elements.append(Paragraph(f"Recommended Crop: {recommended_crop}", normal_style))
        elements.append(Spacer(1, 12))

        # Current Weather Conditions
        elements.append(Paragraph("Current Weather Conditions", heading_style))
        elements.append(Paragraph(f"Temperature: {temp}°C", normal_style))
        elements.append(Paragraph(f"Humidity: {humidity}%", normal_style))
        elements.append(Paragraph(f"Rain Chance: {rain_chance}%", normal_style))
        elements.append(Paragraph(f"Air Quality Index (AQI): {aqi_label}", normal_style))
        elements.append(Spacer(1, 12))

        # Environmental Stress Level
        elements.append(Paragraph("Environmental Stress Level", heading_style))
        elements.append(Paragraph(f"Stress Index: {stress_index} (0-100)", normal_style))
        elements.append(Paragraph(f"Status: {'High' if stress_index > 70 else 'Moderate' if stress_index > 30 else 'Low'}", normal_style))
        elements.append(Spacer(1, 12))

        # Potential Threats to Crops
        elements.append(Paragraph("Potential Threats to Crops", heading_style))
        if threats:
            for threat in threats:
                elements.append(Paragraph(f"• {threat}", bullet_style))
        else:
            elements.append(Paragraph("No significant threats identified.", normal_style))
        elements.append(Spacer(1, 12))

        # Recommendations
        elements.append(Paragraph("Recommendations", heading_style))
        if recommendations:
            for rec in recommendations:
                elements.append(Paragraph(f"• {rec}", bullet_style))
        else:
            elements.append(Paragraph("Continue current practices.", normal_style))
        elements.append(Spacer(1, 12))

        # Irrigation Insights (Water Needed and Predicted Price)
        elements.append(Paragraph("Irrigation Insights", heading_style))
        elements.append(Paragraph(f"Water Needed Today: {water_needed} liters", normal_style))
        elements.append(Paragraph(f"Predicted Irrigation Cost: ₹{cost}", normal_style))
        elements.append(Paragraph(f"Best Time to Irrigate: {irrigation_time}", normal_style))
        elements.append(Spacer(1, 12))

        # Farm Trends (Graphs)
        if graph_files:
            elements.append(Paragraph("Farm Trends Over Time", heading_style))
            for graph_file in graph_files:
                if os.path.exists(graph_file):
                    img = Image(graph_file, width=5*inch, height=3.33*inch)
                    img.hAlign = 'CENTER'
                    elements.append(img)
                    elements.append(Spacer(1, 12))

        # Build the PDF
        doc.build(elements)

        # Verify the PDF exists
        if not os.path.exists(pdf_file):
            st.error("PDF generation failed. The farm_report.pdf file was not created. 📜")
            return None

        # Clean up graph images
        for graph_file in graph_files:
            if os.path.exists(graph_file):
                os.remove(graph_file)

        return pdf_file

    except Exception as e:
        # Clean up graph images in case of failure
        for graph_file in graph_files:
            if os.path.exists(graph_file):
                os.remove(graph_file)
        st.error(f"Failed to generate PDF: {str(e)} 📜")
        return None

def send_pdf_via_email(pdf_path, sender_email, sender_password, receiver_email):
    # Check if the PDF file exists
    if not pdf_path or not os.path.exists(pdf_path):
        return False, "PDF file not found. Cannot send email. 📜"

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = "AgriNexus Farm Report 📊"

        body = "Dear Farmer,\n\nPlease find attached your AgriNexus Farm Report, which includes weather conditions, potential crop threats, and irrigation insights.\n\nBest regards,\nAgriNexus Team 🌱"
        msg.attach(MIMEText(body, 'plain'))

        with open(pdf_path, "rb") as f:
            pdf_attachment = MIMEApplication(f.read(), _subtype="pdf")
            pdf_attachment.add_header('Content-Disposition', 'attachment', filename="farm_report.pdf")
            msg.attach(pdf_attachment)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        
        # Clean up the PDF file after sending
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            
        return True, "Email sent successfully! 📬"
    except Exception as e:
        return False, f"Failed to send email: {str(e)} 🚫"

# Welcome banner
st.markdown('<div class="welcome-banner">Welcome to AgriNexus! 🌾 Let’s Grow Together! 🚜</div>', unsafe_allow_html=True)
st.title("🌱 AgriNexus: Smart Farming Dashboard")
st.markdown("Monitor your farm with AI-powered insights and stunning visuals! 🎉 Let’s make farming fun! 🌟")

st.sidebar.header("🌍 Farm Controls")
refresh_data = st.sidebar.button("Refresh Data 🎯")
land_area = st.sidebar.number_input("Land Area (acres) 📏", min_value=0.1, value=1.0, step=0.1)
st.session_state['land_area'] = land_area
soil_type = st.sidebar.selectbox("Soil Type 🏞️", options=list(soil_adjustments.keys()), index=2)
st.session_state['soil_type'] = soil_type
unit = st.sidebar.selectbox("Water Unit 💧", options=["Liters", "Gallons"], index=0)
water_price = st.sidebar.number_input("Water Price per Liter (INR) 💸", min_value=0.0, value=0.01, step=0.01)
irrigation_method = st.sidebar.selectbox("Irrigation Method 🚿", options=["Drip", "Flood"], index=0)
regional_factor = st.sidebar.selectbox("Regional Water Factor 🌎", options=[0.9, 1.0, 1.2], index=1, help="0.9: Water-rich, 1.0: Neutral, 1.2: Arid")
pump_power = st.sidebar.number_input("Pump Power (kW) ⚡", min_value=1.0, value=5.0, step=0.1)
min_moisture = st.sidebar.slider("Minimum Moisture Threshold (%) 💧", min_value=10.0, max_value=50.0, value=30.0, step=1.0)
max_moisture = st.sidebar.slider("Maximum Moisture Threshold (%) 💧", min_value=50.0, max_value=90.0, value=70.0, step=1.0)
city = st.sidebar.text_input("City for Weather Forecast 🌤️", value="YOUR_CITY")
api_key = " "
selected_crop = st.sidebar.selectbox("Select Crop (Optional) 🌾", options=["Auto", "Wheat", "Rice", "Maize", "Soybean"], index=0)
if selected_crop != "Auto":
    st.session_state.selected_crop = selected_crop


if refresh_data or time.time() - st.session_state.last_update > 10:
    sensor_data = load_sensor_data()
    if sensor_data:
        st.session_state.data = sensor_data
        st.session_state.live_moisture = sensor_data[-1]["moisture"]
        st.session_state.moisture_history.append({
            "timestamp": sensor_data[-1]["timestamp"],
            "moisture": st.session_state.live_moisture
        })
        st.session_state.moisture_history = st.session_state.moisture_history[-100:]
    st.session_state.last_update = time.time()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌍 Current Farm Readings 📡")
    if st.session_state.data:
        latest_data = st.session_state.data[-1]
        latest_data['moisture'] = st.session_state.live_moisture
        st.metric("Soil Moisture 💧", f"{latest_data['moisture']:.1f}%")
        st.metric("Temperature 🌡️", f"{latest_data['temperature']:.1f}°C")
        st.metric("Humidity 🌫️", f"{latest_data['humidity']:.1f}%")
        if latest_data['moisture'] < min_moisture:
            st.markdown('<div class="alert alert-warning">⚠️ Soil too dry! Start irrigation now! 💦</div>', unsafe_allow_html=True)
        elif latest_data['moisture'] > max_moisture:
            st.markdown('<div class="alert alert-warning">⚠️ Risk of waterlogging! Reduce irrigation! 🚱</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert alert-success">✅ Moisture at optimal levels! Great job! 🌱</div>', unsafe_allow_html=True)
        if latest_data['humidity'] < 40:
            st.markdown('<div class="alert alert-warning">⚠️ Air too dry! Let’s irrigate! 💧</div>', unsafe_allow_html=True)
        if latest_data['rain_predicted']:
            st.markdown('<div class="alert alert-info">☔ Rain expected soon! Nature’s helping out! 🌧️</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert alert-info">No data available yet. Let’s get started! 🌱</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌾 Crop Recommendation 🌟")
    recommended_crop = st.session_state.get('selected_crop')
    if not recommended_crop and st.session_state.data:
        model = train_crop_model()
        latest_data = st.session_state.data[-1]
        latest_data['moisture'] = st.session_state.live_moisture
        recommended_crop = predict_crop(model, latest_data['moisture'], latest_data['temperature'])
    crop_info = crop_details.get(recommended_crop, {})
    st.success(f"Recommended Crop: **{recommended_crop.upper() if recommended_crop else 'N/A'}** 🎉")
    if recommended_crop:
        st.write(f"**Optimal Temperature:** {crop_info.get('temp_range', (20, 30))} 🌡️")
        st.write(f"**Optimal Humidity:** {crop_info.get('humidity_range', (50, 70))}% 🌫️")
        st.write(f"**Water Need:** {crop_info.get('water_need', 1000)} liters/acre/day 💧")
        st.info(f"**Growth Tip:** {crop_info.get('tip', 'Check conditions for best results!')} 📝")
    else:
        st.markdown('<div class="alert alert-info">No crop recommendation yet. Let’s gather more data! 🌍</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🌟 Farm Health Summary 🩺")
if st.session_state.data:
    latest_data = st.session_state.data[-1]
    latest_data['moisture'] = st.session_state.live_moisture
    stress_index = calculate_stress_index(latest_data['temperature'], latest_data['humidity'], latest_data['rain_predicted'])
    health_score, moisture_score = calculate_farm_health(latest_data['moisture'], latest_data['temperature'], stress_index)
    st.metric("Farm Health Score 🌿", f"{health_score}%")
    st.metric("Moisture Health 🌧️", f"{moisture_score}%")
    if health_score < 50:
        st.markdown('<div class="alert alert-warning">⚠️ Farm health is low! Let’s take action! 💡</div>', unsafe_allow_html=True)
    elif health_score < 75:
        st.markdown('<div class="alert alert-info">Farm health is moderate. We can do better! 🌱</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert alert-success">Farm health is excellent! Amazing work! 🎉</div>', unsafe_allow_html=True)
        # Trigger confetti if health score is excellent
        st.markdown("""
        <script>
        triggerConfetti();
        </script>
        """, unsafe_allow_html=True)
else:
    st.markdown('<div class="alert alert-info">No data for farm health yet. Let’s check soon! 🌍</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🎮 Water Efficiency Game 🏆")
if st.session_state.data:
    latest_data = st.session_state.data[-1]
    latest_data['moisture'] = st.session_state.live_moisture
    water_needed = calculate_water_requirement(
        temperature=latest_data['temperature'],
        humidity=latest_data['humidity'],
        rain_predicted=latest_data['rain_predicted'],
        acres=land_area,
        soil_type=soil_type,
        crop=recommended_crop,
        live_moisture=latest_data['moisture'],
        min_moisture=min_moisture,
        max_moisture=max_moisture
    )
    optimal_water = crop_water_needs.get(recommended_crop, 1000)
    efficiency, points_earned = calculate_efficiency_and_points(
        water_needed, optimal_water, land_area, latest_data['moisture'], min_moisture, max_moisture
    )
    st.write(f"**Efficiency:** {efficiency:.1f}% 💧")
    st.write(f"**Points Earned Today:** {points_earned} ⭐")
    if min_moisture <= latest_data['moisture'] <= max_moisture:
        st.write("**Moisture Bonus:** +50 points for optimal moisture! 🎉")
    st.write(f"**Total Points:** {st.session_state.points} 🌟")
    progress = min(st.session_state.points / 10000 * 100, 100)
    st.markdown(f'<div class="progress-bar"><div class="progress" style="width: {progress}%">{int(progress)}%</div></div>', unsafe_allow_html=True)
    if st.session_state.badges:
        st.write("**Badges Earned:** 🏅")
        for badge in st.session_state.badges:
            st.markdown(f'<span class="badge">{badge}</span>', unsafe_allow_html=True)
    if st.session_state.new_badge:
        st.markdown("""
        <script>
        triggerConfetti();
        </script>
        """, unsafe_allow_html=True)
        st.session_state.new_badge = False
else:
    st.markdown('<div class="alert alert-info">No data for gamification yet. Let’s play soon! 🌱</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("☀️ Weather Forecast 🌦️")
if city != "YOUR_CITY" and api_key != "YOUR_OPENWEATHERMAP_API_KEY":
    current_weather = fetch_current_weather(city, api_key)
    if current_weather:
        st.write("**Today’s Weather 🌍**")
        st.markdown(
            f'<img src="http://openweathermap.org/img/wn/{current_weather["icon"]}.png" class="weather-icon">',
            unsafe_allow_html=True
        )
        st.write("Weather")
        st.write(f"**Temperature:** {current_weather['temp']}°C 🌡️")
        st.write(f"**Humidity:** {current_weather['humidity']}% 🌫️")
        st.write(f"**Rain Chance:** {current_weather['rain_chance']}% ☔")
        st.write(f"**Air Quality:** {current_weather['aqi']} 🌬️")
    
    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    
    forecast_data = fetch_weather_forecast(city, api_key)
    if forecast_data:
        st.write("**Next 3 Days Forecast 🌟**")
        cols = st.columns(3)
        for i, day in enumerate(forecast_data):
            with cols[i]:
                st.markdown(
                    f'<img src="http://openweathermap.org/img/wn/{day["icon"]}.png" class="weather-icon">',
                    unsafe_allow_html=True
                )
                st.write(f"**{day['date']}** 📅")
                st.write(f"Temp: {day['temp']}°C 🌡️")
                st.write(f"Humidity: {day['humidity']}% 🌫️")
                st.write(f"Rain: {'Yes ☔' if day['rain_predicted'] else 'No ☀️'}")
    else:
        st.markdown('<div class="alert alert-warning">Unable to fetch forecast data. Let’s try again! ⚠️</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="alert alert-info">Please enter your city in the sidebar to see the weather! 🌍</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🌡️ Environmental Stress Index 📊")
if st.session_state.data:
    latest_data = st.session_state.data[-1]
    stress_index = calculate_stress_index(latest_data['temperature'], latest_data['humidity'], latest_data['rain_predicted'])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=stress_index,
        title={'text': "Stress Level (0-100) 🌡️"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#6b48ff"},
            'steps': [
                {'range': [0, 30], 'color': "#00cc99"},
                {'range': [30, 70], 'color': "#ffcc00"},
                {'range': [70, 100], 'color': "#ff6666"}
            ],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 50}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    if stress_index > 70:
        st.markdown('<div class="alert alert-warning">⚠️ High stress! Let’s add shade or adjust watering! 🌱</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="alert alert-info">No data to calculate stress yet. Let’s check soon! 🌍</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("💧 Water & Irrigation Insights 🚿")
if st.session_state.data:
    latest_data = st.session_state.data[-1]
    latest_data['moisture'] = st.session_state.live_moisture
    water_needed = calculate_water_requirement(
        temperature=latest_data['temperature'],
        humidity=latest_data['humidity'],
        rain_predicted=latest_data['rain_predicted'],
        acres=land_area,
        soil_type=soil_type,
        crop=recommended_crop,
        live_moisture=latest_data['moisture'],
        min_moisture=min_moisture,
        max_moisture=max_moisture
    )
    
    # Schedule irrigation based on forecast
    irrigation_time, irrigation_schedule = schedule_irrigation(
        temperature=latest_data['temperature'],
        humidity=latest_data['humidity'],
        forecast_data=forecast_data if 'forecast_data' in locals() else []
    )
    
    cost, energy_consumed = calculate_irrigation_cost_and_energy(
        water_needed=water_needed,
        acres=land_area,
        soil_type=soil_type,
        irrigation_method=irrigation_method,
        water_price=water_price,
        energy_rate=7.0,
        regional_factor=regional_factor,
        pump_power=pump_power
    )
    
    st.session_state.historical_water.append({
        "timestamp": latest_data['timestamp'],
        "water_needed": water_needed,
        "temperature": latest_data['temperature'],
        "humidity": latest_data['humidity'],
        "rain_predicted": latest_data['rain_predicted']
    })
    
    display_water = water_needed
    unit_label = "liters"
    if unit == "Gallons":
        display_water = water_needed * 0.264
        unit_label = "gallons"
    
    st.metric("Water Needed Today 💦", f"{display_water:.2f} {unit_label}")
    st.metric("Irrigation Cost 💸", f"₹{cost:.2f}")
    st.write(f"For {land_area} acres of {soil_type} with {recommended_crop}, using {irrigation_method} method. 🌾")
    st.info(f"**Best Time to Irrigate Today:** {irrigation_time} ⏰")
    
    # Display irrigation schedule for the next 3 days
    st.write("**Irrigation Schedule (Next 3 Days):** 📅")
    for day in irrigation_schedule:
        st.write(f"- {day['date']}: {day['time']}")
else:
    st.markdown('<div class="alert alert-info">No data for water insights yet. Let’s check soon! 🌱</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("⚡ Energy Consumption Tracker 🔋")
if st.session_state.energy_usage:
    energy_df = pd.DataFrame(st.session_state.energy_usage)
    energy_df['timestamp'] = pd.to_datetime(energy_df['timestamp'])
    
    st.metric("Energy Consumed Today ⚡", f"{energy_consumed:.2f} kWh")
    
    fig_energy = go.Figure()
    fig_energy.add_trace(go.Scatter(
        x=energy_df['timestamp'],
        y=energy_df['energy_consumed'],
        name='Energy Consumed (kWh) ⚡',
        line=dict(color='#ffcc00', width=2.5),
        mode='lines+markers',
        marker=dict(size=8, color='#ffcc00')
    ))
    fig_energy.update_layout(
        title='Energy Consumption Trend ⚡',
        xaxis_title='Time ⏰',
        yaxis_title='Energy (kWh)',
        xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.2)'),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(size=12)
    )
    st.plotly_chart(fig_energy, use_container_width=True)
else:
    st.markdown('<div class="alert alert-info">No energy usage data yet. Let’s check after irrigation! ⚡</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🐞 AI-Powered Pest Prediction 🛡️")
if current_weather and forecast_data and recommended_crop:
    pest_predictions = predict_pest_outbreak(current_weather, forecast_data, pest_model, pest_scaler)
    
    if pest_predictions:
        st.write("**Pest Outbreak Risk (Next 3 Days):** 📅")
        for prediction in pest_predictions:
            risk = prediction['pest_risk']
            date = prediction['date']
            st.metric(f"Pest Risk on {date} 🐞", f"{risk:.2f}")
            
            if risk > 0.7:
                st.markdown('<div class="alert alert-warning">⚠️ High risk of pest outbreak!</div>', unsafe_allow_html=True)
                st.write("**Preventive Recommendations:**")
                st.write("- Apply organic pest repellents like neem oil.")
                st.write("- Increase monitoring of crops for early signs of pests.")
                st.write("- Ensure proper field sanitation to reduce pest habitats.")
            elif risk > 0.3:
                st.markdown('<div class="alert alert-info">Moderate risk of pest outbreak.</div>', unsafe_allow_html=True)
                st.write("**Preventive Recommendations:**")
                st.write("- Check weather conditions and adjust irrigation to avoid excess moisture.")
                st.write("- Introduce natural predators like ladybugs if applicable.")
            else:
                st.markdown('<div class="alert alert-success">Low risk of pest outbreak.</div>', unsafe_allow_html=True)
                st.write("**Preventive Recommendations:**")
                st.write("- Maintain regular monitoring as a precaution.")
    else:
        st.markdown('<div class="alert alert-warning">Unable to predict pest risk due to model loading issues.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="alert alert-info">No data to predict pest outbreaks yet. Let’s check soon! 🌍</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("⚠️ Potential Crop Threats 🚨")
if current_weather and recommended_crop:
    threats, recommendations = assess_crop_threats(current_weather, recommended_crop)
    if threats:
        for threat in threats:
            st.markdown(f'<div class="alert alert-warning">{threat}</div>', unsafe_allow_html=True)
        st.write("**Recommendations to Keep Your Crops Happy:** 🌟")
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.markdown('<div class="alert alert-success">No significant threats identified! Your crops are happy! 🌱</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="alert alert-info">No data to assess crop threats yet. Let’s check soon! 🌍</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📤 Share Farm Report 📧")
if current_weather and recommended_crop and st.session_state.data:
    receiver_email = st.text_input("Enter Farmer's Email Address: 📧")
    sender_email = ""
    sender_password = ""
    if st.button("Generate and Send Report via Email 📬"):
        if not receiver_email or not sender_email or not sender_password:
            st.error("Please fill in all email fields! 🚫")
        else:
            pdf_path = generate_pdf_report(current_weather, threats, recommendations, water_needed, irrigation_time, cost, land_area, soil_type, recommended_crop)
            success, message = send_pdf_via_email(pdf_path, sender_email, sender_password, receiver_email)
            if success:
                st.success(message)
            else:
                st.error(message)
else:
    st.markdown('<div class="alert alert-info">Complete farm data required to generate the report. Let’s gather more info! 🌍</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🌅 Tomorrow's Forecast 🔮")
if st.session_state.data and city != "YOUR_CITY" and api_key != "YOUR_OPENWEATHERMAP_API_KEY":
    forecast_data = fetch_weather_forecast(city, api_key)
    if forecast_data:
        forecast_result, yield_potential = forecast_water_and_yield(
            st.session_state.historical_water, recommended_crop, forecast_data, min_moisture, max_moisture
        )
        next_day = forecast_data[0]
        st.write(f"**Date:** {next_day['date']} 📅")
        st.metric("Predicted Water Need 💧", f"{forecast_result[0]['water_needed']:.2f} liters")
        st.write(f"**Yield Potential:** {yield_potential * 100:.1f}% 🌾")
    else:
        st.markdown('<div class="alert alert-warning">Forecast data unavailable. Let’s try again! ⚠️</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="alert alert-info">Enter city and API key to see tomorrow’s forecast! 🌍</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Farm Trends Over Time 📈")
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['temperature'],
        name='Temperature (°C) 🌡️',
        line=dict(color='#ff6f61', width=2.5),
        mode='lines+markers',
        marker=dict(size=8, color='#ff6f61'),
        yaxis='y1'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['humidity'],
        name='Humidity (%) 🌫️',
        line=dict(color='#AAFF00', width=2.5, dash='dash'),
        mode='lines+markers',
        marker=dict(size=8, color='#AAFF00'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Temperature and Humidity Trends 🌡️🌫️',
        xaxis_title='Time ⏰',
        yaxis1=dict(
            title='Temperature (°C)',
            titlefont=dict(color='#ff6f61'),
            tickfont=dict(color='#ff6f61'),
            range=[min(df['temperature'].min(), 0), max(df['temperature'].max(), 40)],
            side='left',
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.2)'
        ),
        yaxis2=dict(
            title='Humidity (%)',
            titlefont=dict(color='#AAFF00'),
            tickfont=dict(color='#AAFF00'),
            range=[0, 100],
            side='right',
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.2)'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.2)'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(size=12),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.moisture_history:
        moisture_df = pd.DataFrame(st.session_state.moisture_history)
        moisture_df['timestamp'] = pd.to_datetime(moisture_df['timestamp'])
        
        fig_moisture = go.Figure()
        fig_moisture.add_trace(go.Scatter(
            x=moisture_df['timestamp'],
            y=moisture_df['moisture'],
            name='Moisture (%) 💧',
            line=dict(color='#1e90ff', width=2.5),
            mode='lines+markers',
            marker=dict(size=8, color='#1e90ff')
        ))
        
        fig_moisture.update_layout(
            title='Soil Moisture Trend 💧',
            xaxis_title='Time ⏰',
            yaxis_title='Moisture (%)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.2)'
            ),
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(size=12)
        )
        
        st.plotly_chart(fig_moisture, use_container_width=True)
    
    if st.session_state.historical_water:
        water_df = pd.DataFrame(st.session_state.historical_water)
        water_df['timestamp'] = pd.to_datetime(water_df['timestamp'])
        
        fig_water = go.Figure()
        fig_water.add_trace(go.Scatter(
            x=water_df['timestamp'],
            y=water_df['water_needed'],
            name='Water Needed (Liters) 💦',
            line=dict(color='#00cc99', width=2.5),
            mode='lines+markers',
            marker=dict(size=8, color='#00cc99')
        ))
        
        fig_water.update_layout(
            title='Water Requirement Trend 💧',
            xaxis_title='Time ⏰',
            yaxis_title='Water Needed (Liters)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.2)'
            ),
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(size=12)
        )
        
        st.plotly_chart(fig_water, use_container_width=True)
else:
    st.markdown('<div class="alert alert-info">No historical data yet. Let’s keep growing! 🌱</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
