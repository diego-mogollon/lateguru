import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
from lateguru_ml.params import *
from lateguru_ml.ml_logic.registry import load_model
from lateguru_ml.ml_logic.preprocessor import preprocess_features_v2
from lateguru_ml.ml_logic.model import predict
from lateguru_ml.ml_logic.weather_utils import get_lat_lon_cordinates, get_weather_data, airport
import os
from lateguru_ml.ml_logic.data import load_airport_geo_data
from lateguru_ml.params import OW_API_KEY

#how the api request url is looking
#http://127.0.0.1:8000/predict_delay?origin=LAX&destination=JFK&carrier=American%20Airlines%20Inc.&hour=10&day_of_week=4&month=4

#building the fast api instance
app = FastAPI()

# Allowing middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

lat, lon = get_lat_lon_cordinates(airport)

X_weather = get_weather_data(lat=lat, lon=lon)


#X features
'''features = [
    'Origin', 'Dest', 'Carrier', 'DayOfWeek', 'HourOfDay',
    'Temperature', 'Feels_Like_Temperature', 'Altimeter_Pressure',
    'Sea_Level_Pressure', 'Visibility', 'Wind_Speed', 'Wind_Gust',
    'Precipitation', 'CarrierAvgDelay', 'Month'
]'''

#coding the predict endpoint function
#double check the dtypes here as they all have to be strings
@app.get("/predict_delay")
def predict_delay(
    origin: str, #Origin airport code e.g JFK
    destination: str, #Destination airport code e.g SFO
    carrier: str, #name of airline carrier
    hour: float, #hour at time of flight departure
    day_of_week: float, #number corresponding to day of week
    month: float, #month of year at time of departure


):
    """Make a prediction based on inputs as to whether the flight is likely
    to be delayed by weather
    Assumes flight is taking place in the USA"""


    X_pred = pd.DataFrame({
        "Origin": [str(origin)],
        "Dest": [str(destination)],
        "Carrier": [str(carrier)],
        "temp": [float(X_weather['temp'])],
        "feels_like_temp": [float(X_weather['feels_like_temp'])],
        "alt_pressure": [float(X_weather['alt_pressure'])],
        "sl_pressure": [float(X_weather['sl_pressure'])],
        "visibility": [float(X_weather['visibility'])],
        "wind_speed": [float(X_weather['wind_speed'])],
        "wind_gust": [float(X_weather['wind_gust'])],
        "DayOfWeek": [float(day_of_week)],
        "HourOfDay": [float(hour)],
        "Precipitation": [float(0.0)],
        "CarrierAvgDelay": [float(20)],
        "Month": [float(month)]
        })

    model = load_model('_model.pkl')

    X_processed = preprocess_features_v2(X_pred)


    y_pred = predict(model, X_processed)

    return dict(likely_to_be_delayed=bool(y_pred))

@app.get("/")
def root():
    # YOUR CODE HERE
    return {'greeting': 'Hello'}
