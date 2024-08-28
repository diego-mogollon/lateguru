import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
from lateguru_ml.params import *
from lateguru_ml.ml_logic.registry import load_model
from lateguru_ml.ml_logic.preprocessor import preprocess_features_v2, create_preprocessing_pipeline
from lateguru_ml.ml_logic.model import predict
from lateguru_ml.ml_logic.weather_utils import get_lat_lon_cordinates, get_weather_data, top_5_airport_coords

import os
#from lateguru_ml.ml_logic.data import load_airport_geo_data
from lateguru_ml.params import OW_API_KEY
import joblib

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

avg_carrier_delay = {
"Alaska Airlines Inc.":12.908908,
"Allegiant Air.":22.017399,
"American Airlines Inc.":20.101923,
"Delta Air Lines Inc.":12.236454,
"Endeavor Air Inc.":7.980184,
"Envoy Air":10.305668,
"Frontier Airlines Inc.":22.357216,
"Hawaiian Airlines Inc.":12.628407,
"Horizon Air":8.841963,
"JetBlue Airways":23.910759,
"Mesa Airlines Inc.":25.380383,
"PSA Airlines Inc.":17.002264,
"Republic Airline":11.582884,
"SkyWest Airlines Inc.":16.060057,
"Southwest Airlines Co.":18.943163,
"Spirit Air Lines ":19.418934,
"United Air Lines Inc.":17.166492,
}

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

    #lat, lon = get_lat_lon_cordinates(origin)

    lat = top_5_airport_coords[origin]['lat']
    lon = top_5_airport_coords[origin]['lon']

    X_weather = get_weather_data(lat=lat, lon=lon)

    X_pred = pd.DataFrame({
        "Origin": [str(origin)],
        "Dest": [str(destination)],
        "Carrier": [str(carrier)],
        "Temperature": [float(X_weather['temp'])],
        "Feels_Like_Temperature": [float(X_weather['feels_like_temp'])],
        "Altimeter_Pressure": [float(X_weather['alt_pressure'])],
        "Sea_Level_Pressure": [float(X_weather['sl_pressure'])],
        "Visibility": [float(X_weather['visibility'])],
        "Wind_Speed": [float(X_weather['wind_speed'])],
        "Wind_Gust": [float(X_weather['wind_gust'])],
        "DayOfWeek": [float(day_of_week)],
        "HourOfDay": [float(hour)],
        "Precipitation": [float(X_weather['rain'])],
        "CarrierAvgDelay": [avg_carrier_delay[carrier]],
        "Month": [float(month)]
        })

    model = joblib.load('/Users/conorjohnston/code/diego-mogollon/lateguru/lateguru/model/20240825_xgb_model_top5.pkl')

    #X_processed = preprocess_only_features(X_pred, onehot_features,target_encoded_feature, numeric_features)

    #X_processed = preprocess_features_v2(X_pred)

    X_processed = X_pred

    y_pred = predict(model, X_processed)

    return dict(likely_to_be_delayed=bool(y_pred))

@app.get("/")
def root():
    # YOUR CODE HERE
    return {'greeting': 'Hello'}
