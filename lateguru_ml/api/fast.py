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
'''categorical_features = ['Origin', 'Dest', 'Route', 'Carrier']
numerical_features = [
    'DayOfWeek', 'HourOfDay', 'Temperature', 'Feels_Like_Temperature',
    'Altimeter_Pressure', 'Sea_Level_Pressure', 'Visibility',
    'Wind_Speed', 'Wind_Gust', 'Precipitation', 'Ice_Accretion_3hr',
    'CarrierAvgDelay', 'Month'
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

    #delayed: bool, #True if flight delayed, else False (default)
    #delay_len: float, #amount of mins flight is delayed
    #carrier_avg_delay: float, #amount of mins on average that each carrier is delayed when delayed
    #temperature: float, #temperature in origin at time of prediction
    #feels_like_temp: float, #feels like temperature in origin at time of prediction
    #alt_pressure: float, #air pressure at origin
    #sl_pressure: float, #sea level pressure at origin
    #visibility: float, #measure of how clear visbility at origin is
    #wind_speed: float, #measure of wind speed in km/h
    #wind_gust: float, #measure of wind gust strength at origin
    #precipitation: float, #measure of rainfall at origin
    #ice_accretion_3hr: float, #measure of ice at origin
    #route: str, #concatenation of origin and destination airport string
):
    """Make a prediction based on inputs as to whether the flight is likely
    to be delayed by weather
    Assumes flight is taking place in the USA"""

    X_pred = pd.DataFrame({
        "Origin": [str(origin)],
        "Dest": [str(destination)],
        "Route": [str('LAX_JFK')],
        "Carrier": [str(carrier)],

        #"delay_len": [float(delay_len)],
        #'''"temperature": [float(X_weather['main']['temp'])],
        #"feels_like_temp": [float(X_weather['main']['feels_like'])],
        #"alt_pressure": [float(X_weather['main']['pressure'])],
        #"sl_pressure": [float(X_weather['main']['sea_level'])],
        #"visibility": [float(X_weather['visibility'])],
        #"wind_speed": [float(X_weather['wind']['speed'])],
        #"wind_gust": [float(X_weather['wind']['gust'])],'''

        "DayOfWeek": [float(day_of_week)],
        "HourOfDay": [float(hour)],
        "Temperature": [float(X_weather['temp'])],
        "Feels_Like_Temperature": [float(14)],
        "Altimeter_Pressure": [float(3478)],
        "Sea_Level_Pressure": [float(1293)],
        "Visibility": [float(1029)],
        "Wind_Speed": [float(15)],
        "Wind_Gust": [float(5)],
        #must still input precipitation
        "Precipitation": [float(0.0)],
        #must still input ice accretion
        "Ice_Accretion_3hr": [float(0.0)],
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
