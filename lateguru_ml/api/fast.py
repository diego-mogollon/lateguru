import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
from lateguru_ml.params import *
from lateguru_ml.ml_logic.registry import load_model
from lateguru_ml.ml_logic.preprocessor import preprocess_features
from lateguru_ml.ml_logic.model import predict
from lateguru_ml.ml_logic.data import load_airport_geo_data
import os

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

#reading airports_geolocation_csv
airport_geo_df = load_airport_geo_data('../raw_data/Dataset_A/Dataset_A_US2023_Kaggle_airports_geolocation.csv')

#should I handle this with functions or hardcode the object like below?
cleaned_airport_geos_df = airport_geo_df[['IATA_CODE', 'LATITUDE', 'LONGITUDE']]\
    .rename(columns={'IATA_CODE': 'airport_acronym',
                     'LATITUDE': 'lat',
                     'LONGITUDE':'lon'})
#global variables
#airport acronym e.g JFK as taken from the frontend
airport = None



'''airport_coords = {
    #e.g 'MEL': {'lat': -37.663712, 'lon': 144.844788},
    #    'JFK': {'lat': 27.739487, 'lon': -54.9384},....

}'''
#Taking in the airport acronym from the frontend and outputting lat and lon coordinates
def get_lat_lon_cordinates(airport):
    '''Function that takes in an airport variable that is a string of the airports IATA code acronym and outputs
    the geo data on this airport as a dictionary'''

    selected_airport = cleaned_airport_geos_df[cleaned_airport_geos_df['airport_acronym']==airport]

    airport_geo_codes = {
        #'name': selected_airport['airport_acronym'],
        'lat': selected_airport['lat'],
        'lon': selected_airport['lon']
    }

    return airport_geo_codes['lat'], airport_geo_codes['lon']


    '''lat = airport_coords[airports]['lat']
    lon = airport_coords[airports]['lon']'''

    #return lat, lon

lat, lon = get_lat_lon_cordinates(airport)
#calling the open weather with lat & lon params as inputs
###Getting weather data for melbourne aiport from the open weather api for current weather
def get_weather_data(lat, lon):
    #lat= -37.663712
    #lon= 144.844788

    url = "https://api.openweathermap.org/data/2.5/weather?"
    params = {'lat': str(lat), 'lon':str(lon), "appid":OW_API_KEY}
    response = requests.get(url, params=params)
    json_response = response.json()

    df = pd.DataFrame({
    "temp": json_response['main']['temp'],
    "feels_like_temp": json_response['main']['feels_like'],
    "alt_pressure": json_response['main']['pressure'],
    "sl_pressure": json_response['main']['sea_level'],
    "visibility": json_response['visibility'],
    "wind_speed": json_response['wind']['speed'],
    "wind_gust": json_response['wind']['deg']
    }, index = [0])

    #clean up json response object before outputting data
    return df


#unpack X_weather variables and insert them into predict endpoint function
#maybe build predict object as a list because it will be ordered

#X features
'''['Origin', 'Dest', 'Carrier', 'Cancelled', 'CancellationReason',
       'Delayed', 'DepDelayMinutes', 'CarrierDelay', 'Weather_Delay_Length',
       'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'Temperature',
       'Feels_Like_Temperature', 'Altimeter_Pressure', 'Sea_Level_Pressure',
       'Visibility', 'Wind_Speed', 'Wind_Gust', 'Precipitation',
       'Ice_Accretion_3hr', 'Hour', 'Day_Of_Week', 'Month', 'Weather_Delayed']'''

#coding the predict endpoint function
@app.get("/predict")
def predict(
    origin: str, #Origin airport code e.g JFK
    destination: str, #Destination airport code e.g SFO
    carrier: str, #name of airline carrier
    cancelled: bool, #True if cancelled, else False (default)
    cancellation_reason: str, #[Default = 'Not Cancelled', 'Weather', 'Carrier', 'Security', 'National Air System']
    delayed: bool, #True if flight delayed, else False (default)
    delay_len: float, #amount of mins flight is delayed
    carrier_delay_len: float, #amount of mins flight is delayed due to a carrier related reason
    nas_delay_len: float, #amount of mins flight is delayed due to a national security matter
    security_delay_len: float, #amount of mins flight is delayed due to a general security matter
    late_aircraft_delay_len: float, #amount of mins flight is delayed due to a late arriving aircraft
    temperature: float, #temperature in origin at time of prediction
    feels_like_temp: float, #feels like temperature in origin at time of prediction
    alt_pressure: float, #air pressure at origin
    sl_pressure: float, #sea level pressure at origin
    visibility: float, #measure of how clear visbility at origin is
    wind_speed: float, #measure of wind speed in km/h
    wind_gust: float, #measure of wind gust strength at origin
    precipitation: float, #measure of rainfall at origin
    ice_accretion_3hr: float, #measure of ice at origin
    hour: float, #hour at time of flight departure
    day_of_week: float, #number corresponding to day of week
    month: float, #month of year at time of departure
):
    """Make a prediction based on inputs as to whether the flight is likely
    to be delayed by weather
    Assumes flight is taking place in the USA"""

    X_weather = get_weather_data(lat, lon)

    X_pred = pd.DataFrame({
        "origin": [str(origin)],
        "destination": [str(destination)],
        "carrier": [str(carrier)],
        "cancelled": [bool(cancelled)],
        "cancellation_reason": [str(cancellation_reason)],
        "delayed": [bool(delayed)],
        "delay_len": [float(delay_len)],
        "carrier_delay_len": [float(carrier_delay_len)],
        "nas_delay_len": [float(nas_delay_len)],
        "security_delay_len": [float(security_delay_len)],
        "late_aircraft_delay_len": [float(late_aircraft_delay_len)],
        "temperature": [float(X_weather['temp'])],
        "feels_like_temp": [float(X_weather['feels_like_temp'])],
        "alt_pressure": [float(X_weather['alt_pressure'])],
        "sl_pressure": [float(X_weather['sl_pressure'])],
        "visibility": [float(X_weather['visibility'])],
        "wind_speed": [float(X_weather['wind_speed'])],
        "wind_gust": [float(X_weather['wind_gust'])],
        #must still input precipitation
        "precipitation": [float(precipitation)],
        #must still input ice accretion
        "ice_accretion_3hr": [float(ice_accretion_3hr)],
        "hour": [float(hour)],
        "day_of_week": [float(day_of_week)]
        "month": [float(month)]
        })

    model = load_model('test_xgb_model.pkl')

    X_processed = preprocess_features(X_pred)

    y_pred = model.predict(model, X_processed)

    return dict(likely_to_be_delayed=bool(y_pred))

@app.get("/")
def root():
    # YOUR CODE HERE
    return {'greeting': 'Hello'}
