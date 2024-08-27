import pandas as pd
from lateguru_ml.ml_logic.data import load_airport_geo_data
from lateguru_ml.params import OW_API_KEY
import requests

#should I handle this with functions or hardcode the object like below?
#global variables
#airport acronym e.g JFK as taken from the frontend, change to handle input
airport = 'LAX'

#reading airports_geolocation_csv
airport_geo_df = load_airport_geo_data('/Users/conorjohnston/code/diego-mogollon/lateguru/lateguru/raw_data/Dataset_A/Dataset_A_US2023_Kaggle_airports_geolocation.csv')

cleaned_airport_geos_df = airport_geo_df[['IATA_CODE', 'LATITUDE', 'LONGITUDE']]\
    .rename(columns={'IATA_CODE': 'airport_acronym',
                     'LATITUDE': 'lat',
                     'LONGITUDE':'lon'})

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

    return selected_airport['lat'].values[0], selected_airport['lon'].values[0]


#calling the open weather with lat & lon params as inputs
###Getting weather data for melbourne aiport from the open weather api for current weather
def get_weather_data(lat, lon):
    #lat= -37.663712
    #lon= 144.844788

    url = "https://api.openweathermap.org/data/2.5/weather?"
    params = {'lat': str(lat), 'lon':str(lon), "appid":OW_API_KEY, "units":'imperial'}
    response = requests.get(url, params=params)
    json_response = response.json()


    weather_dict = {
    "temp": json_response["main"]['temp'],
    "feels_like_temp": json_response['main']['feels_like'],
    "alt_pressure": json_response['main']['pressure'],
    "sl_pressure": json_response['main']['sea_level'],
    "visibility": json_response['visibility'],
    "wind_speed": json_response['wind']['speed'],
    "wind_gust": json_response['wind']['deg'],
    "rain": json_response['rain']["1h"]
    #precipitation and ice accretion to be added
    }

    #clean up json response object before outputting data
    return weather_dict


#function to convert kelvin to fahrenheit as
#openweather temp and feels like temp variables are default in Kelvin
#but dataset temp is fahrenheit and feels like temp is kelvin

def fahrenheit_to_kelvin(fahrenheit_feels_like_temp):
    '''Function to convert the feels like temperature from Fahrenheit
    in open weather response to Kelvin as it is in the FORWW dataset'''
    kelvin_feels_like_temp = (fahrenheit_feels_like_temp + 459.67) * 1.8
    return float(kelvin_feels_like_temp)
