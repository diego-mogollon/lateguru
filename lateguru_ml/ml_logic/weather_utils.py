import pandas as pd
from lateguru_ml.ml_logic.data import load_airport_geo_data
from lateguru_ml.params import OW_API_KEY
import requests
import os

#should I handle this with functions or hardcode the object like below?
#global variables
#airport acronym e.g JFK as taken from the frontend, change to handle input
#airport = 'LAX'

#reading airports_geolocation_csv
#dictionary with top 5 airport lat and lons

top_5_airport_coords = {
    "LAX": {"lat":33.942791, "lon": -118.410042},
    "ATL": {"lat":33.640411, "lon":-84.419853},
    "DEN": {"lat":39.849312, "lon":-104.673828},
    "DFW": {"lat":32.897480, "lon":-97.040443},
    "ORD": {"lat":41.978611, "lon":-87.904724}
}

def get_lat_lon_cordinates(airport):
    script_dir = os.path.dirname(__file__)
    raw_data_path = os.path.join(script_dir, '..', '..', 'raw_data', 'Dataset_A_US2023_Kaggle_Airports_Geolocation.csv')
    airport_geo_df = load_airport_geo_data(raw_data_path)

    cleaned_airport_geos_df = airport_geo_df[['IATA_CODE', 'LATITUDE', 'LONGITUDE']]\
        .rename(columns={'IATA_CODE': 'airport_acronym',
                        'LATITUDE': 'lat',
                        'LONGITUDE':'lon'})

    selected_airport = cleaned_airport_geos_df[cleaned_airport_geos_df['airport_acronym']==airport]

    return selected_airport['lat'].values[0], selected_airport['lon'].values[0]

'''#Taking in the airport acronym from the frontend and outputting lat and lon coordinates
def get_lat_lon_cordinates(airport):
    return airport, airport'''


def fahrenheit_to_kelvin(fahrenheit_feels_like_temp):
    '''Function to convert the feels like temperature from Fahrenheit
    in open weather response to Kelvin as it is in the FORWW dataset'''
    kelvin_feels_like_temp = (fahrenheit_feels_like_temp + 459.67) * 1.8
    return float(kelvin_feels_like_temp)


#calling the open weather with lat & lon params as inputs
###Getting weather data for melbourne aiport from the open weather api for current weather
def get_weather_data(lat, lon):
    #lat= -37.663712
    #lon= 144.844788

    url = "https://api.openweathermap.org/data/2.5/weather?"
    params = {'lat': str(lat), 'lon':str(lon), "appid":'9efa0a3dd9883bae2b9f3bda6eff2e36', "units":'imperial'}
    response = requests.get(url, params=params)
    json_response = response.json()
    #handling the fahrenheit feels like temp and converting to kelvin
    kelvin_feels_like_temp = fahrenheit_to_kelvin(json_response['main']['feels_like'])

    #control flow to handle what to set rain to depending if it is in the response
    if json_response.get('rain', None) == None:

        weather_dict = {
        "temp": json_response["main"]['temp'],
        "feels_like_temp": kelvin_feels_like_temp,
        "alt_pressure": json_response['main']['grnd_level'],
        "sl_pressure": json_response['main']['sea_level'],
        "visibility": json_response['visibility'],
        "wind_speed": json_response['wind']['speed'],
        "wind_gust": json_response['wind']['deg'],
        "rain": float(0.0)
        #precipitation and ice accretion to be added
        }

    else:

        weather_dict = {
        "temp": json_response["main"]['temp'],
        "feels_like_temp": kelvin_feels_like_temp,
        "alt_pressure": json_response['main']['grnd_level'],
        "sl_pressure": json_response['main']['sea_level'],
        "visibility": json_response['visibility'],
        "wind_speed": json_response['wind']['speed'],
        "wind_gust": json_response['wind']['deg'],
        "rain": json_response['rain']["1h"]
        }



    #clean up json response object before outputting data
    return weather_dict
