import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
#from ml_logic import load_model() function
#from ml_logic import preprocess_features() function

app = FastAPI()

# Allowing middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

'''['Time', 'Origin', 'Dest', 'Carrier', 'Cancelled', 'CancellationReason',
       'Delayed', 'DepDelayMinutes', 'CarrierDelay', 'Weather_Delay_Length',
       'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'Temperature',
       'Feels_Like_Temperature', 'Altimeter_Pressure', 'Sea_Level_Pressure',
       'Visibility', 'Wind_Speed', 'Wind_Gust', 'Precipitation',
       'Ice_Accretion_3hr', 'Hour', 'Day_Of_Week', 'Month', 'Weather_Delayed']'''

@app.get("/predict")
def predict(
     departure_time: str, #time of flight departure e.g 2014-07-06 19:00:00
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

    X_pred = pd.DataFrame({
    "departure_time":[pd.Timestamp(departure_time, tz='America/New_York')],
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
    "temperature": [float(temperature)],
    "feels_like_temp": [float(feels_like_temp)],
    "alt_pressure": [float(alt_pressure)],
    "sl_pressure": [float(sl_pressure)],
    "visibility": [float(visibility)],
    "wind_speed": [float(wind_speed)],
    "wind_gust": [float(wind_gust)],
    "precipitation": [float(precipitation)],
    "ice_accretion_3hr": [float(ice_accretion_3hr)],
    "hour": [float(hour)],
    "day_of_week": [float(day_of_week)]
    "month": [float(month)]
    })

    #model = load_model("Production")

    #X_processed = preprocess_features(X_pred)

    #y_pred = model.predict(X_processed)

    #return dict(likely_to_be_delayed=bool(y_pred))

@app.get("/")
def root():
    # YOUR CODE HERE
    return {'greeting': 'Hello'}
