import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import joblib
from lateguru_ml.params import *
from lateguru_ml.ml_logic.registry import load_model
# from lateguru_ml.ml_logic.preprocessor import preprocess_features_v2, create_preprocessing_pipeline
from lateguru_ml.ml_logic.model import predict
from lateguru_ml.ml_logic.weather_utils import get_lat_lon_cordinates, get_weather_data, top_5_airport_coords

# Initialize FastAPI app
app = FastAPI()

# Allowing CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Average carrier delay data
avg_carrier_delay = {
    "Alaska Airlines Inc.": 12.908908,
    "Allegiant Air.": 22.017399,
    "American Airlines Inc.": 20.101923,
    "Delta Air Lines Inc.": 12.236454,
    "Endeavor Air Inc.": 7.980184,
    "Envoy Air": 10.305668,
    "Frontier Airlines Inc.": 22.357216,
    "Hawaiian Airlines Inc.": 12.628407,
    "Horizon Air": 8.841963,
    "JetBlue Airways": 23.910759,
    "Mesa Airlines Inc.": 25.380383,
    "PSA Airlines Inc.": 17.002264,
    "Republic Airline": 11.582884,
    "SkyWest Airlines Inc.": 16.060057,
    "Southwest Airlines Co.": 18.943163,
    "Spirit Air Lines ": 19.418934,
    "United Air Lines Inc.": 17.166492,
}

# Load the preprocessor and model from files
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'model')
MODEL_FILE = os.path.join(MODEL_DIR, 'xgb_model.pkl')
PREPROCESSOR_FILE = os.path.join(MODEL_DIR, 'preprocessor.pkl')

model = joblib.load(MODEL_FILE)
preprocessor = joblib.load(PREPROCESSOR_FILE)

@app.get("/predict_delay")
def predict_delay(
    origin: str,
    destination: str,
    carrier: str,
    hour: float,
    day_of_week: float,
    month: float
):
    """Make a prediction based on inputs as to whether the flight is likely
    to be delayed by weather. Assumes flight is taking place in the USA."""

    # Get lat/lon for origin airport
    lat = top_5_airport_coords[origin]['lat']
    lon = top_5_airport_coords[origin]['lon']

    # Retrieve weather data based on coordinates
    X_weather = get_weather_data(lat=lat, lon=lon)

    # Create input DataFrame for prediction
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

    # # Debug: Print input
    # print("Input DataFrame before preprocessing:")
    # print(X_pred)
    # print(f"Input DataFrame shape: {X_pred.shape}")
    # print(f"Input DataFrame columns: {X_pred.columns.tolist()}")

    # Preprocess the data using the preprocessor.pkl
    try:
        X_processed = preprocessor.transform(X_pred)
        # Debug: Print preprocessed DataFrame
        print("Preprocessed DataFrame:")
        print(X_processed)
        print(f"Preprocessed DataFrame shape: {X_processed.shape}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return {"error": str(e)}

    # Make prediction using the model.pkl
    try:
        y_pred = model.predict(X_processed)
        # Debug: Print prediction result
        print(f"Prediction result: {y_pred}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}

    return {
        "likely_to_be_delayed": bool(y_pred[0]),
        "input_data": X_pred.to_dict(orient="records"),  # Debugging: Include original data
        "preprocessed_data": X_processed.tolist()  # Debugging: Preprocessed data
    }

@app.get("/")
def root():
    return {'greeting': 'Hello'}