import os
import numpy as np

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "lateguru", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "lateguru", "model")

COLUMN_NAMES_RAW = ['Time', 'Origin', 'Dest', 'Carrier', 'Cancelled', 'CancellationReason',
       'Delayed', 'DepDelayMinutes', 'CarrierDelay', 'Weather_Delay_Length',
       'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'Temperature',
       'Feels_Like_Temperature', 'Altimeter_Pressure', 'Sea_Level_Pressure',
       'Visibility', 'Wind_Speed', 'Wind_Gust', 'Precipitation',
       'Ice_Accretion_3hr', 'Weather_Delayed']

DTYPES_RAW = {
    "Time": "object",
    "Origin": "object",
    "Dest": "object",
    "Carrier": "object",
    "Cancelled": "bool",
    "CancellationReason": "object",
    "Delayed": "bool",
    "DepDelayMinutes": "float16",
    "CarrierDelay": "float16",
    "Weather_Delay_Length": "float16",
    "NASDelay": "float16",
    "SecurityDelay": "float16",
    "LateAircraftDelay": "float16",
    "Temperature": "float16",
    "Feels_Like_Temperature": "float16",
    "Altimeter_Pressure": "float16",
    "Sea_Level_Pressure": "float16",
    "Visibility": "float16",
    "Wind_Speed": "float16",
    "Wind_Gust": "float16",
    "Precipitation": "float16",
    "Ice_Accretion_3hr": "float16",
    "Weather_Delayed": "bool"
}

DTYPES_PROCESSED = np.float32

OW_API_KEY=os.environ.get("OW_API_KEY")

################## VALIDATIONS #################

# env_valid_options = dict(
#     MODEL_TARGET=["local", "gcs", "mlflow"],
# )

# def validate_env_value(env, valid_options):
#     env_value = os.environ[env]
#     if env_value not in valid_options:
#         raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")

# for env, valid_options in env_valid_options.items():
#     validate_env_value(env, valid_options)
