#This file manages all data-related tasks, such as loading, splitting, and feature retrieval.

import pandas as pd
from sklearn.model_selection import train_test_split

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from lateguru_ml.params import *

#Load preprocessed data from path
def load_preprocessed_data(file_path):
    preprocessed_df = pd.read_csv(file_path)
    return preprocessed_df

#Load data from gcp
def get_data(
        gcp_project:str,
        query:str,
        cache_path:Path,
        data_has_header=True
    ) -> pd.DataFrame:

    # Checking if cache_path already exists otherwise will download from BigQuery
    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    else:
        print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df

def check_time(df):
    # Check the datatype of the 'Time' column
    print(f"Data type of 'Time': {df['Time'].dtype}")

    # If not datetime, convert it to datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Time']):
    df['Time'] = pd.to_datetime(df['Time'])
    print(f"After conversion, Data type of 'Time': {df['Time'].dtype}")

    return df

def add_data_features(df):
    # Add time based columns
    df['Hour'] = df['Time'].dt.hour
    df['Day_Of_Week'] = df['Time'].dt.dayofweek
    df['Month'] = df['Time'].dt.month

    # Adding origin and arrival feature
    df['Route'] = df['Origin'] + '_' + df['Dest']

#Define X and y
def define_X_and_y(preprocessed_df):
    # Define X and y
    X = preprocessed_df.drop(columns=['Weather_Delay_Length', 'Weather_Delayed'])
    y = preprocessed_df['Weather_Delayed']

    # Scale down data types for 'int' and 'float' columns
    for col in X.select_dtypes(include=['int']).columns:
        X[col] = X[col].astype('int32')

    for col in X.select_dtypes(include=['float']).columns:
        X[col] = X[col].astype('float32')

    return X, y

#Define Split_train_test
def split_train_test(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

#Define Sampling method to reduce data size
def sample_down(X, y, sample_size=0.01, random_state=42):
    X_sample, _, y_sample, _ = train_test_split(X, y, test_size=1-sample_size, random_state=random_state, stratify=y)
    return X_sample, y_sample

#Define categorical, binary, numeric features
def get_features():
    categorical_features = ['CancellationReason', 'Origin', 'Dest', 'Carrier']
    binary_features = ['Cancelled', 'Delayed']
    numeric_features = ['DepDelayMinutes', 'CarrierDelay', 'NASDelay',
                        'SecurityDelay', 'LateAircraftDelay', 'Temperature', 'Feels_Like_Temperature',
                        'Altimeter_Pressure', 'Sea_Level_Pressure', 'Visibility', 'Wind_Speed',
                        'Wind_Gust', 'Precipitation', 'Ice_Accretion_3hr', 'Hour', 'Day_Of_Week', 'Month']

    return categorical_features, binary_features, numeric_features

#load airport geolocation data

def load_airport_geo_data(filepath):
   df =  pd.read_csv(filepath)
   return df

# Basic compression of data for efficiency
def compress_data(df: pd.DataFrame) -> pd.DataFrame:

    # Compress raw_data by setting types to DTYPES_RAW
    df = df.astype(DTYPES_RAW)

    print("✅ data compressed")

    return df

# Can make a def clean_data(df) function if we have time

# Upload data to BigQuery
def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)

    # Load data onto full_table_name

    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")
