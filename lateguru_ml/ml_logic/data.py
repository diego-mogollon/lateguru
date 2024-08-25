#This data.py file manages all data-related tasks, such as loading, splitting, and feature retrieval.

import pandas as pd
from sklearn.model_selection import train_test_split

#Load preprocessed data from path
def load_preprocessed_data(file_path):
    preprocessed_df = pd.read_csv(file_path)
    return preprocessed_df

#Load data from gcp (PENDING)

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
    onehot_features = ['CancellationReason', 'Origin', 'Carrier']  # For OneHotEncoder
    target_encoded_feature = ['Dest']  # For TargetEncoder
    binary_features = ['Cancelled', 'Delayed']
    numeric_features = ['DepDelayMinutes', 'CarrierDelay', 'NASDelay',
                        'SecurityDelay', 'LateAircraftDelay', 'Temperature', 'Feels_Like_Temperature',
                        'Altimeter_Pressure', 'Sea_Level_Pressure', 'Visibility', 'Wind_Speed',
                        'Wind_Gust', 'Precipitation', 'Ice_Accretion_3hr', 'Hour', 'Day_Of_Week', 'Month']

    return onehot_features, target_encoded_feature, binary_features, numeric_features

#load airport geolocation data

def load_airport_geo_data(filepath):
   df =  pd.read_csv(filepath)
   return df
