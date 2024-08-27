#This preprocessor.py file focuses on scaling, feature concatenation, and PCA.

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder

def create_preprocessing_pipeline(onehot_features, target_encoded_feature, numeric_features, binary_features, apply_pca=False, n_components=10):
    # Scaling numeric features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # One-hot encoding for most categorical features
    onehot_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Target encoding for 'Dest' feature
    target_encoder_transformer = Pipeline(steps=[
        ('target_encoder', TargetEncoder())
    ])

    # Handling binary features as-is
    binary_transformer = 'passthrough'

    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('onehot', onehot_transformer, onehot_features),
            ('target', target_encoder_transformer, target_encoded_feature),
            ('bin', binary_transformer, binary_features)
        ]
    )

    if apply_pca:
        # Enable PCA if apply_pca=True
        preprocessor = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=n_components))
        ])
    return preprocessor

def preprocess_features(X, onehot_features, target_encoded_feature, numeric_features, y, binary_features=None):
    # Ensure y is not None and has the correct length
    if y is None or len(y) == 0:
        raise ValueError("y cannot be None or empty during preprocessing.")

    if binary_features is None:
        binary_features = []

    # Debugging: Print length and content of y
    print(f"Length of y before preprocessing: {len(y)}")
    print(f"First few rows of y:\n{y.head()}")

    # Create the preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(
        onehot_features=onehot_features,
        target_encoded_feature=target_encoded_feature,
        numeric_features=numeric_features,
        binary_features=binary_features
    )

    # Fit and transform the data using the preprocessor
    try:
        X_preprocessed = preprocessor.fit_transform(X, y)
    except ValueError as e:
        print(f"Error during preprocessing: {e}")
        print(f"X shape: {X.shape}, y length: {len(y)}")
        print(f"X index: {X.index}, y index: {y.index}")
        raise e

    return X_preprocessed

'''

Below is a test pipeline by Connor for API. We will end with one pipeline after completing tests and models.

'''

def create_sklearn_preprocessor():

    categorical_features = ['Origin', 'Dest', 'Route', 'Carrier']

    numerical_features = [
        'DayOfWeek', 'HourOfDay', 'Temperature', 'Feels_Like_Temperature',
        'Altimeter_Pressure', 'Sea_Level_Pressure', 'Visibility',
        'Wind_Speed', 'Wind_Gust', 'Precipitation', 'Ice_Accretion_3hr',
        'CarrierAvgDelay', 'Month'
    ]

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])


    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])


    final_preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        n_jobs=-1
    )

    return final_preprocessor


def preprocess_features_v2(X) :

    preprocessor = create_sklearn_preprocessor()

    X_processed = preprocessor.fit_transform(X)

    X_processed = pd.DataFrame(X_processed, columns=X.columns)

    # print(f"✅ X_processed, with shape {X_processed.shape}")

    return X_processed
