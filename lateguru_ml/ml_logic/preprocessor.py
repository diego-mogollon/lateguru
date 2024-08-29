#This preprocessor.py file focuses on scaling, feature concatenation, and PCA.

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder

def create_numeric_transformer():
    print("Creating numeric transformer...")
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    print(f"Numeric transformer: {numeric_transformer}")
    return numeric_transformer

def create_onehot_transformer():
    print("Creating one-hot encoder transformer...")
    onehot_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    print(f"One-hot transformer: {onehot_transformer}")
    return onehot_transformer

def create_target_encoder_transformer():
    print("Creating target encoder transformer...")
    target_encoder_transformer = Pipeline(steps=[
        ('target_encoder', TargetEncoder())
    ])
    print(f"Target encoder transformer: {target_encoder_transformer}")
    return target_encoder_transformer

def create_column_transformer(numeric_transformer, onehot_transformer, target_encoder_transformer, numeric_features, onehot_features, target_encoded_feature):
    print("Creating ColumnTransformer...")
    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('onehot', onehot_transformer, onehot_features),
        ('target', target_encoder_transformer, target_encoded_feature)
    ]
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    print(f"ColumnTransformer configuration: {preprocessor}")
    return preprocessor

def create_preprocessing_pipeline(onehot_features, target_encoded_feature, numeric_features, apply_pca=False, n_components=10):
    numeric_transformer = create_numeric_transformer()
    onehot_transformer = create_onehot_transformer()
    target_encoder_transformer = create_target_encoder_transformer()

    preprocessor = create_column_transformer(
        numeric_transformer, onehot_transformer, target_encoder_transformer,
        numeric_features, onehot_features, target_encoded_feature
    )

    if apply_pca:
        print("Adding PCA to the preprocessing pipeline...")
        preprocessor = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=n_components))
        ])
        print(f"Preprocessor with PCA: {preprocessor}")
    
    return preprocessor

def preprocess_features(X, onehot_features, target_encoded_feature, numeric_features, y):
    print("Starting preprocessing of features...")

    if y is None or len(y) == 0:
        raise ValueError("y cannot be None or empty during preprocessing.")
    
    print(f"Length of y before preprocessing: {len(y)}")
    print(f"First few rows of y:\n{y.head()}")
    print(f"Columns in X before preprocessing: {X.columns.tolist()}")
    all_features = onehot_features + target_encoded_feature + numeric_features
    print(f"All features to be transformed: {all_features}")

    missing_features = [feature for feature in all_features if feature not in X.columns]
    if missing_features:
        print(f"Missing features in X: {missing_features}")
        raise ValueError(f"X is missing required features: {missing_features}")

    preprocessor = create_preprocessing_pipeline(
        onehot_features=onehot_features,
        target_encoded_feature=target_encoded_feature,
        numeric_features=numeric_features
    )

    print("Fitting the preprocessor and transforming the data...")
    try:
        X_preprocessed = preprocessor.fit_transform(X, y)
        print(f"Preprocessing complete. Shape of preprocessed data: {X_preprocessed.shape}")
    except ValueError as e:
        print(f"Error during preprocessing: {e}")
        raise e
    except KeyError as e:
        print(f"KeyError during preprocessing: {e}")
        raise e

    return X_preprocessed

'''

OLD PIPELINE BY CONNOR HERE BELOW. WE WILL DELETE AFTER CONFIRMING FAST.PY WORKS

'''

# def create_sklearn_preprocessor():

#     categorical_features = ['Origin', 'Dest', 'Route', 'Carrier']

#     numerical_features = [
#         'DayOfWeek', 'HourOfDay', 'Temperature', 'Feels_Like_Temperature',
#         'Altimeter_Pressure', 'Sea_Level_Pressure', 'Visibility',
#         'Wind_Speed', 'Wind_Gust', 'Precipitation', 'Ice_Accretion_3hr',
#         'CarrierAvgDelay', 'Month'
#     ]

#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
#     ])


#     numerical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='mean')),
#         ('scaler', StandardScaler())
#     ])


#     final_preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numerical_transformer, numerical_features),
#             ('cat', categorical_transformer, categorical_features)
#         ],
#         n_jobs=-1
#     )

#     return final_preprocessor


# def preprocess_features_v2(X) :

#     preprocessor = create_sklearn_preprocessor()

#     X_processed = preprocessor.fit_transform(X)

#     X_processed = pd.DataFrame(X_processed, columns=X.columns)

#     # print(f"✅ X_processed, with shape {X_processed.shape}")

#     return X_processed
