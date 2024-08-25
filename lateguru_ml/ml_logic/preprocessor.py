#This file focuses on scaling, feature concatenation, and PCA.

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

#Scale numeric features
def scale_numeric_features(X_train, X_test, numeric_features):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled = scaler.transform(X_test[numeric_features])

    return X_train_scaled, X_test_scaled

#Combine categorical, scaled numeric, and binary features into single array
def concatenate_features(X_train_encoded, X_train_scaled, X_train_binary, X_test_encoded, X_test_scaled, X_test_binary):
    X_train_preprocessed = np.hstack([X_train_encoded.toarray(), X_train_scaled, X_train_binary])
    X_test_preprocessed = np.hstack([X_test_encoded.toarray(), X_test_scaled, X_test_binary])

    return X_train_preprocessed, X_test_preprocessed

#Apply PCA to scaled numeric features
def apply_pca(X_train_scaled, X_test_scaled, n_components=10):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca


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

    # print(f"✅ X_processed, with shape {X_processed.shape}")

    return X_processed
