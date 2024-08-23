#This file focuses on scaling, feature concatenation, and PCA.

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

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


def preprocess_features(df):

    #instantiating a standard scaler
    scaler = StandardScaler()
    #instantiating a PCA function with 10 principal components
    pca = PCA(n_components=10)
    #instantiating a OneHotEncoder, check on sparse output
    ohe = OneHotEncoder(sparse_output=False, drop='if_binary')

    #Defining subsets of X features based on data type
    X_num = df.select_dtypes(include='number')
    X_cat = df.select_dtypes(include='object')
    X_binary = df.select_dtypes(include='bool')

    #scaling numerical features and encoding categorical features
    X_num_scaled = scaler.transform(X_num)
    X_cat_encoded = ohe.transform(X_cat)

    #Joining the preprocessed features back up with each other
    X_pred = np.hstack([X_cat_encoded.toarray()], X_num_scaled, X_binary)

    #transforming the data with PCA
    X_pred = pca.transform(X_pred)

    #returning fully preprocessed features, output as a numpy array
    return X_pred
