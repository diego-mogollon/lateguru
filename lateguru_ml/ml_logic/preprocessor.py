#This file focuses on scaling, feature concatenation, and PCA.

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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