#This file handles the encoding of categorical features.

from sklearn.preprocessing import OneHotEncoder

#One hot encoding for categorical features
def encode_categorical_features(X_train, X_test, categorical_features):
    encoder = OneHotEncoder()
    X_train_encoded = encoder.fit_transform(X_train[categorical_features])
    X_test_encoded = encoder.transform(X_test[categorical_features])
    
    return X_train_encoded, X_test_encoded