#This encoders.py file handles the encoding of categorical features.

from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder  # Import TargetEncoder

# Target encoding for categorical features
def encode_categorical_features(X_train, X_test, y_train, categorical_features):
    encoder = TargetEncoder()  # Initialize TargetEncoder

    # Fit the encoder on the training data and transform both training and test data
    X_train_encoded = encoder.fit_transform(X_train[categorical_features], y_train)
    X_test_encoded = encoder.transform(X_test[categorical_features])
    
    # Replace the original categorical columns with the encoded values
    X_train = X_train.drop(categorical_features, axis=1)
    X_test = X_test.drop(categorical_features, axis=1)
    X_train = X_train.join(X_train_encoded)
    X_test = X_test.join(X_test_encoded)
    
    return X_train, X_test