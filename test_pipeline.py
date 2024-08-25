import numpy as np
from lateguru_ml.ml_logic.data import load_preprocessed_data, define_y_and_X, split_train_test, sample_down, get_features
from lateguru_ml.ml_logic.encoders import encode_categorical_features
from lateguru_ml.ml_logic.preprocessor import scale_numeric_features, concatenate_features, apply_pca, preprocess_features
from lateguru_ml.ml_logic.model import initialise_xgboost_model, fit_model, predict
from lateguru_ml.ml_logic.registry import save_model, load_model

# Define the file path to the preprocessed data
file_path = 'data/preprocessed_treated_outliers.csv'

# Load the preprocessed data
preprocessed_df = load_preprocessed_data(file_path)

# Define X and y
X, y = define_y_and_X(preprocessed_df)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = split_train_test(X, y)

# Sample down the data for testing (very small sample for local testing only!)
X_sample, y_sample = sample_down(X, y, sample_size=0.005)

# Get feature lists
categorical_features, binary_features, numeric_features = get_features()

# Encode categorical features
X_train_encoded, X_test_encoded = encode_categorical_features(X_train, X_test, categorical_features)

# Scale numeric features and test output
X_train_scaled, X_test_scaled = scale_numeric_features(X_train, X_test, numeric_features)

# Ensure that scaling produces arrays with the correct shape
assert X_train_scaled.shape == (X_train.shape[0], len(numeric_features)), "Scaling failed on training data."
assert X_test_scaled.shape == (X_test.shape[0], len(numeric_features)), "Scaling failed on test data."

# Apply PCA and test output
X_train_pca, X_test_pca = apply_pca(X_train_scaled, X_test_scaled, n_components=10)

# Ensure that PCA produces the correct number of components
assert X_train_pca.shape[1] == 10, "PCA did not produce the correct number of components for training data."
assert X_test_pca.shape[1] == 10, "PCA did not produce the correct number of components for test data."

# Combine all features into a single array
X_train_preprocessed, X_test_preprocessed = concatenate_features(X_train_encoded, X_train_scaled, X_train[binary_features],
                                                                 X_test_encoded, X_test_scaled, X_test[binary_features])

# Ensure the feature concatenation process outputs the correct shape
expected_train_shape = (X_train_encoded.shape[0], X_train_encoded.shape[1] + X_train_scaled.shape[1] + X_train[binary_features].shape[1])
expected_test_shape = (X_test_encoded.shape[0], X_test_encoded.shape[1] + X_test_scaled.shape[1] + X_test[binary_features].shape[1])

assert X_train_preprocessed.shape == expected_train_shape, "Feature concatenation failed for training data."
assert X_test_preprocessed.shape == expected_test_shape, "Feature concatenation failed for test data."

# Initialise XGBoost model
xgb_model = initialise_xgboost_model()

# Fit the model on preprocessed data
xgb_model = fit_model(xgb_model, X_train_preprocessed, y_train)

# Predict using the trained model
y_pred = predict(xgb_model, X_test_preprocessed)

# Save the model
save_model(xgb_model, 'test_xgb_model.pkl')

# Load the model back
loaded_model = load_model('test_xgb_model.pkl')

# Predict again with the loaded model
y_pred_loaded = predict(loaded_model, X_test_preprocessed)

# Verify that the predictions match between the original and loaded models
assert np.array_equal(y_pred, y_pred_loaded), "Predictions from the original and loaded models do not match!"

print("All tests passed successfully. Model saved and predictions are consistent.")