#this main.py file will handle all the ml_logic logic, calling main functions from params.py, utils.py, data.py, encoder.py, preprocessor.py, model.py, and registry.py

import numpy as np
import pandas as pd
import os
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from lateguru_ml.ml_logic.data import (
    load_preprocessed_data,
    check_time,
    add_data_features,
    define_X_and_y,
    split_train_test,
    sample_down,
    get_features,
    upload_model_to_gcs
)
from lateguru_ml.ml_logic.preprocessor import create_preprocessing_pipeline, preprocess_features
from lateguru_ml.ml_logic.model import model as xgb_model, fit_model
from lateguru_ml.params import *

# Define file paths
DATA_FILE = 'Top_5_Airports.csv'
DATA_DIR = 'data'
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_FILE = os.path.join(MODEL_DIR, 'xgb_model.pkl')

def main():
    # Load the preprocessed data
    print("Loading preprocessed data...")
    preprocessed_df = load_preprocessed_data(DATA_FILE, DATA_DIR)

    # or use below for data table from BQ
    # Define parameters for query
    # query = f"""
    #         SELECT *
    #         FROM `{GCP_PROJECT}.{BQ_DATASET}.{BQ_DATA_TABLE}`
    #         """

    # #  Get data from local file or BigQuery
    # preprocessed_df = get_data(
    #         gcp_project=GCP_PROJECT,
    #         query=query,
    #         cache_path=LOCAL_DATA_PATH,
    #         data_has_header=True
    #     )

    # Ensure the 'Time' column is correctly formatted
    print("Checking and formatting 'Time' column...")
    preprocessed_df = check_time(preprocessed_df)

    # Add time-based features and carrier average delay
    print("Adding time-based features and carrier average delay...")
    preprocessed_df = add_data_features(preprocessed_df)

    # Verify the DataFrame to ensure all required columns are present
    print("DataFrame columns after adding features:", preprocessed_df.columns.tolist())

    # Define X and y
    print("Defining X and y...")
    X, y = define_X_and_y(preprocessed_df)

    # Sample down the dataset to 1%
    print("Sampling down the dataset...")
    X_sampled, y_sampled = sample_down(X, y, sample_size=0.01)

    # Reset index after sampling
    X_sampled = X_sampled.reset_index(drop=True)
    y_sampled = y_sampled.reset_index(drop=True)

    # Split the data into train and test sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_train_test(X_sampled, y_sampled)

    # Reset index after splitting
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Ensure indices are aligned before preprocessing
    y_train.index = X_train.index
    y_test.index = X_test.index

    # Get feature lists for preprocessing
    print("Retrieving feature lists for preprocessing...")
    onehot_features, target_encoded_feature, numeric_features = get_features()

    # Create preprocessing pipeline
    print("Creating preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline(
        onehot_features=onehot_features,
        target_encoded_feature=target_encoded_feature,
        numeric_features=numeric_features,
        binary_features=[],  # No binary features
        apply_pca=False  # Adjust if using PCA (not needed for this new data set - may remove PCA function this week)
    )

    # Preprocess features - Important: TargetEncoder requires to pass 'y'
    print("Preprocessing features...")
    try:
        X_train_preprocessed = preprocess_features(X_train, onehot_features, target_encoded_feature, numeric_features, y_train, [])
        X_test_preprocessed = preprocess_features(X_test, onehot_features, target_encoded_feature, numeric_features, y_test, [])
    except ValueError as e:
        print(f"Error during preprocessing: {e}")
        print(f"X_train index: {X_train.index}, y_train index: {y_train.index}")
        return

    # Train the model
    print("Training the model...")
    model = fit_model(xgb_model, X_train_preprocessed, y_train)

    # Evaluate the model
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")

    print("Evaluating the model...")
    evaluate_model(model, X_test_preprocessed, y_test)

    # Save the model to a pkl file
    print("Saving the model to a pkl file...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    dump(model, MODEL_FILE)

    print(f"Model saved to {MODEL_FILE}")

    # Upload model pickle file to GCS
    upload_model_to_gcs(
        bucket_name=BUCKET_NAME,
        source_file_name=MODEL_FILE,
        destination_blob_name="model_training/model.pkl"
    )

    #Set learning curve
    def plot_learning_curve(model, X, y):
        train_sizes, train_scores, validation_scores = learning_curve(
            model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='recall'
        )

        train_scores_mean = train_scores.mean(axis=1)
        train_scores_std = train_scores.std(axis=1)
        validation_scores_mean = validation_scores.mean(axis=1)
        validation_scores_std = validation_scores.std(axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, label='Training score', color='blue')
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.2, color='blue')
        plt.plot(train_sizes, validation_scores_mean, label='Validation score', color='green')
        plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                        validation_scores_mean + validation_scores_std, alpha=0.2, color='green')
        plt.title('Learning Curve for Lateguru XGBoost Model')
        plt.xlabel('Training Set Size')
        plt.ylabel('Recall Score')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    # Plot learning curve
    print("Plotting learning curve...")
    plot_learning_curve(model, X_train_preprocessed, y_train)

if __name__ == "__main__":
    main()
