# import numpy as np
# import pandas as pd

# from pathlib import Path
# from dateutil.parser import parse

# from lateguru_ml.params import *
# from lateguru_ml.ml_logic.data import load_preprocessed_data, define_y_and_X, split_train_test, sample_down, get_features, get_data, compress_data, check_time, add_data_features
# from lateguru_ml.ml_logic.encoders import encode_categorical_features
# from lateguru_ml.ml_logic.preprocessor import scale_numeric_features, concatenate_features, apply_pca, preprocess_features,
# from lateguru_ml.ml_logic.model import initialise_xgboost_model, fit_model, predict
# from lateguru_ml.ml_logic.registry import save_model, load_model

import os
import pandas as pd
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from lateguru_ml.ml_logic.data import (
    load_preprocessed_data,
    check_time,
    add_data_features,
    define_X_and_y,
    split_train_test,
    sample_down,
    get_features,
    scale_down_data_types
)
from lateguru_ml.ml_logic.preprocessor import create_preprocessing_pipeline, preprocess_features 
from lateguru_ml.ml_logic.model import model as xgb_model, fit_model

# Define file paths
DATA_FILE = 'Top_5_Airports.csv'
DATA_DIR = 'data'
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_FILE = os.path.join(MODEL_DIR, 'xgb_model.pkl')

def main():
    # Load the preprocessed data
    print("Loading preprocessed data...")
    preprocessed_df = load_preprocessed_data(DATA_FILE, DATA_DIR)
    
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
    
    # Sample down the dataset
    print("Sampling down the dataset...")
    X_sampled, y_sampled = sample_down(X, y, sample_size=0.01)
    
    # Split the data into train and test sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_train_test(X_sampled, y_sampled)
    
    # Get feature lists for preprocessing
    print("Retrieving feature lists for preprocessing...")
    onehot_features, target_encoded_feature, numeric_features = get_features()
    
    # Create preprocessing pipeline
    print("Creating preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline(
        onehot_features=onehot_features,
        target_encoded_feature=target_encoded_feature,
        numeric_features=numeric_features,
        binary_features=[],  # No binary features in the current setup
        apply_pca=False  # Adjust if you want to use PCA
    )
    
    # Preprocess features
    print("Preprocessing features...")
    X_train_preprocessed = preprocess_features(X_train, onehot_features, target_encoded_feature, numeric_features, [])
    X_test_preprocessed = preprocess_features(X_test, onehot_features, target_encoded_feature, numeric_features, [])
    
    # Train the model
    print("Training the model...")
    model = fit_model(xgb_model, X_train_preprocessed, y_train)
    
    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test_preprocessed, y_test)
    
    # Save the model to a pkl file
    print("Saving the model to a pkl file...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    dump(model, MODEL_FILE)
    
    print(f"Model saved to {MODEL_FILE}")
    
    # Plot learning curve
    print("Plotting learning curve...")
    plot_learning_curve(model, X_train_preprocessed, y_train)
    
def evaluate_model(model, X_test, y_test):
    """Evaluate the model using various performance metrics."""
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

def plot_learning_curve(model, X, y):
    """Plot the learning curve to show the model's performance."""
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
    plt.title('Optimized Learning Curve for XGBClassifier')
    plt.xlabel('Training Set Size')
    plt.ylabel('Recall Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()


"""

CODE BY MARK - YET TO BE TESTED

"""
# Download dataset and preprocess before model training

# def preprocess(min_date:str = '2021-01-01 00:00:00', max_date:str = '2023-12-31 00:00:00') -> None:

#     print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

#     # Query raw data from BigQuery using `get_data_with_cache`
#     min_date = parse(min_date).strftime('%Y-%m-%d %H:%M:%S') # e.g '2021-01-01 00:00:00'
#     max_date = parse(max_date).strftime('%Y-%m-%d %H:%M:%S') # e.g '2023-12-31 00:00:00'

#     query = f"""
#         SELECT {",".join(COLUMN_NAMES_RAW)}
#         FROM `{GCP_PROJECT}`.{BQ_DATASET}.FORWW_data_table'
#         WHERE Time BETWEEN '{min_date}' AND '{max_date}'
#         ORDER BY Time
#     """

#     # Retrieve data using `get_data`
#     data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("data", f"query_{min_date}_{max_date}.csv")
#     data_query = get_data(
#         query=query,
#         gcp_project=GCP_PROJECT,
#         cache_path=data_query_cache_path,
#         data_has_header=True
#     )

#     # Compress data
#     data_compress = compress_data(data_query)

#     # Check 'Time' dtype
#     data_time = check_time(data_compress)

#     # Feature engineering



#     # Process data
#     preprocessed_df =

#     print("✅ preprocess() done \n")

# def train(
#         min_date:str = '2021-01-01 00:00:00',
#         max_date:str = '2015-01-01 00:00:00',
#         split_ratio: float = 0.02, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
#         learning_rate=0.0005,
#         batch_size = 256,
#         patience = 2
#     ) -> float:

#     """
#     - Download processed data from your BQ table (or from cache if it exists)
#     - Train on the preprocessed dataset (which should be ordered by date)
#     - Store training results and model weights

#     Return val_mae as a float
#     """

#     print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
#     print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

#     min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
#     max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

#     # Load processed data using `get_data_with_cache` in chronological order
#     # Try it out manually on console.cloud.google.com first!

#     # $CHA_BEGIN
#     # Below, our columns are called ['_0', '_1'....'_66'] on BQ, student's column names may differ
#     query = f"""
#         SELECT * EXCEPT(_0)
#         FROM `{GCP_PROJECT}`.{BQ_DATASET}.
#         WHERE _0 BETWEEN '{min_date}' AND '{max_date}'
#         ORDER BY _0 ASC
#     """
