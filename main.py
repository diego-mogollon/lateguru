import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from lateguru_ml.params import *
from lateguru_ml.ml_logic.data import load_preprocessed_data, define_y_and_X, split_train_test, sample_down, get_features, get_data, compress_data, check_time, add_data_features
from lateguru_ml.ml_logic.encoders import encode_categorical_features
from lateguru_ml.ml_logic.preprocessor import scale_numeric_features, concatenate_features, apply_pca, preprocess_features,
from lateguru_ml.ml_logic.model import initialise_xgboost_model, fit_model, predict
from lateguru_ml.ml_logic.registry import save_model, load_model

# Download dataset and preprocess before model training

def preprocess(min_date:str = '2021-01-01 00:00:00', max_date:str = '2023-12-31 00:00:00') -> None:

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Query raw data from BigQuery using `get_data_with_cache`
    min_date = parse(min_date).strftime('%Y-%m-%d %H:%M:%S') # e.g '2021-01-01 00:00:00'
    max_date = parse(max_date).strftime('%Y-%m-%d %H:%M:%S') # e.g '2023-12-31 00:00:00'

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM `{GCP_PROJECT}`.{BQ_DATASET}.FORWW_data_table'
        WHERE Time BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY Time
    """

    # Retrieve data using `get_data`
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("data", f"query_{min_date}_{max_date}.csv")
    data_query = get_data(
        query=query,
        gcp_project=GCP_PROJECT,
        cache_path=data_query_cache_path,
        data_has_header=True
    )

    # Compress data
    data_compress = compress_data(data_query)

    # Check 'Time' dtype
    data_time = check_time(data_compress)

    # Feature engineering



    # Process data
    preprocessed_df =

    print("✅ preprocess() done \n")

def train(
        min_date:str = '2021-01-01 00:00:00',
        max_date:str = '2015-01-01 00:00:00',
        split_ratio: float = 0.02, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
        learning_rate=0.0005,
        batch_size = 256,
        patience = 2
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    # Load processed data using `get_data_with_cache` in chronological order
    # Try it out manually on console.cloud.google.com first!

    # $CHA_BEGIN
    # Below, our columns are called ['_0', '_1'....'_66'] on BQ, student's column names may differ
    query = f"""
        SELECT * EXCEPT(_0)
        FROM `{GCP_PROJECT}`.{BQ_DATASET}.
        WHERE _0 BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY _0 ASC
    """
