#This registry.py file handles the saving/loading of pkl file and in the future may be used for MLFlow/Prefect

import os
import joblib
from lateguru_ml.params import *

# Define the directory path where models should be saved and loaded
model_directory = os.path.join(os.path.dirname(__file__), '..', '..', 'model')

# Save the model
def save_model(model, filename):
    filepath = os.path.join(model_directory, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

# Load the model
def load_model(filename):
    filepath = os.path.join(model_directory, filename)
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
