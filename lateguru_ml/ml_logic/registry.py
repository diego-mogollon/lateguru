#This file is used for saving and loading models, as well as potentially logging model performance or tracking experiments.

#Save the model
def save_model(model, filepath):
    import joblib
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

#Load model
def load_model(filepath):
    import joblib
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model