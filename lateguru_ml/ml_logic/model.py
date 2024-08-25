#This model.py file manages model creation, training, and predictions.

import xgboost as xgb

# Define the XGBClassifier 
model = xgb.XGBClassifier(
    use_label_encoder=False,  # Disable label encoder
    eval_metric='logloss',    # Logloss as the evaluation metric
    random_state=42,          # For reproducibility
    max_depth=9,              # Maximum depth of the trees
    n_estimators=900,         # Number of boosting rounds
    learning_rate=0.005,       # Learning rate for the boosting process
    n_jobs=4,                 # For testing, going with 4 instead of -1 to avoid crashing
    min_child_weight=10,       # Makes the model more conservative and prevents learning overly specific rules
    gamma=1.0,                # Makes the model more conservative by requiring a larger reduction in the loss to make a split
    scale_pos_weight=1.4965582588005606  # Balancing of positive and negative weights
)

#Fit the model
def fit_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

#Predict using trained model
def predict(model, X_pred):
    y_pred = model.predict(X_pred)
    return y_pred
