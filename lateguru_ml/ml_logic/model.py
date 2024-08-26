#This model.py file manages model creation, training, and predictions.

import xgboost as xgb

# Define the XGBClassifier 
model = xgb.XGBClassifier(
    eval_metric='logloss',    # Logloss as the evaluation metric
    random_state=42,          # For reproducibility
    max_depth=8,              # Maximum depth of the trees
    n_estimators=800,         # Number of boosting rounds
    learning_rate=0.005,       # Learning rate for the boosting process
    n_jobs=4,                 # Number of parallel threads
    min_child_weight=15,       # Minimum sum of instance weight (hessian) needed in a child
    gamma=1.0,                # Minimum loss reduction required to make a further partition
    scale_pos_weight=1.4965582588005606,  # Balancing of positive and negative weights
    alpha=2.0,
    reg_lambda=3.0
)

#Fit the model
def fit_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

#Predict using trained model
def predict(model, X_pred):
    y_pred = model.predict(X_pred)
    return y_pred
