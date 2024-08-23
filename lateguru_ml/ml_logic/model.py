#This file manages model creation, training, and predictions.

import xgboost as xgb

#Initialise XGBoost
def initialise_xgboost_model(max_depth=5, n_estimators=100, random_state=42):
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state,
        max_depth=max_depth,
        n_estimators=n_estimators
    )
    return model

#Fit the model
def fit_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

#Predict using trained model
def predict(model, X_pred):
    y_pred = model.predict(X_pred)
    return y_pred
