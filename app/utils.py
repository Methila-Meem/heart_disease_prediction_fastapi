import joblib
import numpy as np
import pandas as pd
import os
from sklearn.base import BaseEstimator

MODEL_PATH = os.path.join("model", "heart_model.joblib")

def load_artifacts(path=MODEL_PATH):
    loaded = joblib.load(path)
    # If the saved file is a dict with keys 'model', 'scaler', 'feature_names'
    if isinstance(loaded, dict):
        model = loaded.get("model")
        scaler = loaded.get("scaler", None)
        feature_names = loaded.get("feature_names", None)
        return model, scaler, feature_names

    # If the saved file is a scikit-learn estimator (directly saved)
    if isinstance(loaded, BaseEstimator):
        # Return model, but no scaler/feature_names
        return loaded, None, None

    raise ValueError(f"Unexpected model format loaded from {path}: type={type(loaded)}")

def prepare_input(data: dict, feature_names, scaler=None):
    # Build DataFrame in the requested order
    df = pd.DataFrame([data], columns=feature_names)
    if scaler is not None:
        X = scaler.transform(df)
    else:
        X = df.values  # 2D array
    return X