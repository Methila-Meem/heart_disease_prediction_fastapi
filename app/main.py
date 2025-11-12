"""
=========================================================
Model: Heart Disease Prediction (K-NearestNeighbors)
Pydantic models defined in app/schemas.py
=========================================================
"""

from fastapi import FastAPI
from pydantic import BaseModel
from app.schemas import HeartInput
from app.utils import load_artifacts, prepare_input
import numpy as np

app = FastAPI(title="Heart Didsease predictior")

# Loading model/artifacts once at startup
model, scaler, feature_names = load_artifacts()


@app.get("/info")
def info():
    """Basic model info"""
    return{
        "model_type": type(model).__name__,
        "features": feature_names
    }

@app.post("/predict")
def predict(payload: HeartInput):
    # Pydantic v2
    data = payload.model_dump()

    # Determine feature order:
    if feature_names is None:
        # Fall back to payload keys order (JSON preserves insertion order)
        feature_names_local = list(data.keys())
    else:
        feature_names_local = feature_names

    X = prepare_input(data, feature_names_local, scaler)

    # prediction & probability (handle models without predict_proba)
    try:
        proba = model.predict_proba(X)[0, 1]
    except Exception:
        pred = model.predict(X)[0]
        return {"heart_disease": bool(pred), "probability": None}

    pred = model.predict(X)[0]
    return {"heart_disease": bool(pred), "probability": float(proba)}
    