from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="MLOps ML API",
    description="Containerized ML API for W12D1 MLOps CI/CD",
    version="1.0.0"
)

MODEL_PATH = Path("model.joblib")


class PredictionRequest(BaseModel):
    features: list[float]


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "MLOps ML API is running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    model = load_model()

    if model is None:
        prediction = 1 if np.mean(request.features) >= 0.5 else 0
    else:
        prediction = int(model.predict([request.features])[0])

    return {
        "prediction": prediction
    }