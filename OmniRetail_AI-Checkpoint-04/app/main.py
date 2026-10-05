from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "models" / "best_core_model.joblib"

model = joblib.load(MODEL_FILE)


app = FastAPI(
    title="OmniRetail AI - Checkpoint 4 API",
    description="FastAPI serving layer for the OmniRetail AI core Random Forest model.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    product_views: int = Field(ge=0)
    add_to_cart: int = Field(ge=0)
    session_duration_mins: float = Field(ge=0)
    return_visitor: int = Field(ge=0, le=1)
    discount_applied: int = Field(ge=0, le=1)
    recommendation_clicked: int = Field(ge=0, le=1)
    city: str
    device: str
    category: str


@app.get("/")
def root():
    return {
        "project": "OmniRetail AI",
        "checkpoint": "Checkpoint 4",
        "service": "Core Model Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "RandomForestClassifier",
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    input_data = pd.DataFrame(
        [
            {
                "product_views": request.product_views,
                "add_to_cart": request.add_to_cart,
                "session_duration_mins": request.session_duration_mins,
                "return_visitor": request.return_visitor,
                "discount_applied": request.discount_applied,
                "recommendation_clicked": request.recommendation_clicked,
                "city": request.city,
                "device": request.device,
                "category": request.category,
            }
        ]
    )

    prediction = int(model.predict(input_data)[0])

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    return {
        "prediction": prediction,
        "purchase_probability": round(probability, 4),
        "model": "RandomForestClassifier",
    }