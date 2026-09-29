from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(
    title="W12D3 ML Monitoring API",
    description="Containerised ML API for production monitoring",
    version="1.0.0",
)

iris = load_iris()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

model.fit(iris.data, iris.target)


class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {
        "message": "W12D3 ML Monitoring API is running",
        "status": "healthy",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "RandomForestClassifier",
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    features = [
        [
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width,
        ]
    ]

    prediction = int(model.predict(features)[0])

    return {
        "prediction": prediction,
    }