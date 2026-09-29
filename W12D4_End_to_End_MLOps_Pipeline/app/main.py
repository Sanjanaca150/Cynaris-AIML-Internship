from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="W12D4 End-to-End MLOps API")

iris = load_iris()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

model.fit(iris.data, iris.target)


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {
        "message": "W12D4 End-to-End MLOps API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "RandomForestClassifier",
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([request.features])

    return {
        "prediction": int(prediction[0]),
        "class_name": iris.target_names[prediction[0]],
    }