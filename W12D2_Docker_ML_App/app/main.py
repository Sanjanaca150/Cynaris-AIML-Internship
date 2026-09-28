from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Iris ML Prediction API")


iris = load_iris()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

model.fit(iris.data, iris.target)


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {"message": "Iris ML API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction),
        "class_name": iris.target_names[prediction],
    }