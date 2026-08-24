from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from sklearn.datasets import load_iris

model = None
iris = load_iris()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = joblib.load("ml/saved_model/model.joblib")
    print("Model loaded successfully.")

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "ML API is alive"}


@app.post("/predict")
def predict():
    features = {
    "sepal_length": 6.0,
    "sepal_width": 2.9,
    "petal_length": 4.5,
    "petal_width": 1.5
}

    sample = [[
        features["sepal_length"],
        features["sepal_width"],
        features["petal_length"],
        features["petal_width"]
    ]]

    prediction = model.predict(sample)[0]
    flower_name = iris.target_names[prediction]

    return {
        "prediction": int(prediction),
        "flower": flower_name
    }