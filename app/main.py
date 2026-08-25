from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from sklearn.datasets import load_iris
from app.models.schemas import PredictionInput
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
def predict(data: PredictionInput):
    sample = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(sample)[0]
    flower_name = iris.target_names[prediction]

    return {
        "prediction": int(prediction),
        "flower": flower_name
    }