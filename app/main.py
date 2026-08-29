from contextlib import asynccontextmanager
import time
import uuid

import joblib
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from sklearn.datasets import load_iris

from app.models.schemas import PredictionInput, PredictionOutput
from app.logging_config import setup_logger


model = None
iris = load_iris()
logger = setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = joblib.load("ml/saved_model/model.joblib")
    logger.info("Model loaded successfully")

    yield


app = FastAPI(lifespan=lifespan)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        duration = time.perf_counter() - start_time

        logger.info(
            f"request_id={request_id} "
            f"method={request.method} "
            f"path={request.url.path} "
            f"status_code={response.status_code} "
            f"duration={duration:.4f}s"
        )

        return response

    except Exception as exc:
        duration = time.perf_counter() - start_time

        logger.error(
            f"request_id={request_id} "
            f"method={request.method} "
            f"path={request.url.path} "
            f"duration={duration:.4f}s "
            f"error={exc}"
        )

        raise

@app.get("/")
def root():
    return {"message": "ML API is alive"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid prediction data",
            "detail": "The provided data could not be processed."
        }
    )


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput, request: Request):

    request_id = request.state.request_id

    try:
        sample = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]

        prediction = model.predict(sample)[0]

        probabilities = model.predict_proba(sample)[0]
        confidence = float(max(probabilities))

        flower_name = iris.target_names[prediction]

        logger.info(
            f"Prediction successful | "
            f"request_id={request_id} | "
            f"prediction={flower_name} | "
            f"confidence={confidence:.4f}"
        )

        return {
            "request_id": request_id,
            "prediction": int(prediction),
            "flower": flower_name,
            "confidence": confidence,
            "model_version": "1.0"
        }

    except Exception as exc:

        logger.error(
            f"Prediction failed | "
            f"request_id={request_id} | "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )