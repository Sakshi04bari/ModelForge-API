import json
import time
from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import PredictionInput, PredictionOutput


router = APIRouter(prefix="/api/v1")

from app.models.schemas import (
    PredictionInput,
    PredictionOutput,
    PredictionBatchInput,
    PredictionBatchOutput
)

@router.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": request.app.state.model is not None
    }


@router.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput, request: Request):

    model = request.app.state.model
    iris = request.app.state.iris
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

        return {
            "request_id": request_id,
            "prediction": int(prediction),
            "flower": flower_name,
            "confidence": confidence,
            "model_version": "1.0"
        }

    except Exception as exc:

        request.app.state.logger.error(
            f"Prediction failed | "
            f"request_id={request_id} | "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )
@router.post("/predict-batch", response_model=PredictionBatchOutput)
def predict_batch(
    data: PredictionBatchInput,
    request: Request
):
    request_id = request.state.request_id
    model = request.app.state.model
    logger = request.app.state.logger
    iris = request.app.state.iris

    start_time = time.perf_counter()

    try:
        samples = [
            [
                item.sepal_length,
                item.sepal_width,
                item.petal_length,
                item.petal_width
            ]
            for item in data.inputs
        ]

        # Predict the entire batch at once
        predictions = model.predict(samples)
        probabilities = model.predict_proba(samples)

        results = []

        for prediction, probability in zip(
            predictions,
            probabilities
        ):
            confidence = float(max(probability))
            flower_name = iris.target_names[prediction]

            results.append(
                PredictionOutput(
                    request_id=request_id,
                    prediction=int(prediction),
                    flower=flower_name,
                    confidence=confidence,
                    model_version="1.0"
                )
            )

        duration = time.perf_counter() - start_time

        logger.info(
            f"Batch prediction successful | "
            f"request_id={request_id} | "
            f"batch_size={len(data.inputs)} | "
            f"duration={duration:.4f}s"
        )

        return PredictionBatchOutput(
            predictions=results
        )

    except Exception as exc:

        logger.error(
            f"Batch prediction failed | "
            f"request_id={request_id} | "
            f"batch_size={len(data.inputs)} | "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed"
        )            
        
@router.get("/model-info")
def model_info(request: Request):

    try:
        with open(
            "ml/saved_model/model_info.json",
            "r"
        ) as file:
            metadata = json.load(file)

        return metadata

    except Exception as exc:

        request.app.state.logger.error(
            f"Failed to load model metadata | error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Model information unavailable"
        )        