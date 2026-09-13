import json
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from app.security import verify_api_key
from app.config import settings
from app.models.schemas import (
    PredictionInput,
    PredictionOutput,
    PredictionBatchInput,
    PredictionBatchOutput
)


router = APIRouter(
    prefix="/api/v1",
    dependencies=[Depends(verify_api_key)]
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
    logger = request.app.state.logger
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


@router.post(
    "/predict-batch",
    response_model=PredictionBatchOutput
)
def predict_batch(
    data: PredictionBatchInput,
    request: Request
):

    model = request.app.state.model
    iris = request.app.state.iris
    logger = request.app.state.logger
    request_id = request.state.request_id

    batch_size = len(data.inputs)

    # Enforce configured batch size
    if batch_size > settings.MAX_BATCH_SIZE:

        logger.warning(
            f"Batch size exceeded | "
            f"request_id={request_id} | "
            f"batch_size={batch_size} | "
            f"max_batch_size={settings.MAX_BATCH_SIZE}"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Maximum batch size is "
                f"{settings.MAX_BATCH_SIZE}"
            )
        )

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

        # Predict the complete batch at once
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
            f"batch_size={batch_size} | "
            f"duration={duration:.4f}s"
        )

        return PredictionBatchOutput(
            predictions=results
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.error(
            f"Batch prediction failed | "
            f"request_id={request_id} | "
            f"batch_size={batch_size} | "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed"
        )


@router.get("/model-info")
def model_info(request: Request):

    logger = request.app.state.logger

    try:

        with open(
            "ml/saved_model/model_info.json",
            "r"
        ) as file:

            metadata = json.load(file)

        return metadata

    except Exception as exc:

        logger.error(
            f"Failed to load model metadata | "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Model information unavailable"
        )