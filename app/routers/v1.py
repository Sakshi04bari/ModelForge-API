from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import PredictionInput, PredictionOutput


router = APIRouter(prefix="/api/v1")


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