import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from app.models.schemas import PredictionInput
from app.security import verify_api_key
router = APIRouter(
    prefix="/api/v2",
    dependencies=[Depends(verify_api_key)]
)

@router.post("/predict")
def predict_v2(data: PredictionInput, request: Request):

    model = request.app.state.model
    iris = request.app.state.iris

    sample = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(sample)[0]

    probabilities = model.predict_proba(sample)[0]

    probability_distribution = {
        iris.target_names[i]: float(probabilities[i])
        for i in range(len(probabilities))
    }

    return {
        "request_id": str(uuid.uuid4()),
        "prediction": int(prediction),
        "flower": iris.target_names[prediction],
        "probabilities": probability_distribution,
        "model_version": "2.0"
    }