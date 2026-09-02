from typing import List

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    sepal_length: float = Field(
        ..., gt=0, le=10, description="Sepal length in cm"
    )
    sepal_width: float = Field(
        ..., gt=0, le=10, description="Sepal width in cm"
    )
    petal_length: float = Field(
        ..., gt=0, le=10, description="Petal length in cm"
    )
    petal_width: float = Field(
        ..., gt=0, le=10, description="Petal width in cm"
    )
    
    
class PredictionOutput(BaseModel):
    request_id: str
    prediction: int
    flower: str
    confidence: float
    model_version: str
    
class PredictionBatchInput(BaseModel):
    inputs: List[PredictionInput] = Field(..., min_length=1, max_length=100)


class PredictionBatchOutput(BaseModel):
    predictions: List[PredictionOutput]