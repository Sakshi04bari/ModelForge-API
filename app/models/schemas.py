from typing import List

from pydantic import BaseModel, ConfigDict, Field


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
    model_config = ConfigDict(extra="forbid")
    request_id: str
    prediction: int
    flower: str
    confidence: float
    model_version: str
    
class PredictionBatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: List[PredictionInput] = Field(..., min_length=1, max_length=100)


class PredictionBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predictions: List[PredictionOutput]