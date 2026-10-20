"""Request / response schemas with input validation."""
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientFeatures(BaseModel):
    """One patient's health record (UCI Heart Disease feature set)."""

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment (1-3)")
    ca: int = Field(..., ge=0, le=3, description="Major vessels coloured by fluoroscopy (0-3)")
    thal: int = Field(..., description="3 = normal, 6 = fixed defect, 7 = reversible defect")

    @field_validator("thal")
    @classmethod
    def thal_must_be_valid(cls, v: int) -> int:
        if v not in (3, 6, 7):
            raise ValueError("thal must be one of 3, 6, 7")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
                "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
                "ca": 0, "thal": 6,
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease present, 0 = absent")
    label: str
    probability_disease: float
    confidence: float
    model_name: str
    model_version: str


class BatchRequest(BaseModel):
    records: List[PatientFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
