"""Response schemas for pump fault risk prediction API."""
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Single prediction response schema - MUST MATCH API SPEC EXACTLY."""
    asset_id: str = Field(..., description="Asset ID from request")
    failure_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of failure (0-1)"
    )
    fault_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the prediction (0-1)"
    )
    top_signals: List[str] = Field(
        ...,
        description="Top contributing signals to the prediction"
    )
    explanation: str = Field(
        default="",
        description="Brief machine-generated explanation of the prediction rationale"
    )
    inference_ms: int = Field(
        ...,
        ge=0,
        description="Inference time in milliseconds"
    )
    model_version: str = Field(
        ...,
        description="Model version used for prediction"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "asset_id": "pump_017",
                "failure_probability": 0.79,
                "fault_confidence": 0.73,
                "top_signals": ["vibration_spike", "acoustic_anomaly", "maintenance_overdue"],
                "explanation": "High failure risk (79%) driven by vibration_spike and acoustic_anomaly. Confidence: 73%.",
                "inference_ms": 510,
                "model_version": "v1.0.0"
            }
        }
    }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response - array of individual responses."""
    predictions: List[PredictionResponse] = Field(
        ...,
        description="Array of prediction responses in same order as requests"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "predictions": [
                    {
                        "asset_id": "pump_017",
                        "failure_probability": 0.79,
                        "fault_confidence": 0.73,
                        "top_signals": ["vibration_spike"],
                        "inference_ms": 510,
                        "model_version": "v1.0.0"
                    }
                ]
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status")
    model_version: str = Field(..., description="Loaded model version")
    uptime_s: float = Field(..., description="Service uptime in seconds")