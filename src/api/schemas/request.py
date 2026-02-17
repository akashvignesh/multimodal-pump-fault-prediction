"""Request schemas for pump fault risk prediction API."""
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Single prediction request schema."""
    asset_id: str = Field(..., description="Unique identifier for the pump asset")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the request")
    sensor_window: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Array of sensor readings with timestamps and sensor values"
    )
    image_refs: List[str] = Field(
        default_factory=list,
        description="Optional image file paths/refs relative to data/"
    )

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate timestamp is ISO 8601 format."""
        try:
            if v.endswith('Z'):
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            else:
                datetime.fromisoformat(v)
        except ValueError as e:
            raise ValueError(f"Invalid ISO 8601 timestamp format: {v}") from e
        return v

    @field_validator('sensor_window')
    @classmethod
    def validate_sensor_window(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate sensor window contains valid readings."""
        for i, reading in enumerate(v):
            if not isinstance(reading, dict):
                raise ValueError(f"sensor_window[{i}] must be a dict, got {type(reading)}")
            for key, val in reading.items():
                if key in ('timestamp', 'ts', 'time'):
                    continue
                if val is not None and not isinstance(val, (int, float)):
                    try:
                        float(val)
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"sensor_window[{i}]['{key}'] must be numeric, got {type(val)}"
                        )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "asset_id": "pump_017",
                "timestamp": "2026-02-12T10:30:00Z",
                "sensor_window": [
                    {"sensor_00": 2.44, "sensor_01": 46.31, "sensor_02": 52.34}
                ],
                "image_refs": ["images/img_00000001.png"]
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request - array of individual requests."""
    items: List[PredictionRequest] = Field(
        ...,
        description="Array of prediction requests",
        min_length=1,
        max_length=100
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "items": [
                    {
                        "asset_id": "pump_017",
                        "timestamp": "2026-02-12T10:30:00Z",
                        "sensor_window": [{"sensor_00": 2.44}],
                        "image_refs": []
                    }
                ]
            }
        }
    }