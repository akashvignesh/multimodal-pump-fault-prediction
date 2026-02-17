from fastapi import Depends
from .services.prediction_service import PredictionService
from .utils.validators import validate_input

def get_prediction_service() -> PredictionService:
    return PredictionService()

def validate_request(data: dict) -> dict:
    return validate_input(data)