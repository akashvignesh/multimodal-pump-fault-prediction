"""Prediction endpoints for pump fault risk API."""
import json
import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from src.api.schemas.request import PredictionRequest, BatchPredictionRequest
from src.api.schemas.response import PredictionResponse, BatchPredictionResponse
from src.services.orchestrator import get_orchestrator

logger = logging.getLogger(__name__)
router = APIRouter(tags=["prediction"])


# ------------------------------------------------------------------
# Prediction endpoints
# ------------------------------------------------------------------

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single prediction",
    description="Predict pump fault risk for a single asset using sensor data and/or image refs",
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        orchestrator = get_orchestrator()
        return await orchestrator.predict(request)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch prediction",
    description="Predict pump fault risk for multiple assets",
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        orchestrator = get_orchestrator()
        if len(request.items) > 100:
            raise HTTPException(status_code=400, detail="Batch size exceeds 100")
        predictions = await orchestrator.predict_batch(request.items)
        return BatchPredictionResponse(predictions=predictions)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Multimodal file-upload endpoint
# ------------------------------------------------------------------

@router.post(
    "/predict/multimodal",
    response_model=PredictionResponse,
    summary="Multimodal file-upload prediction",
    description="Upload images and/or PDFs and optionally sensor JSON. "
    "Any combination of sensor + image inputs is accepted.",
)
async def predict_multimodal(
    asset_id: str = Form("upload", description="Asset identifier"),
    sensor_json: Optional[str] = Form(None, description="Optional JSON array of sensor dicts"),
    images: List[UploadFile] = File(default=[], description="Image files"),
    pdfs: List[UploadFile] = File(default=[], description="PDF files"),
):
    """Accept any combination of sensor data + image/PDF files."""
    orchestrator = get_orchestrator()

    # Parse optional sensor JSON
    sensor_window = None
    if sensor_json:
        try:
            sensor_window = json.loads(sensor_json)
            if not isinstance(sensor_window, list):
                sensor_window = None
        except json.JSONDecodeError:
            pass

    image_bytes_list = []
    for img_file in images:
        image_bytes_list.append(await img_file.read())

    pdf_bytes_list = []
    for pdf_file in pdfs:
        pdf_bytes_list.append(await pdf_file.read())

    response = await orchestrator.predict_multimodal(
        asset_id=asset_id,
        sensor_window=sensor_window,
        image_bytes_list=image_bytes_list or None,
        pdf_bytes_list=pdf_bytes_list or None,
    )
    return response
