"""Inference orchestrator for pump fault risk prediction.

Coordinates:
- Request validation
- Model loading (once at startup)
- Parallel modality encoding (sensor + image)
- Fusion (hybrid: transformer cross-modal + gated baseline)
- Response generation

Supports **two modes**:
1. Baseline – sensor-only (uses SensorBaselineModel)
2. Multimodal – sensor + image (with optional PDF image extraction)
   Sensor data is OPTIONAL in multimodal mode.
"""
import asyncio
import hashlib
import io
import json
import logging
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


def _generate_explanation(fp: float, fc: float, signals: List[str]) -> str:
    """Generate a brief prose explanation from prediction outputs."""
    if fp >= 0.8:
        risk = "Very high"
    elif fp >= 0.6:
        risk = "Elevated"
    elif fp >= 0.4:
        risk = "Moderate"
    elif fp >= 0.2:
        risk = "Low"
    else:
        risk = "Minimal"

    if fc >= 0.7:
        conf_q = "high confidence"
    elif fc >= 0.4:
        conf_q = "moderate confidence"
    else:
        conf_q = "low confidence"

    clean_signals = [s.replace("_", " ") for s in signals[:3]]
    if len(clean_signals) == 0:
        signal_phrase = "no sensor or image data provided"
    elif "no input data" in " ".join(signals):
        signal_phrase = "no sensor or image data provided"
    elif len(clean_signals) == 1:
        signal_phrase = clean_signals[0]
    else:
        signal_phrase = ", ".join(clean_signals[:-1]) + f" and {clean_signals[-1]}"

    return (
        f"{risk} failure risk ({fp:.0%}) with {conf_q} ({fc:.0%}). "
        f"Top contributing factors: {signal_phrase}."
    )


from src.models.risk_model import SensorBaselineModel, JointSensorImageModel
from src.models.multimodal import (
    MultimodalEncoderManager,
    ModalityOutput,
)
from src.models.fusion import FusionModule
from src.services.preprocessing import compute_sensor_anomalies
from src.api.schemas.request import PredictionRequest
from src.api.schemas.response import PredictionResponse

_RESPONSE_CACHE_MAX = 1024

# Optional imports for PDF image extraction
try:
    import fitz as pymupdf  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def extract_pdf_images(pdf_bytes: bytes, max_images: int = 5) -> list:
    """Extract PIL images embedded in a PDF. Returns list of PIL.Image."""
    images: list = []
    if not HAS_PYMUPDF:
        logger.warning(
            "PyMuPDF not installed - cannot extract images from PDFs. "
            "Install with: pip install PyMuPDF>=1.23.0"
        )
        return images
    if not HAS_PIL:
        logger.warning("PIL not installed - cannot process images")
        return images
    
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        logger.info(f"Processing PDF with {page_count} pages")
        
        for page_num, page in enumerate(doc, 1):
            if len(images) >= max_images:
                break
            page_images = page.get_images(full=True)
            logger.debug(f"Page {page_num}: found {len(page_images)} images")
            
            for img_info in page_images:
                if len(images) >= max_images:
                    break
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                if base_image:
                    img_data = base_image["image"]
                    try:
                        pil_img = PILImage.open(io.BytesIO(img_data)).convert("RGB")
                        images.append(pil_img)
                        logger.debug(f"Successfully extracted image {len(images)}")
                    except Exception as e:
                        logger.debug(f"Failed to decode image: {e}")
        doc.close()
        
        if images:
            logger.info(f"Extracted {len(images)} images from PDF")
        else:
            logger.warning("No images found in PDF document")
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}", exc_info=True)
    
    return images


class InferenceOrchestrator:
    """Orchestrates multimodal inference pipeline.

    Singleton pattern ensures models are loaded once at startup.
    """

    _instance: Optional['InferenceOrchestrator'] = None
    _initialized: bool = False

    def __new__(cls) -> 'InferenceOrchestrator':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info("Initializing InferenceOrchestrator...")
        
        # Set random seeds for reproducibility
        import random
        import numpy as np
        random.seed(42)
        np.random.seed(42)
        
        # Also set for torch if available
        try:
            import torch
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

        # Load models
        self.sensor_model = SensorBaselineModel()
        self.joint_model = JointSensorImageModel()
        self.encoder_manager = MultimodalEncoderManager()
        self.fusion = FusionModule()

        if self.joint_model.available:
            logger.info("Joint sensor+image model loaded – will use when both modalities present")

        # Thread pool for parallel encoding (sized for concurrency)
        import os
        self.executor = ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) + 4))

        # Response cache for sensor-only requests
        self._resp_cache: OrderedDict[str, PredictionResponse] = OrderedDict()

        self._initialized = True
        logger.info("InferenceOrchestrator initialized successfully")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_sensor_only(request: PredictionRequest) -> bool:
        return not request.image_refs

    @staticmethod
    def _request_cache_key(request: PredictionRequest) -> str:
        raw = json.dumps(
            {"a": request.asset_id, "s": request.sensor_window},
            sort_keys=True, default=str,
        )
        return hashlib.md5(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Standard predict (PredictionRequest)
    # ------------------------------------------------------------------

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Run full multimodal inference pipeline.

        sensor_window may be empty — in that case only image is used.
        """
        start_time = time.perf_counter()

        # --- cache for sensor-only ---
        sensor_only = self._is_sensor_only(request)
        cache_key: Optional[str] = None
        if sensor_only and request.sensor_window:
            cache_key = self._request_cache_key(request)
            cached = self._resp_cache.get(cache_key)
            if cached is not None:
                ms = int((time.perf_counter() - start_time) * 1000)
                return PredictionResponse(
                    asset_id=cached.asset_id,
                    failure_probability=cached.failure_probability,
                    fault_confidence=cached.fault_confidence,
                    top_signals=cached.top_signals,
                    inference_ms=ms,
                    model_version=cached.model_version,
                )

        try:
            loop = asyncio.get_event_loop()

            # 1. Sensor model – OPTIONAL
            sensor_output: Optional[Tuple[float, float, List[str]]] = None
            sensor_embedding: Optional[np.ndarray] = None
            if request.sensor_window:
                sensor_output = await loop.run_in_executor(
                    self.executor, self.sensor_model.predict, request.sensor_window
                )
                sensor_embedding = self.sensor_model.extract_features(request.sensor_window)

            # 2. Image encoder
            image_output: Optional[ModalityOutput] = None
            if request.image_refs:
                image_output = await loop.run_in_executor(
                    self.executor, self.encoder_manager.encode_images, request.image_refs
                )

            # 3. Joint model upgrade: when sensor + image are both available
            if (
                self.joint_model.available
                and sensor_embedding is not None
                and image_output is not None
                and image_output.embedding is not None
            ):
                joint_result = self.joint_model.predict(
                    sensor_embedding, image_output.embedding,
                )
                if joint_result is not None:
                    logger.debug(
                        "Using joint sensor+image model (prob=%.3f)",
                        joint_result[0],
                    )
                    sensor_output = joint_result

            # 4. Fuse (hybrid: transformer cross-modal + gated baseline)
            fp, fc, ts = self.fusion.fuse(
                sensor_output=sensor_output,
                image_output=image_output,
                sensor_embedding=sensor_embedding,
            )

            # 5. Enrich top_signals with anomaly detections
            if request.sensor_window:
                anomaly_signals = compute_sensor_anomalies(request.sensor_window) or []
                # Deduplicate while preserving order
                seen = set(ts)
                for sig in anomaly_signals:
                    if sig not in seen:
                        ts.append(sig)
                        seen.add(sig)

            inference_ms = int((time.perf_counter() - start_time) * 1000)

            explanation = _generate_explanation(fp, fc, ts)

            response = PredictionResponse(
                asset_id=request.asset_id,
                failure_probability=round(fp, 4),
                fault_confidence=round(fc, 4),
                top_signals=ts[:5],
                explanation=explanation,
                inference_ms=inference_ms,
                model_version=settings.model_version,
            )

            if sensor_only and cache_key is not None:
                if len(self._resp_cache) >= _RESPONSE_CACHE_MAX:
                    for _ in range(_RESPONSE_CACHE_MAX // 4):
                        self._resp_cache.popitem(last=False)
                self._resp_cache[cache_key] = response

            return response

        except Exception as e:
            logger.error(f"Prediction failed for {request.asset_id}: {e}", exc_info=True)
            ms = int((time.perf_counter() - start_time) * 1000)
            return PredictionResponse(
                asset_id=request.asset_id,
                failure_probability=0.5,
                fault_confidence=0.1,
                top_signals=["prediction_error"],
                explanation="Prediction failed — returning neutral estimate. Check input data.",
                inference_ms=ms,
                model_version=settings.model_version,
            )

    # ------------------------------------------------------------------
    # Multimodal predict (raw bytes — Streamlit file uploads)
    # ------------------------------------------------------------------

    async def predict_multimodal(
        self,
        *,
        asset_id: str = "upload",
        sensor_window: Optional[List[Dict[str, Any]]] = None,
        image_bytes_list: Optional[List[bytes]] = None,
        pdf_bytes_list: Optional[List[bytes]] = None,
    ) -> PredictionResponse:
        """Predict from *raw bytes* — sensor + images + PDFs.

        This is the entry-point used by the Streamlit UI when the user
        uploads files directly (instead of passing file-path refs).
        """
        start_time = time.perf_counter()

        loop = asyncio.get_event_loop()

        sensor_output: Optional[Tuple[float, float, List[str]]] = None
        image_output: Optional[ModalityOutput] = None

        all_images: list = []  # PIL Images from PDFs

        # ---- PDF processing (extract images) ----
        if pdf_bytes_list:
            logger.info(f"Processing {len(pdf_bytes_list)} PDF document(s)")
        for pdf_b in (pdf_bytes_list or []):
            pdf_imgs = extract_pdf_images(pdf_b)
            all_images.extend(pdf_imgs)
        if pdf_bytes_list and not all_images:
            logger.warning(
                "PDF(s) provided but no images extracted. "
                "Ensure PyMuPDF is installed and PDFs contain images."
            )

        # ---- Sensor ----
        sensor_embedding = None
        if sensor_window:
            sensor_output = await loop.run_in_executor(
                self.executor, self.sensor_model.predict, sensor_window
            )
            sensor_embedding = self.sensor_model.extract_features(sensor_window)

        # ---- Images (from bytes + PDF-extracted PIL images) ----
        all_image_bytes: List[bytes] = list(image_bytes_list or [])
        for pil_img in all_images:
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            all_image_bytes.append(buf.getvalue())

        if all_image_bytes:
            image_output = await loop.run_in_executor(
                self.executor,
                self._encode_image_bytes,
                all_image_bytes,
            )

        # Joint model upgrade: when sensor + image are both available
        if (
            self.joint_model.available
            and sensor_embedding is not None
            and image_output is not None
            and image_output.embedding is not None
        ):
            joint_result = self.joint_model.predict(
                sensor_embedding, image_output.embedding,
            )
            if joint_result is not None:
                logger.debug(
                    "predict_multimodal: using joint model (prob=%.3f)",
                    joint_result[0],
                )
                sensor_output = joint_result

        # Fuse (hybrid: transformer cross-modal + gated baseline)
        fp, fc, ts = self.fusion.fuse(
            sensor_output=sensor_output,
            image_output=image_output,
            sensor_embedding=sensor_embedding,
        )

        # Enrich top_signals with anomaly detections
        if sensor_window:
            anomaly_signals = compute_sensor_anomalies(sensor_window) or []
            seen = set(ts)
            for sig in anomaly_signals:
                if sig not in seen:
                    ts.append(sig)
                    seen.add(sig)

        inference_ms = int((time.perf_counter() - start_time) * 1000)

        explanation = _generate_explanation(fp, fc, ts)

        return PredictionResponse(
            asset_id=asset_id,
            failure_probability=round(fp, 4),
            fault_confidence=round(fc, 4),
            top_signals=ts[:5],
            explanation=explanation,
            inference_ms=inference_ms,
            model_version=settings.model_version,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_image_bytes(self, image_bytes_list: List[bytes]) -> ModalityOutput:
        """Encode images from raw bytes using CLIP."""
        enc = self.encoder_manager.image_encoder

        pil_images: list = []
        for img_bytes in image_bytes_list:
            try:
                pil_images.append(PILImage.open(io.BytesIO(img_bytes)).convert("RGB"))
            except Exception as e:
                logger.warning(f"Image decoding from bytes failed: {e}")

        return enc.encode_pil_images(pil_images)

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    async def predict_batch(
        self, requests: List[PredictionRequest]
    ) -> List[PredictionResponse]:
        tasks = [self.predict(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                logger.error(f"Batch prediction {i} failed: {resp}")
                results.append(PredictionResponse(
                    asset_id=requests[i].asset_id,
                    failure_probability=0.5,
                    fault_confidence=0.1,
                    top_signals=["batch_error"],
                    inference_ms=0,
                    model_version=settings.model_version,
                ))
            else:
                results.append(resp)
        return results


# Global orchestrator instance
_orchestrator: Optional[InferenceOrchestrator] = None


def get_orchestrator() -> InferenceOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = InferenceOrchestrator()
    return _orchestrator


async def initialize_orchestrator() -> None:
    global _orchestrator
    _orchestrator = InferenceOrchestrator()
    logger.info("Orchestrator initialized at startup")
