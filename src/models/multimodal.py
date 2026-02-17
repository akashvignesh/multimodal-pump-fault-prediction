"""Multimodal encoders for pump fault risk prediction.

Includes encoders for:
- Image (CLIP ViT-B/32 via CLIPImageEncoder)
"""
import logging
from typing import Dict, List, Optional, Any

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class ModalityOutput:
    """Standard output from a modality encoder."""
    
    def __init__(
        self,
        score: float,
        confidence: float,
        signals: List[str],
        embedding: Optional[np.ndarray] = None
    ):
        self.score = float(np.clip(score, 0.0, 1.0))
        self.confidence = float(np.clip(confidence, 0.0, 1.0))
        self.signals = signals[:5] if signals else []
        self.embedding = embedding
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "confidence": self.confidence,
            "signals": self.signals
        }


class ImageEncoder:
    """VLM-based image encoder using CLIP for semantic pump fault analysis.

    Primary: CLIP ViT-B/32 (Vision-Language Model)
      - 512-dim semantically rich embeddings
      - Zero-shot fault classification via image-text similarity
      - Understands corrosion, leaks, wear without hardcoded pixel rules

    Delegates to ``src.models.clip_encoder.CLIPImageEncoder``.
    """

    def __init__(self):
        from src.models.clip_encoder import CLIPImageEncoder
        self._encoder = CLIPImageEncoder()
        self.model = self._encoder.clip_model
        self.transform = None
        self.embedding_cache = self._encoder.embedding_cache
        self.embedding_dim = self._encoder.embedding_dim

    def encode(self, image_refs: List[str]) -> ModalityOutput:
        """Encode image file-path references using VLM."""
        return self._encoder.encode(image_refs)

    def encode_pil_images(self, pil_images: list) -> ModalityOutput:
        """Encode PIL images directly (used for uploaded bytes)."""
        return self._encoder.encode_pil_images(pil_images)


class MultimodalEncoderManager:
    """Manager for modality encoders with lazy initialization.
    
    Currently supports image encoding only (sensor handled separately).
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.image_encoder = ImageEncoder()
        
        self._initialized = True
        logger.info("MultimodalEncoderManager initialized (image encoder)")
    
    def encode_images(self, image_refs: List[str]) -> ModalityOutput:
        return self.image_encoder.encode(image_refs)
