"""CLIP-based Vision-Language Model (VLM) encoder for pump fault image analysis.

Uses CLIP ViT-B/32 for semantic understanding of images:

    Image → CLIP ViT-B/32 → 512-dim embedding
                           ↘ cosine similarity with fault / normal text prompts
                           → fault signals + risk score

Key advantages:
- **Semantic understanding**: Recognises corrosion, leaks, wear from image content,
  not just pixel colour thresholds.
- **Zero-shot classification**: No labelled pump images required — CLIP compares
  the image embedding against pre-defined fault-description prompts.
- **Rich embeddings**: 512-dim vision-language aligned embeddings feed directly
  into the Transformer cross-modal fusion.
"""
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    PILImage = None

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

try:
    from transformers import CLIPModel, CLIPProcessor
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    CLIPModel = None
    CLIPProcessor = None

from src.config import settings


# ---------------------------------------------------------------------------
# VLM fault prompt catalogue
# ---------------------------------------------------------------------------

# Each tuple: (text_prompt, signal_name, weight)
FAULT_PROMPTS: List[Tuple[str, str, float]] = [
    ("a photo of corroded metal on industrial pump equipment", "corrosion_indicator", 0.40),
    ("a photo of rust and oxidation on a pipe or valve", "corrosion_indicator", 0.35),
    ("a photo of fluid leaking from a pump or pipe joint", "leak_detected", 0.40),
    ("a photo of oil stains and puddles under machinery", "leak_detected", 0.35),
    ("a photo of a worn damaged mechanical bearing", "wear_detected", 0.40),
    ("a photo of a cracked or broken pump impeller", "imminent_failure", 0.50),
    ("a photo of smoke coming from overheating industrial equipment", "thermal_stress", 0.50),
    ("a photo of misaligned pump shaft coupling", "alignment_deviation", 0.35),
    ("a photo of a clogged dirty filter on a pump", "filter_clogged", 0.35),
]

NORMAL_PROMPTS: List[str] = [
    "a photo of a clean well-maintained industrial pump",
    "a photo of new shiny metal pump equipment in good condition",
    "a photo of a properly functioning mechanical system",
]

# Similarity above this threshold (and higher than average normal-similarity)
# triggers the corresponding fault signal.
FAULT_THRESHOLD = 0.22


class CLIPImageEncoder:
    """VLM image encoder using CLIP ViT-B/32.

    Public API (backward-compatible with the old ``ImageEncoder``):
        encode(image_refs)          – file-path references
        encode_pil_images(images)   – list of PIL.Image objects

    Both return ``ModalityOutput`` (imported lazily to avoid circular deps).
    """

    def __init__(self) -> None:
        self.clip_model: Optional[Any] = None
        self.clip_processor: Optional[Any] = None

        self.embedding_cache: Dict[str, np.ndarray] = {}
        self._fault_text_features: Optional[Any] = None
        self._normal_text_features: Optional[Any] = None
        self._use_clip: bool = False
        self.embedding_dim: int = 512

        self._load_model()

    # ------------------------------------------------------------------
    #  Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load CLIP ViT-B/32 model."""
        if HAS_CLIP and HAS_TORCH:
            try:
                self.clip_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                self.clip_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                self.clip_model.eval()
                self._precompute_text_features()
                self._use_clip = True
                self.embedding_dim = 512
                logger.info(
                    "Loaded CLIP VLM image encoder "
                    "(512-dim, zero-shot fault classification)"
                )
            except Exception as exc:
                logger.warning("CLIP load failed: %s", exc)
        else:
            logger.warning(
                "CLIP image encoder not available "
                "(install transformers and torch)"
            )

    # ------------------------------------------------------------------
    #  CLIP text pre-computation
    # ------------------------------------------------------------------

    def _precompute_text_features(self) -> None:
        """Pre-encode fault/normal text prompts for zero-shot classification."""
        if self.clip_model is None or self.clip_processor is None:
            return
        try:
            fault_texts = [p[0] for p in FAULT_PROMPTS]
            all_texts = fault_texts + NORMAL_PROMPTS

            inputs = self.clip_processor(
                text=all_texts, return_tensors="pt", padding=True, truncation=True,
            )
            # Only pass text-related keys to get_text_features
            text_inputs = {
                k: v for k, v in inputs.items()
                if k in ("input_ids", "attention_mask")
            }
            with torch.no_grad():
                raw = self.clip_model.get_text_features(**text_inputs)
                # transformers >=5.x returns BaseModelOutputWithPooling
                feats = raw.pooler_output if hasattr(raw, "pooler_output") else raw
                feats = feats / feats.norm(dim=-1, keepdim=True)

            n_fault = len(fault_texts)
            self._fault_text_features = feats[:n_fault]
            self._normal_text_features = feats[n_fault:]
            logger.info(
                "Pre-computed %d fault + %d normal text embeddings for VLM",
                n_fault, len(NORMAL_PROMPTS),
            )
        except Exception as exc:
            logger.warning("Text-feature pre-computation failed: %s", exc)

    # ------------------------------------------------------------------
    #  Single-image analysis
    # ------------------------------------------------------------------

    def _clip_analyze(self, pil_image) -> Tuple[np.ndarray, List[str], float]:
        """Run CLIP VLM analysis on one PIL image.

        Returns (embedding_512, fault_signals, fault_score).
        """
        inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        # Only pass image-related keys
        image_inputs = {
            k: v for k, v in inputs.items()
            if k in ("pixel_values",)
        }
        with torch.no_grad():
            raw = self.clip_model.get_image_features(**image_inputs)
            # transformers >=5.x returns BaseModelOutputWithPooling
            img_feat = raw.pooler_output if hasattr(raw, "pooler_output") else raw
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        embedding = img_feat.cpu().numpy().flatten()

        signals: List[str] = []
        fault_score = 0.0

        if self._fault_text_features is not None:
            fault_sims = (
                (img_feat @ self._fault_text_features.T).squeeze().cpu().numpy()
            )
            normal_sims = (
                (img_feat @ self._normal_text_features.T).squeeze().cpu().numpy()
            )
            avg_normal = float(np.mean(normal_sims))

            for i, (_, signal, weight) in enumerate(FAULT_PROMPTS):
                sim = float(fault_sims[i])
                if sim > FAULT_THRESHOLD and sim > avg_normal + 0.02:
                    if signal not in signals:
                        signals.append(signal)
                    fault_score += weight * max(sim - avg_normal, 0)

            fault_score = float(np.clip(fault_score, 0.0, 1.0))

        return embedding, signals, fault_score

    # ------------------------------------------------------------------
    #  Public encode methods
    # ------------------------------------------------------------------

    def encode_pil_images(self, pil_images: list):
        """Encode a list of PIL images using CLIP.

        Returns ``ModalityOutput``.
        """
        from src.models.multimodal import ModalityOutput

        if not pil_images:
            return ModalityOutput(0.0, 0.0, [], None)

        if not self._use_clip:
            return ModalityOutput(0.0, 0.1, ["clip_unavailable"], None)

        embeddings: List[np.ndarray] = []
        all_signals: List[str] = []
        total_score = 0.0
        valid = 0

        for img in pil_images:
            try:
                emb, sigs, score = self._clip_analyze(img)
                if emb is not None:
                    embeddings.append(emb)
                for s in sigs:
                    if s not in all_signals:
                        all_signals.append(s)
                total_score += score
                valid += 1
            except Exception as exc:
                logger.warning("Image encoding failed: %s", exc)

        if valid == 0:
            return ModalityOutput(0.0, 0.0, ["image_processing_failed"], None)

        avg_emb = np.mean(embeddings, axis=0) if embeddings else None
        avg_score = min(total_score / valid, 1.0)
        confidence = min(0.6 + valid / max(len(pil_images), 1) * 0.35, 0.95)

        return ModalityOutput(avg_score, confidence, all_signals[:5], avg_emb)

    def _get_cache_key(self, path: str) -> str:
        return hashlib.md5(path.encode()).hexdigest()

    def encode(self, image_refs: List[str]):
        """Encode image file-path references.

        Returns ``ModalityOutput``.
        """
        from src.models.multimodal import ModalityOutput

        if not image_refs:
            return ModalityOutput(0.0, 0.0, [], None)

        pil_images: list = []
        for ref in image_refs:
            try:
                image_path = settings.data_dir / ref
                if not image_path.exists():
                    image_path = settings.multimodal_data_dir / ref
                if not image_path.exists():
                    logger.warning("Image not found: %s", ref)
                    continue
                if HAS_PIL:
                    pil_images.append(PILImage.open(image_path).convert("RGB"))
            except Exception as exc:
                logger.warning("Failed to load image %s: %s", ref, exc)

        return self.encode_pil_images(pil_images)
