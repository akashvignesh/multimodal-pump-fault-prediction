"""Fusion module for combining multimodal predictions.

Two fusion strategies — used in a **hybrid** configuration:

1. **TransformerCrossModalFusion** (primary, when PyTorch available)
   - Projects sensor + image embeddings to a common 256-dim space
   - Multi-head self-attention enables cross-modal reasoning
   - [CLS] token aggregates information across modalities
   - Produces attention-based risk and confidence estimates

2. **GatedFusion** (baseline / fallback)
   - Confidence-weighted average of per-modality scores
   - No cross-modal interaction (each modality independent)
   - Always available (no PyTorch dependency)

The ``FusionModule`` blends both strategies: the transformer contributes
cross-modal reasoning while the gated fusion provides a stable baseline.
Supports **any subset** of inputs — sensor data is no longer required.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models.multimodal import ModalityOutput
from src.models.transformer_fusion import TransformerFusion

logger = logging.getLogger(__name__)


class GatedFusion:
    """Gated attention fusion for combining multimodal predictions.
    
    Uses confidence-weighted averaging with gating mechanism.
    Supports graceful degradation when modalities are missing.
    """
    
    # Base importance weights for each modality
    DEFAULT_WEIGHTS = {
        "sensor": 0.6,
        "image": 0.4,
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.temperature = 1.0

    def fuse(
        self,
        sensor_output: Optional[Tuple[float, float, List[str]]] = None,
        image_output: Optional[ModalityOutput] = None,
    ) -> Tuple[float, float, List[str]]:
        """Fuse multimodal outputs into a final prediction.

        *All* arguments are optional. At least one must carry signal;
        otherwise a low-confidence unknown result is returned.
        """
        modality_scores: list = []
        modality_confidences: list = []
        modality_weights: list = []
        all_signals: List[Tuple[str, float]] = []

        # Sensor
        if sensor_output is not None:
            prob, conf, sigs = sensor_output
            modality_scores.append(prob)
            modality_confidences.append(conf)
            modality_weights.append(self.weights["sensor"])
            all_signals.extend([(s, conf) for s in sigs])

        # Image
        if image_output is not None and image_output.confidence > 0:
            modality_scores.append(image_output.score)
            modality_confidences.append(image_output.confidence)
            modality_weights.append(self.weights["image"])
            all_signals.extend([(s, image_output.confidence) for s in image_output.signals])

        # Nothing available -> uncertain fallback
        if not modality_scores:
            return 0.5, 0.1, ["insufficient_data"]

        scores = np.array(modality_scores)
        confidences = np.array(modality_confidences)
        weights = np.array(modality_weights)

        # Re-normalise weights so they sum to 1 among *available* modalities
        gate_logits = weights * confidences / self.temperature
        gates = self._softmax(gate_logits)

        failure_probability = float(np.clip(np.sum(gates * scores), 0.0, 1.0))
        fault_confidence = self._compute_confidence(
            scores, confidences, weights, gates, failure_probability
        )
        top_signals = self._aggregate_signals(all_signals)

        logger.debug(
            f"Fusion: prob={failure_probability:.3f}, "
            f"conf={fault_confidence:.3f}, signals={top_signals}"
        )
        return failure_probability, fault_confidence, top_signals
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _compute_confidence(
        self,
        scores: np.ndarray,
        confidences: np.ndarray,
        weights: np.ndarray,
        gates: np.ndarray,
        final_prob: float
    ) -> float:
        """Compute overall prediction confidence."""
        # Weighted average confidence
        avg_conf = float(np.sum(gates * confidences))
        
        # Agreement penalty (high variance = lower confidence)
        if len(scores) > 1:
            variance = np.var(scores)
            agreement_factor = 1.0 / (1.0 + variance)
        else:
            agreement_factor = 0.8
        
        # Certainty bonus (predictions near 0 or 1 are more confident)
        certainty_factor = abs(2 * final_prob - 1)
        
        confidence = (0.4 * avg_conf + 
                      0.3 * agreement_factor + 
                      0.3 * certainty_factor)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _aggregate_signals(
        self,
        signal_confidence_pairs: List[Tuple[str, float]],
        top_k: int = 5
    ) -> List[str]:
        """Aggregate signals from all modalities using confidence-weighted voting."""
        if not signal_confidence_pairs:
            return ["normal_operation"]
        
        signal_scores: Dict[str, float] = {}
        for signal, conf in signal_confidence_pairs:
            if signal in signal_scores:
                signal_scores[signal] += conf
            else:
                signal_scores[signal] = conf
        
        sorted_signals = sorted(
            signal_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_signals = [s for s, _ in sorted_signals[:top_k]]
        return top_signals if top_signals else ["unknown_signal"]


class FusionModule:
    """Hybrid fusion: Transformer cross-modal + GatedFusion baseline.

    Strategy:
    1. GatedFusion always runs (stable baseline from modality scores).
    2. If TransformerFusion is available AND embeddings are supplied,
       the transformer produces a cross-modal prediction.
    3. The final result blends both via an adaptive weight.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.gated = GatedFusion()
            cls._instance.transformer = TransformerFusion()
            if cls._instance.transformer.available:
                logger.info(
                    "FusionModule: Hybrid mode "
                    "(TransformerCrossModal + GatedFusion)"
                )
            else:
                logger.info("FusionModule: GatedFusion only (no PyTorch)")
        return cls._instance

    def fuse(
        self,
        sensor_output: Optional[Tuple[float, float, List[str]]] = None,
        image_output: Optional[ModalityOutput] = None,
        *,
        sensor_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, List[str]]:
        """Fuse multimodal outputs.

        Parameters
        ----------
        sensor_output :
            (failure_prob, confidence, signals) from sensor model.
        image_output :
            ModalityOutput from image encoder.
        sensor_embedding :
            Raw 260-dim sensor feature vector for the transformer path.
        """
        # --- 1. Gated fusion (always) ---
        gated_prob, gated_conf, top_signals = self.gated.fuse(
            sensor_output=sensor_output,
            image_output=image_output,
        )

        # --- 2. Transformer fusion (if available + embeddings) ---
        if not self.transformer.available:
            return gated_prob, gated_conf, top_signals

        embeddings: Dict[str, np.ndarray] = {}
        if sensor_embedding is not None:
            embeddings["sensor"] = sensor_embedding
        if image_output is not None and image_output.embedding is not None:
            embeddings["image"] = image_output.embedding

        if len(embeddings) < 1:
            return gated_prob, gated_conf, top_signals

        transformer_result = self.transformer.fuse_embeddings(embeddings)
        if transformer_result is None:
            return gated_prob, gated_conf, top_signals

        trans_prob, trans_conf = transformer_result

        # --- 3. Adaptive blend ---
        n_modalities = len(embeddings)
        # 1 modality → 0.20, 2 modalities → 0.35
        transformer_weight = min(0.10 + n_modalities * 0.125, 0.40)

        blended_prob = float(
            (1 - transformer_weight) * gated_prob
            + transformer_weight * trans_prob
        )
        blended_conf = float(
            (1 - transformer_weight) * gated_conf
            + transformer_weight * trans_conf
        )

        blended_prob = float(np.clip(blended_prob, 0.0, 1.0))
        blended_conf = float(np.clip(blended_conf, 0.0, 1.0))

        logger.debug(
            "Hybrid fusion: gated=%.3f, transformer=%.3f, "
            "blend=%.3f (weight=%.2f, %d modalities)",
            gated_prob, trans_prob, blended_prob,
            transformer_weight, n_modalities,
        )

        return blended_prob, blended_conf, top_signals
