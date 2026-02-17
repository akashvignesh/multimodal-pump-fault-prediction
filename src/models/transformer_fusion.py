"""Transformer-based cross-modal fusion for multimodal pump fault prediction.

Architecture
============

    Input Tokens (one per available modality + learnable [CLS] token):

        [CLS]  [sensor]  [text]  [image]  [audio]  [video]  [attachment]
          |       |        |       |        |        |          |
          v       v        v       v        v        v          v
        Modality Projectors  (each: Linear → LayerNorm → GELU → d_model=256)
          |       |        |       |        |        |          |
          v       v        v       v        v        v          v
        + Learnable Modality-Type Embeddings
          |       |        |       |        |        |          |
        ┌──────────────────────────────────────────────────────────┐
        │  Transformer Encoder  (2 layers, 4 heads)                │
        │                                                          │
        │  Multi-Head Self-Attention enables cross-modal reasoning │
        │  • sensor embedding attends to image & text embeddings   │
        │  • image embedding attends to sensor & audio embeddings  │
        │  • each modality context-enriched by all other modes     │
        └────────────────────────┬─────────────────────────────────┘
                                 |
                  [CLS] output (aggregated cross-modal representation)
                                 |
                   ┌─────────────┴─────────────┐
                   v                           v
             Risk Head (MLP→σ)          Confidence Head (MLP→σ)
                   |                           |
             failure_probability          prediction_confidence


Cross-Modal Reasoning Examples
==============================
• "High vibration sensor" + "Image of worn bearing"  → elevated fault score
• "Normal sensor readings" + "Text: pump running smoothly" → low fault score
• "Elevated temperature" + "Image of corrosion" + "Audio: grinding" → very high

The module works in a *hybrid* fashion:

1. **Transformer path** – produces an attention-based risk estimate from the
   raw modality embeddings (cross-modal reasoning).
2. **GatedFusion path** – produces a confidence-weighted average from the
   per-modality scores/confidences (stable baseline).
3. **Blending** – the final prediction is an adaptive mix of both, controlled
   by the number of available modalities and the transformer's own confidence.

Because the Transformer weights are randomly initialised (no multimodal
labelled data to train on yet), the GatedFusion acts as a strong prior and the
Transformer contributes a cross-modal adjustment.  As labelled data becomes
available the blend weight can shift toward the Transformer.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


# =====================================================================
#  Transformer Cross-Modal Fusion  (PyTorch)
# =====================================================================

if HAS_TORCH:

    class _TransformerCrossModalFusion(nn.Module):
        """PyTorch module for cross-modal transformer fusion.

        Parameters
        ----------
        d_model : int
            Hidden dimension of the transformer (default 256).
        nhead : int
            Number of attention heads (default 4).
        num_layers : int
            Depth of the transformer encoder (default 2).
        dropout : float
            Dropout probability (default 0.1).
        """

        # Modality → integer ID (for type embedding look-up)
        MODALITY_IDS: Dict[str, int] = {
            "cls": 0,
            "sensor": 1,
            "text": 2,
            "image": 3,
            "audio": 4,
            "video": 5,
            "attachment": 6,
        }

        # Expected raw embedding dimensions per modality
        DEFAULT_DIMS: Dict[str, int] = {
            "sensor": 260,     # 5 stats x 52 sensors
            "text": 384,       # all-MiniLM-L6-v2
            "image": 512,      # CLIP ViT-B/32  (fallback 1280 for MobileNetV2)
            "audio": 128,      # mel-spectrogram mean
            "video": 512,      # CLIP key-frames
            "attachment": 384,  # text encoder via attachment
        }

        def __init__(
            self,
            d_model: int = 256,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.d_model = d_model

            # ---- Modality projectors ----
            self.projectors = nn.ModuleDict()
            for name, dim in self.DEFAULT_DIMS.items():
                self.projectors[name] = nn.Sequential(
                    nn.Linear(dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                )

            # ---- Learnable modality-type embeddings (7 types) ----
            self.modality_embeddings = nn.Embedding(
                len(self.MODALITY_IDS), d_model,
            )

            # ---- Learnable [CLS] token ----
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

            # ---- Transformer encoder ----
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers,
            )

            # ---- Risk prediction head  (CLS → failure probability) ----
            self.risk_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )

            # ---- Confidence head ----
            self.confidence_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )

            self._init_weights()

        # ----------------------------------------------------------------

        def _init_weights(self) -> None:
            """Xavier-uniform init with neutral output bias."""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            # sigmoid(0) = 0.5  → neutral starting point
            with torch.no_grad():
                # risk_head[-2] is the last Linear before Sigmoid
                self.risk_head[-2].bias.fill_(0.0)
                # confidence_head[-2] is the Linear before Sigmoid
                self.confidence_head[-2].bias.fill_(0.5)

        # ----------------------------------------------------------------

        def forward(
            self,
            modality_embeddings: Dict[str, torch.Tensor],
        ) -> Dict[str, torch.Tensor]:
            """Run cross-modal transformer fusion.

            Parameters
            ----------
            modality_embeddings : dict
                ``{modality_name: tensor}`` — only the *available* modalities.
                Each tensor should be shape ``(dim,)`` or ``(1, dim)``.

            Returns
            -------
            dict with keys ``failure_prob``, ``confidence``, ``cls_embedding``.
            """
            tokens: list = []
            type_ids: list = []

            # Infer batch size from the first available modality embedding
            batch_size = 1
            for _emb in modality_embeddings.values():
                if _emb.dim() >= 2:
                    batch_size = _emb.shape[0]
                    break

            # [CLS] token
            tokens.append(self.cls_token.expand(batch_size, -1, -1))
            type_ids.append(self.MODALITY_IDS["cls"])

            for name, emb in modality_embeddings.items():
                if name not in self.projectors:
                    continue
                # Handle dimension mismatches gracefully
                expected_dim = self.DEFAULT_DIMS[name]
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                if emb.shape[-1] != expected_dim:
                    # Pad or truncate to expected dim
                    if emb.shape[-1] < expected_dim:
                        pad = torch.zeros(
                            *emb.shape[:-1], expected_dim - emb.shape[-1],
                        )
                        emb = torch.cat([emb, pad], dim=-1)
                    else:
                        emb = emb[..., :expected_dim]

                projected = self.projectors[name](emb)          # (B, d_model)
                if projected.dim() == 2:
                    projected = projected.unsqueeze(1)           # (B, 1, d_model)
                tokens.append(projected)
                type_ids.append(self.MODALITY_IDS.get(name, 0))

            # If only CLS (no modalities available) → neutral result
            if len(tokens) == 1:
                return {
                    "failure_prob": torch.tensor(0.5),
                    "confidence": torch.tensor(0.1),
                }

            # Assemble sequence: (B, num_tokens, d_model)
            sequence = torch.cat(tokens, dim=1)

            # Add modality-type embeddings
            ids_tensor = torch.tensor(type_ids, dtype=torch.long)
            type_embs = self.modality_embeddings(ids_tensor).unsqueeze(0)
            sequence = sequence + type_embs

            # Transformer forward (cross-modal self-attention)
            output = self.transformer(sequence)

            # Extract [CLS] output for classification
            cls_out = output[:, 0, :]                            # (B, d_model)

            failure_prob = self.risk_head(cls_out).squeeze()
            confidence = self.confidence_head(cls_out).squeeze()

            return {
                "failure_prob": failure_prob,
                "confidence": confidence,
                "cls_embedding": cls_out.detach(),
            }


# =====================================================================
#  Public wrapper  (works with or without PyTorch)
# =====================================================================

class TransformerFusion:
    """High-level wrapper around ``_TransformerCrossModalFusion``.

    Accepts numpy arrays, converts to tensors, runs the model in eval /
    ``torch.no_grad()`` mode, and returns plain Python floats.

    Falls back gracefully to ``None`` outputs when PyTorch is not available.
    """

    def __init__(self) -> None:
        self._model: Optional[object] = None
        if HAS_TORCH:
            try:
                self._model = _TransformerCrossModalFusion()
                self._load_trained_weights()
                self._model.eval()
                logger.info(
                    "TransformerCrossModalFusion initialised "
                    "(d_model=256, 2 layers, 4 heads)"
                )
            except Exception as exc:
                logger.warning("Transformer fusion init failed: %s", exc)

    def _load_trained_weights(self) -> None:
        """Load trained fusion weights from artifacts if available."""
        from pathlib import Path

        weights_path = (
            Path(__file__).resolve().parent.parent.parent
            / "artifacts"
            / "transformer_fusion_trained.pt"
        )
        if weights_path.exists():
            try:
                checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
                self._model.load_state_dict(checkpoint["state_dict"])
                metrics = checkpoint.get("metrics", {})
                logger.info(
                    "Loaded trained transformer fusion weights from %s  "
                    "(AUC=%.4f)",
                    weights_path,
                    metrics.get("roc_auc", 0.0),
                )
            except Exception as exc:
                logger.warning(
                    "Could not load trained fusion weights (%s) – "
                    "using random init",
                    exc,
                )
        else:
            logger.info(
                "No trained fusion weights found at %s – using random init. "
                "Run: python scripts/train_joint_multimodal.py",
                weights_path,
            )

    @property
    def available(self) -> bool:
        return self._model is not None

    def fuse_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
    ) -> Optional[Tuple[float, float]]:
        """Run transformer fusion on modality embeddings.

        Parameters
        ----------
        embeddings : dict
            ``{modality_name: numpy_array}`` for each available modality.

        Returns
        -------
        ``(failure_probability, confidence)`` or ``None`` if unavailable.
        """
        if self._model is None or not embeddings:
            return None

        try:
            tensor_embs = {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in embeddings.items()
            }
            with torch.no_grad():
                out = self._model(tensor_embs)

            fp = float(out["failure_prob"].item())
            conf = float(out["confidence"].item())
            return fp, conf
        except Exception as exc:
            logger.warning("Transformer fusion forward failed: %s", exc)
            return None
