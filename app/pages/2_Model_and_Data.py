"""
🏗️ Model & Data — Architecture, pipeline, dataset, and schemas.

Covers:
- End-to-end pipeline overview
- Transformer-centric architecture explanation
- Missing-modality handling
- Modality extensibility blueprint
- Dataset summary
- Request / response schema
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path

from _shared import (
    inject_css, render_sidebar, demo_script,
    PROJECT_ROOT, load_sample_data, load_ablation_results,
)

st.set_page_config(page_title="Model & Data", page_icon="🏗️", layout="wide")
inject_css()
render_sidebar()

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1565C0 0%, #42A5F5 100%);
                color: white; padding: 1.2rem 2rem; border-radius: 0.8rem;
                margin-bottom: 1rem;">
        <h2 style="margin:0;">🏗️ Model & Data</h2>
        <p style="margin:0.3rem 0 0 0; opacity:0.9;">
            Pipeline architecture, dataset summary, and how predictions are produced.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

demo_script(
    "Let me show you what's under the hood. Our system uses a hybrid architecture — "
    "LightGBM for tabular sensor data, CLIP for images, and a trained Transformer "
    "fusion layer that learns cross-attention between modalities."
)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">End-to-End Pipeline</div>', unsafe_allow_html=True)

st.markdown(
    """
    ```
    ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
    │   INPUT DATA   │────▶│   ML MODELS    │────▶│    OUTPUT      │
    │                │     │                │     │                │
    │ • 52 sensors   │     │ • Sensor model │     │ • Risk score   │
    │ • Pump images  │     │ • Image model  │     │ • Confidence   │
    │ • PDF reports  │     │ • Fusion layer │     │ • Top signals  │
    └────────────────┘     └────────────────┘     │ • Explanation  │
                                                   └────────────────┘
    ```

    **In plain terms:** Sensor readings and images go in, a risk score with
    explainable signals comes out. The system works with any combination of
    inputs — sensor only, image only, or both together.
    """
)

st.markdown("##### Detailed Architecture")

st.markdown(
    """
    ```
    ┌─ Preprocessing ──────────────────────────────────────────────────────────────┐
        │                                                                               │
        │  sensor_window[]  ──▶  extract_features()  ──▶  [260-dim float vector]       │
        │                        (mean, std, min,                                       │
        │                         max, range × 52)                                      │
        │                                                                               │
        │  image bytes      ──▶  CLIP ViT-B/32       ──▶  [512-dim float vector]       │
        │                        (frozen encoder)                                       │
        │                                                                               │
        │  PDF bytes        ──▶  PyMuPDF extract      ──▶  embedded images ──▶ CLIP     │
        │                        (images from pages)                                    │
        └──────────────────────────────────────────────────────────────────────────────┘
                                         │
        ┌─ Models ─────────────────────────────────────────────────────────────────────┐
        │                                                                               │
        │  SensorBaselineModel (LightGBM)     input: 260-dim  ──▶ prob + confidence    │
        │                                                                               │
        │  CLIPImageEncoder                   input: image     ──▶ fault_conf + label   │
        │    └─ 9 fault prompts + 3 normal prompts, cosine similarity, threshold 0.22  │
        │                                                                               │
        │  JointSensorImageModel (LightGBM)   input: 772-dim  ──▶ prob + confidence    │
        │    └─ [260 sensor features ‖ 512 CLIP embeddings]                             │
        │                                                                               │
        └──────────────────────────────────────────────────────────────────────────────┘
                                         │
        ┌─ Fusion ─────────────────────────────────────────────────────────────────────┐
        │                                                                               │
        │  TransformerCrossModalFusion (trained, d=256, 2 layers, 4 heads)             │
        │    ├─ sensor_proj: Linear(260 → 256)                                         │
        │    ├─ image_proj:  Linear(512 → 256)                                         │
        │    ├─ [CLS] token  (learnable)                                               │
        │    ├─ TransformerEncoder (2 layers, 4 heads, 512 FFN dim)                    │
        │    ├─ risk_head:   Linear(256 → 1) + sigmoid                                │
        │    └─ conf_head:   Linear(256 → 1) + sigmoid                                │
        │                                                                               │
        │  GatedFusion (confidence-weighted softmax over modality outputs)              │
        │                                                                               │
        │  Adaptive Blend:                                                              │
        │    w = min(0.10 + n_modalities × 0.125, 0.40)                                │
        │    final = (1 - w) × gated + w × transformer                                 │
        │                                                                               │
        └──────────────────────────────────────────────────────────────────────────────┘
                                         │
        ┌─ Post-processing ────────────────────────────────────────────────────────────┐
        │                                                                               │
        │  compute_sensor_anomalies()  ──▶  z-score spikes, dot-product trends,        │
        │                                   variance alerts  →  top_signals[]          │
        │                                                                               │
        │  _generate_explanation()     ──▶  prose summary of risk + signals            │
        │                                                                               │
        └──────────────────────────────────────────────────────────────────────────────┘
        ```
        """
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER FLOW
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Transformer-Centric Architecture</div>', unsafe_allow_html=True)

st.markdown(
    """
    The **TransformerCrossModalFusion** is the core fusion mechanism. It is a
    genuine PyTorch `nn.Module` that was **trained end-to-end** on 241 paired
    sensor + image samples.

    #### Step-by-Step Data Flow

    | Step | Operation | Input Shape | Output Shape |
    |:----:|:----------|:-----------|:-------------|
    | 1 | `extract_features(sensor_window)` | `(batch, n_readings, 52)` | `(batch, 260)` |
    | 2 | `CLIP.encode_image(image)` | `(batch, 3, 224, 224)` | `(batch, 512)` |
    | 3 | `sensor_proj(sensor_feats)` | `(batch, 260)` | `(batch, 256)` |
    | 4 | `image_proj(clip_emb)` | `(batch, 512)` | `(batch, 256)` |
    | 5 | Concatenate `[CLS] + sensor + image` tokens | — | `(batch, 3, 256)` |
    | 6 | `TransformerEncoder` (2 layers, 4 heads) | `(3, batch, 256)` | `(3, batch, 256)` |
    | 7 | Extract `[CLS]` output | `(batch, 256)` | `(batch, 256)` |
    | 8 | `risk_head(cls_out)` → sigmoid | `(batch, 256)` | `(batch, 1)` |
    | 9 | `conf_head(cls_out)` → sigmoid | `(batch, 256)` | `(batch, 1)` |
    """
)

# ── Mermaid diagram (in expander since Streamlit doesn't natively render) ──
with st.expander("📊 Architecture Diagram (Mermaid source — paste into mermaid.live)"):
    st.code(
        """
flowchart TB
    subgraph Input["Input Modalities"]
        S["sensor_window\\n(N × 52 values)"]
        I["image bytes\\n(JPG/PNG)"]
        P["PDF bytes\\n(optional)"]
    end

    subgraph Preprocessing
        FE["extract_features()\\n→ 260-dim vector"]
        CE["CLIP ViT-B/32\\n→ 512-dim embedding"]
        PE["PyMuPDF\\nimage extraction"]
    end

    subgraph Models
        SBM["SensorBaselineModel\\nLightGBM (260-dim)"]
        JM["JointSensorImageModel\\nLightGBM (772-dim)"]
    end

    subgraph Fusion["TransformerCrossModalFusion"]
        SP["sensor_proj\\n260 → 256"]
        IP["image_proj\\n512 → 256"]
        CLS["[CLS] token\\nlearnable"]
        TE["TransformerEncoder\\n2 layers, 4 heads"]
        RH["risk_head → σ"]
        CH["conf_head → σ"]
    end

    subgraph GF["GatedFusion"]
        GW["Confidence-weighted\\nsoftmax attention"]
    end

    AB["Adaptive Blend\\nw = min(0.10 + n×0.125, 0.40)"]
    OUT["PredictionResponse\\nfailure_prob, confidence,\\ntop_signals, explanation"]

    S --> FE --> SBM
    FE --> SP
    I --> CE --> IP
    P --> PE --> CE
    FE --> JM
    CE --> JM
    SP --> TE
    IP --> TE
    CLS --> TE
    TE --> RH
    TE --> CH
    SBM --> GW
    CE --> GW
    RH --> AB
    GW --> AB
    JM --> AB
    AB --> OUT
        """,
        language="mermaid",
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MISSING MODALITY HANDLING
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">How Missing Modalities Are Handled</div>', unsafe_allow_html=True)

st.markdown(
    """
    The system **gracefully degrades** when inputs are missing. No single modality
    is required — the orchestrator adapts the inference path based on what's available:

    | Available Inputs | Inference Path | Models Used |
    |:----------------|:---------------|:-----------|
    | Sensor only | Baseline path | SensorBaselineModel → GatedFusion (sensor weight) |
    | Image only | Image path | CLIPImageEncoder → zero-shot classification |
    | Sensor + Image | Full multimodal | SensorBaseline + CLIP + Joint LightGBM + TransformerFusion + GatedFusion → Adaptive Blend |
    | PDF only | PDF → Image extraction | PyMuPDF extracts embedded images → CLIP path |
    | Nothing | Error 400 | Graceful error: "Provide at least one input" |

    **Edge cases handled in code:**
    - **Empty sensor window** → `extract_features()` returns zero vector → model still produces output
    - **Unreadable PDF** → `extract_pdf_images()` returns empty list → falls back to sensor-only or error
    - **Low-quality image** → CLIP still produces embedding (may have low confidence) → GatedFusion down-weights it
    - **NaN / Inf in sensor values** → `_sanitize_sensor_records()` replaces with `None` before processing
    """
)

with st.expander("🔧 Code: Orchestrator modality routing logic"):
    st.code(
        """
# From src/services/orchestrator.py (simplified)

async def predict(self, request: PredictionRequest):
    sensor_result = None
    image_result = None

    # 1. Sensor path (always attempted if sensor_window provided)
    if request.sensor_window:
        features = self.sensor_model.extract_features(request.sensor_window)
        sensor_result = self.sensor_model.predict(features)

    # 2. Image path (attempted if image_refs provided)
    if request.image_refs:
        image_result = self.image_encoder.encode_and_classify(image_path)

    # 3. Joint upgrade (if both modalities available)
    if sensor_result and image_result and self.joint_model:
        joint_features = concat(sensor_features, clip_embedding)  # 772-dim
        joint_result = self.joint_model.predict(joint_features)

    # 4. Fusion
    if sensor_result and image_result:
        # Full hybrid: GatedFusion + TransformerFusion + Adaptive Blend
        ...
    elif sensor_result:
        # Sensor-only baseline
        ...
    elif image_result:
        # Image-only classification
        ...
        """,
        language="python",
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# EXTENSIBILITY: ADDING NEW MODALITIES
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Adding New Modalities</div>', unsafe_allow_html=True)

st.markdown(
    """
    The architecture is designed to be **modality-extensible**. Adding a new modality
    (e.g., audio, vibration waveform, thermal camera) requires changes to:

    #### Extension Blueprint

    | Step | Action | What Changes | What Stays the Same |
    |:----:|:-------|:-------------|:-------------------|
    | 1 | **Add encoder module** | New file in `src/models/` | Core models unchanged |
    | 2 | **Add projection layer** | Register in TransformerFusion | Existing projectors unchanged |
    | 3 | **Register modality** | Update `MultimodalEncoderManager` | Fusion mechanism unchanged |
    | 4 | **Update fusion attention mask** | Add new token position | CLS + existing tokens unchanged |
    | 5 | **Update training pipeline** | Include new modality data + dropout | Existing training unchanged |

    #### Interface Contract (Pseudocode)

    ```python
    class NewModalityEncoder:
        \"\"\"Every encoder must satisfy this interface.\"\"\"

        def encode(self, raw_input: bytes) -> ModalityOutput:
            \"\"\"
            Encode raw input → standardised output.

            Returns:
                ModalityOutput(
                    embedding: np.ndarray,    # shape: (embedding_dim,)
                    confidence: float,        # 0–1
                    signals: List[str],       # contributing factors
                    metadata: dict            # encoder-specific info
                )
            \"\"\"
            ...

    # In TransformerCrossModalFusion:
    # Add projection layer:
    self.audio_proj = nn.Linear(audio_dim, d_model)

    # In fusion forward():
    # tokens = [CLS, sensor_token, image_token, audio_token]  # dynamic based on available modalities
    # mask = generate_mask(available_modalities)
    # output = transformer_encoder(tokens, src_key_padding_mask=mask)
    ```

    #### What Remains Unchanged
    - **TransformerEncoder** core (layers, heads, FFN)
    - **Risk / confidence heads** (read from [CLS] token)
    - **GatedFusion** (automatically includes new modality via softmax)
    - **API contract** (`PredictionResponse` schema)
    """
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# DATASET SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Dataset Summary</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Multimodal Training Set (Primary)")
    st.markdown(
        """
        | Property | Value |
        |:---------|:------|
        | Rows | 241 (120 NORMAL, 121 RECOVERING) |
        | Sensors | 52 channels (sensor_00–sensor_51) |
        | Images | 241 paired pump photos |
        | Join key | `serial_number` |
        | Label mapping | NORMAL↔Normal, RECOVERING↔Corroded |
        | Class balance | 49.8% / 50.2% (near-perfect) |
        | Missing values | sensor_00 NaN for RECOVERING; sensor_15 ~5% NaN |
        """
    )

with col2:
    st.markdown("#### Baseline Sensor Dataset")
    st.markdown(
        """
        | Property | Value |
        |:---------|:------|
        | Rows | 220,320 |
        | Columns | 55 (52 sensors + timestamp + status + serial_number) |
        | Labels | NORMAL (205,836), RECOVERING (14,477), BROKEN (7) |
        | Source | Kaggle (CC0 license) |
        | Usage | Optional full-scale training (`train_baseline_full.py`) |
        | Note | BROKEN collapsed into RECOVERING for binary task |
        """
    )

# ── Sample data preview ──
data = load_sample_data()
with st.expander("🔍 Sample Sensor Records"):
    tab_n, tab_r = st.tabs(["Normal", "Recovering"])
    with tab_n:
        if data.get("normal"):
            st.dataframe(pd.DataFrame(data["normal"][:3]), use_container_width=True)
    with tab_r:
        if data.get("recovering"):
            st.dataframe(pd.DataFrame(data["recovering"][:3]), use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# API SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">API Schema</div>', unsafe_allow_html=True)

col_req, col_res = st.columns(2)

with col_req:
    st.markdown("#### Request (`POST /predict`)")
    st.json({
        "asset_id": "pump_017",
        "timestamp": "2026-02-12T10:30:00Z",
        "sensor_window": [
            {"sensor_00": 2.44, "sensor_01": 46.31, "sensor_02": 52.34, "...": "..."}
        ],
        "image_refs": ["multimodal_model/images/img_001.png"],
    })

with col_res:
    st.markdown("#### Response")
    st.json({
        "asset_id": "pump_017",
        "failure_probability": 0.0045,
        "fault_confidence": 0.7964,
        "top_signals": ["flow_rate_anomaly", "pressure_drop", "temperature_rise"],
        "explanation": "Minimal failure risk (0%) with high confidence (80%).",
        "inference_ms": 2,
        "model_version": "v1.0.0",
    })

st.markdown("#### Multimodal File Upload (`POST /predict/multimodal`)")
st.code(
    """
curl -X POST http://localhost:8000/predict/multimodal \\
  -F "asset_id=pump_017" \\
  -F "images=@photo.jpg" \\
  -F "pdfs=@report.pdf" \\
  -F 'sensor_json=[{"sensor_00": 2.44}]'
    """,
    language="bash",
)

st.markdown("#### Batch (`POST /predict/batch`)")
st.json({
    "items": [
        {"asset_id": "pump_001", "timestamp": "2026-02-12T10:30:00Z",
         "sensor_window": [{"sensor_00": 2.44}], "image_refs": []},
        {"asset_id": "pump_002", "timestamp": "2026-02-12T11:00:00Z",
         "sensor_window": [{"sensor_00": 0.0}], "image_refs": []},
    ]
})

st.divider()
st.caption("See docs/ARCHITECTURE.md for the full technical specification, including tensor shapes and modality extension guide.")
