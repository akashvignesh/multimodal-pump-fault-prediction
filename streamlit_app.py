"""Streamlit UI for Pump Fault Risk Prediction Service.

Three modes:
1. **Baseline** – Upload / paste sensor CSV/JSON → predict using the
   sensor-only LightGBM model (NORMAL / RECOVERING).
2. **Multimodal** – Upload images, PDFs, and/or optional sensor data →
   predict fault risk using the multimodal fusion pipeline.
3. **Batch** – Bulk JSON prediction via the /predict/batch endpoint.

The multimodal pipeline uses a **jointly trained** sensor + CLIP image model:
- Sensor data (260-dim) and CLIP ViT-B/32 image embeddings (512-dim) are
  joined on serial_number during training (NORMAL↔Normal, RECOVERING↔Corroded).
- TransformerCrossModalFusion (2 layers, 4 heads) is trained end-to-end with
  BCE loss so cross-modal attention is no longer randomly initialised.
- Joint LightGBM on concatenated 772-dim features as an additional model.

Run with:
    streamlit run streamlit_app.py
"""
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# API Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Pump Fault Risk Predictor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- helpers ---------------------------------------------------------

def make_gauge(value: float, title: str, color_ranges=None):
    if not HAS_PLOTLY:
        return None
    if color_ranges is None:
        color_ranges = [
            {"range": [0, 0.3], "color": "#2ecc71"},
            {"range": [0.3, 0.7], "color": "#f39c12"},
            {"range": [0.7, 1.0], "color": "#e74c3c"},
        ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "", "font": {"size": 36}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1},
                "bar": {"color": "#2c3e50"},
                "steps": color_ranges,
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 0.7},
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def check_health():
    if not HAS_HTTPX:
        return None
    try:
        with httpx.Client(timeout=5) as c:
            r = c.get(f"{API_URL}/health")
            if r.status_code == 200:
                return r.json()
    except Exception:
        return None


# ---------- API callers -----------------------------------------------------

def call_predict(payload: dict):
    """Call original /predict JSON endpoint."""
    if not HAS_HTTPX:
        st.error("httpx not installed")
        return None
    try:
        with httpx.Client(timeout=30) as c:
            r = c.post(f"{API_URL}/predict", json=payload)
            return r.json() if r.status_code == 200 else _api_err(r)
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def _sanitize_sensor_records(records: list) -> list:
    """Replace NaN / Inf floats with None so JSON serialization succeeds."""
    import math
    cleaned = []
    for row in records:
        cleaned.append(
            {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
             for k, v in row.items()}
        )
    return cleaned


def call_predict_baseline(sensor_json_list: list, asset_id: str = "pump_baseline"):
    """Call /predict endpoint with sensor-only data."""
    if not HAS_HTTPX:
        st.error("httpx not installed")
        return None
    try:
        payload = {
            "asset_id": asset_id,
            "timestamp": "2026-01-01T00:00:00Z",
            "sensor_window": _sanitize_sensor_records(sensor_json_list),
            "image_refs": [],
        }
        with httpx.Client(timeout=30) as c:
            r = c.post(f"{API_URL}/predict", json=payload)
            return r.json() if r.status_code == 200 else _api_err(r)
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def call_predict_baseline_batch(sensor_records: list, asset_id: str = "pump_baseline"):
    """Call /predict for each individual record.
    
    Args:
        sensor_records: List of sensor dicts (each with sensor_00 to sensor_51)
        asset_id: Asset identifier
        
    Returns:
        List of prediction results, one per record. Returns None on error.
    """
    if not HAS_HTTPX:
        st.error("httpx not installed")
        return None
    
    results = []
    errors = []
    
    try:
        with httpx.Client(timeout=30) as c:
            for idx, record in enumerate(sensor_records):
                try:
                    payload = {
                        "asset_id": asset_id,
                        "timestamp": "2026-01-01T00:00:00Z",
                        "sensor_window": _sanitize_sensor_records([record]),
                        "image_refs": [],
                    }
                    r = c.post(f"{API_URL}/predict", json=payload)
                    if r.status_code == 200:
                        result = r.json()
                        result["_row_index"] = idx
                        results.append(result)
                    else:
                        errors.append(f"Row {idx}: API error {r.status_code}")
                except Exception as e:
                    errors.append(f"Row {idx}: {str(e)}")
        
        if errors:
            st.warning(f"⚠️ Encountered {len(errors)} errors:\n" + "\n".join(errors))
        
        return results if results else None
    except Exception as e:
        st.error(f"Batch prediction connection error: {e}")
        return None


def call_predict_multimodal(
    *,
    asset_id="upload",
    sensor_json=None,
    image_files=None,
    pdf_files=None,
):
    """Call /predict/multimodal file-upload endpoint."""
    if not HAS_HTTPX:
        st.error("httpx not installed")
        return None
    try:
        data = {"asset_id": asset_id}
        if sensor_json:
            data["sensor_json"] = sensor_json

        files = []
        for img in (image_files or []):
            files.append(("images", (img.name, img.getvalue(), "image/png")))
        for pdf in (pdf_files or []):
            files.append(("pdfs", (pdf.name, pdf.getvalue(), "application/pdf")))

        with httpx.Client(timeout=60) as c:
            r = c.post(f"{API_URL}/predict/multimodal", data=data, files=files if files else None)
            return r.json() if r.status_code == 200 else _api_err(r)
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def _api_err(r):
    st.error(f"API Error {r.status_code}: {r.text[:500]}")
    return None


# ---------- display helpers ------------------------------------------------

def display_multimodal_result(result: dict):
    """Display multimodal PredictionResponse."""
    c1, c2, c3 = st.columns(3)
    with c1:
        if HAS_PLOTLY:
            st.plotly_chart(make_gauge(result["failure_probability"], "Failure Probability"), use_container_width=True)
        else:
            st.metric("Failure Probability", f"{result['failure_probability']:.2%}")
    with c2:
        if HAS_PLOTLY:
            st.plotly_chart(
                make_gauge(
                    result["fault_confidence"],
                    "Confidence",
                    [
                        {"range": [0, 0.4], "color": "#e74c3c"},
                        {"range": [0.4, 0.7], "color": "#f39c12"},
                        {"range": [0.7, 1.0], "color": "#2ecc71"},
                    ],
                ),
                use_container_width=True,
            )
        else:
            st.metric("Confidence", f"{result['fault_confidence']:.2%}")
    with c3:
        st.metric("Inference Time", f"{result.get('inference_ms', '?')} ms")
        st.metric("Model Version", result.get("model_version", "?"))
        st.metric("Asset ID", result.get("asset_id", "?"))

    if result.get("explanation"):
        st.info(f"💡 **Explanation:** {result['explanation']}")

    if result.get("top_signals"):
        st.subheader("🔍 Top Signals")
        cols = st.columns(len(result["top_signals"]))
        for i, sig in enumerate(result["top_signals"]):
            with cols[i]:
                sev = "🔴" if result["failure_probability"] > 0.7 else "🟡" if result["failure_probability"] > 0.3 else "🟢"
                st.info(f"{sev} {sig}")

    with st.expander("📄 Raw JSON"):
        st.json(result)


def display_baseline_result(result: dict):
    """Display baseline prediction (failure probability + confidence)."""
    fp = result.get("failure_probability", 0)
    conf = result.get("fault_confidence", 0)
    label = "RECOVERING" if fp > 0.5 else "NORMAL"

    # Colour badge
    colour_map = {"NORMAL": "🟢", "RECOVERING": "🟡"}
    badge = colour_map.get(label, "⚪")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"### {badge} Predicted: **{label}**")
        st.metric("Failure Probability", f"{fp:.2%}")
    with c2:
        st.metric("Fault Confidence", f"{conf:.2%}")
        st.metric("Inference", f"{result.get('inference_ms', 0)} ms")
    with c3:
        st.metric("Model", result.get("model_version", "?"))
        st.metric("Asset", result.get("asset_id", "?"))
        if result.get("top_signals"):
            st.write("**Top signals:**", ", ".join(result["top_signals"]))

    if result.get("explanation"):
        st.info(f"💡 **Explanation:** {result['explanation']}")

    # Bar chart
    if HAS_PLOTLY:
        fig = go.Figure(
            go.Bar(
                x=["Failure", "Normal"],
                y=[fp, 1 - fp],
                marker_color=["#e74c3c", "#2ecc71"],
            )
        )
        fig.update_layout(title="Failure Probability", yaxis_range=[0, 1], height=300)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Raw JSON"):
        st.json(result)


def display_baseline_batch_results(results: list):
    """Display batch baseline predictions (one per row)."""
    st.subheader(f"📋 Results ({len(results)} records)")
    
    # Summary stats
    normal_count = sum(1 for r in results if r.get("failure_probability", 0) <= 0.5)
    recovering_count = len(results) - normal_count
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(results))
    with col2:
        st.metric("🟢 NORMAL", normal_count)
    with col3:
        st.metric("🟡 RECOVERING", recovering_count)
    
    st.divider()
    
    # Individual results table
    table_data = []
    for idx, result in enumerate(results):
        fp = result.get("failure_probability", 0)
        conf = result.get("fault_confidence", 0)
        label = "RECOVERING" if fp > 0.5 else "NORMAL"
        badge = "🟡" if fp > 0.5 else "🟢"
        
        table_data.append({
            "Row": idx + 1,
            "Status": f"{badge} {label}",
            "Failure Prob": f"{fp:.4f}",
            "Confidence": f"{conf:.2%}",
            "Top Signals": ", ".join(result.get("top_signals", [])[:3]),
        })
    
    df_results = pd.DataFrame(table_data)
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Expandable detailed views
    for idx, result in enumerate(results):
        with st.expander(f"🔍 Row {idx + 1} Details", expanded=(idx == 0)):
            col1, col2 = st.columns([1, 2])
            with col1:
                fp = result.get("failure_probability", 0)
                conf = result.get("fault_confidence", 0)
                label = "RECOVERING" if fp > 0.5 else "NORMAL"
                badge = "🟡" if fp > 0.5 else "🟢"
                
                st.markdown(f"**{badge} {label}** ({conf:.2%})")
                st.metric("Failure Probability", f"{fp:.4f}")
                st.metric("Fault Confidence", f"{conf:.4f}")
            
            with col2:
                if HAS_PLOTLY:
                    fig = go.Figure(
                        go.Bar(
                            x=["Failure", "Normal"],
                            y=[fp, 1 - fp],
                            marker_color=["#e74c3c", "#2ecc71"],
                        )
                    )
                    fig.update_layout(title="Failure Probability", yaxis_range=[0, 1], height=250, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            if result.get("top_signals"):
                st.write("**Top signals:**", ", ".join(result["top_signals"]))
            
            with st.expander("Raw JSON"):
                st.json(result)




def baseline_tab():
    """Baseline: upload/paste sensor data → predict NORMAL/RECOVERING."""
    st.header("📊 Baseline Prediction")
    st.markdown(
        "Use the sensor-only baseline model (LightGBM with Optuna-tuned hyperparameters) "
        "to predict pump fault risk from sensor readings."
    )

    asset_id = st.text_input("Asset ID", value="pump_baseline", key="bl_asset")

    # ----- Load real sample data for downloads -----------------------------------------
    try:
        with open("artifacts/sample_data.json", "r") as f:
            sample_data = json.load(f)
        sample_normal = sample_data.get("normal", [])
        sample_recovering = sample_data.get("recovering", [])
    except Exception:
        sample_normal = []
        sample_recovering = []

    # ----- Download Sample CSVs -----------------------------------------
    st.subheader("⬇️ Download Sample CSV")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if sample_normal:
            sample_csv_df = pd.DataFrame(sample_normal)
            csv_bytes = sample_csv_df.to_csv(index=False).encode()
            st.download_button(
                "🟢 NORMAL",
                csv_bytes,
                file_name="sample_baseline_normal.csv",
                mime="text/csv",
                use_container_width=True,
            )
    
    with col2:
        if sample_recovering:
            sample_csv_df = pd.DataFrame(sample_recovering)
            csv_bytes = sample_csv_df.to_csv(index=False).encode()
            st.download_button(
                "🟡 RECOVERING",
                csv_bytes,
                file_name="sample_baseline_recovering.csv",
                mime="text/csv",
                use_container_width=True,
            )
    


    st.divider()

    sensor_window = []

    # Initialize default JSON if not in session state
    if "bl_json_data" not in st.session_state:
        # Try to use real sample data
        if sample_normal:
            default_json = json.dumps(sample_normal, indent=2)
        else:
            # Fallback to generated data
            default_json = json.dumps(
                [
                    {f"sensor_{i:02d}": round(45.0 + np.random.normal(0, 2), 2) for i in range(52)}
                    for _ in range(5)
                ],
                indent=2,
            )
    else:
        default_json = st.session_state.bl_json_data
    
    st.markdown("**JSON Format:** Array of objects, each with 52 sensor keys (`sensor_00` … `sensor_51`)")
    raw = st.text_area("Sensor JSON (array of dicts)", value=default_json, height=300, key="bl_json")
    try:
        sensor_window = json.loads(raw)
        if isinstance(sensor_window, dict):
            sensor_window = [sensor_window]
        st.info(f"✓ Valid JSON: {len(sensor_window)} record(s)")
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        sensor_window = []

    st.divider()

    if st.button("🚀 Predict (Baseline)", type="primary", use_container_width=True, key="bl_btn"):
        if not sensor_window:
            st.warning("Provide sensor data first.")
        else:
            with st.spinner(f"Running predictions for {len(sensor_window)} record(s)…"):
                results = call_predict_baseline_batch(sensor_window, asset_id)
            if results:
                st.success(f"✓ Completed {len(results)}/{len(sensor_window)} predictions!")
                display_baseline_batch_results(results)
    
    # Example section - show real examples from each class
    with st.expander("💡 Example JSON (Real Data from All 3 Classes)", expanded=False):
        try:
            with open("artifacts/example_data.json", "r") as f:
                example_data = json.load(f)
            
            tab1, tab2 = st.tabs(["🟢 NORMAL", "🟡 RECOVERING"])
            
            with tab1:
                st.json(example_data.get("normal", []))
            
            with tab2:
                st.json(example_data.get("recovering", []))
        except Exception as e:
            st.error(f"Could not load example data: {e}")


def multimodal_tab():
    """Multimodal: upload image/PDF + optional sensor → predict."""
    st.header("🧠 Multimodal Prediction")
    st.markdown(
        "Upload **any combination** of images and PDFs, with optional sensor data. "
        "The model fuses all available modalities using a **trained** Transformer "
        "cross-modal fusion layer (jointly trained on sensor + CLIP image data) "
        "to predict pump fault risk."
    )

    with st.expander("ℹ️ How joint training works", expanded=False):
        st.markdown(
            """
**Data mapping** (from `image_mapping.csv`):
- `NORMAL` machine status ↔ `Normal` image type (healthy pump)
- `RECOVERING` machine status ↔ `Corroded` image type (fault indicator)

**Joint training pipeline** (`train_joint_multimodal.py`):
1. Sensor data and image data are **joined** on `serial_number` (241 paired samples)
2. Sensor features: 260-dim (5 stats × 52 sensors)
3. Image features: 512-dim CLIP ViT-B/32 embeddings (frozen encoder)
4. **LightGBM** trained on concatenated 772-dim vector → perfect separation
5. **TransformerCrossModalFusion** trained end-to-end (2 layers, 4 heads, BCE loss)
   so sensor embeddings attend to image embeddings and vice versa

**Result**: Both models achieve 1.0 AUC on the test set, with cross-modal
attention weights now learned from real data instead of random initialization.
            """
        )

    asset_id = st.text_input("Asset ID", value="pump_upload", key="mm_asset")

    # ----- Inputs ---------------------------------------------------------
    st.subheader("📷 Images")
    images = st.file_uploader(
        "Upload images (png, jpg, jpeg, bmp, webp)",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=True,
        key="mm_img",
    )
    if images:
        cols = st.columns(min(len(images), 4))
        for i, img in enumerate(images):
            with cols[i % 4]:
                st.image(img, caption=img.name, use_container_width=True)

    st.subheader("📄 PDFs")
    pdfs = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="mm_pdf",
    )
    if pdfs:
        for p in pdfs:
            st.caption(f"📎 {p.name} ({len(p.getvalue()) / 1024:.0f} KB)")



    st.subheader("📟 Sensor Data")
    use_sensor = st.checkbox("Include sensor data", value=False, key="mm_sensor_chk")
    sensor_json_str = None
    if use_sensor:
        raw = st.text_area("Sensor JSON", value="[]", height=100, key="mm_s_json")
        try:
            sw = json.loads(raw)
            if sw:
                sensor_json_str = json.dumps(sw)
        except json.JSONDecodeError:
            st.warning("Invalid JSON")

    # ----- Predict --------------------------------------------------------
    if st.button("🚀 Predict (Multimodal)", type="primary", use_container_width=True, key="mm_btn"):
        # Ensure at least one input
        has_any = images or pdfs or sensor_json_str
        if not has_any:
            st.warning("Please provide at least one input (image, PDF, or sensor).")
            return

        with st.spinner("Running multimodal prediction…"):
            result = call_predict_multimodal(
                asset_id=asset_id,
                sensor_json=sensor_json_str,
                image_files=images if images else None,
                pdf_files=pdfs if pdfs else None,
            )
        if result:
            st.success("Prediction complete!")
            display_multimodal_result(result)


def batch_prediction_tab():
    """Batch prediction supporting JSON sensor data, images, and PDFs."""
    st.header("📦 Batch Prediction")
    st.markdown(
        "Run predictions in bulk. Upload a **JSON array** of sensor requests, "
        "multiple **images**, and/or multiple **PDFs**."
    )

    # ---- 1. Sensor JSON batch (optional) ----------------------------------
    st.subheader("📊 Sensor JSON Batch")
    sample = [
        {
            "asset_id": "pump_001",
            "timestamp": "2026-02-12T10:30:00Z",
            "sensor_window": [{"sensor_00": 2.44, "sensor_01": 46.31}],
            "image_refs": [],
        },
        {
            "asset_id": "pump_002",
            "timestamp": "2026-02-12T11:30:00Z",
            "sensor_window": [{"sensor_00": 0.0, "sensor_04": 3.1}],
            "image_refs": [],
        },
    ]
    st.download_button(
        "📥 Sample batch JSON",
        data=json.dumps(sample, indent=2),
        file_name="sample_batch.json",
        mime="application/json",
        key="batch_sample_dl",
    )

    uploaded_json = st.file_uploader("Upload Batch JSON", type=["json"], key="batch_up")
    json_items = None
    if uploaded_json:
        try:
            json_items = json.loads(uploaded_json.read())
            if not isinstance(json_items, list):
                st.error("Must be a JSON array")
                json_items = None
            else:
                st.info(f"✓ Loaded {len(json_items)} sensor batch items")
                with st.expander("Preview"):
                    st.json(json_items[:3])
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

    st.divider()

    # ---- 2. Image batch (optional) ----------------------------------------
    st.subheader("🖼️ Image Batch")
    batch_images = st.file_uploader(
        "Upload multiple images",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
        key="batch_imgs",
    )
    if batch_images:
        st.info(f"✓ {len(batch_images)} image(s) uploaded")

    st.divider()

    # ---- 3. PDF batch (optional) ------------------------------------------
    st.subheader("📄 PDF Batch")
    batch_pdfs = st.file_uploader(
        "Upload multiple PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="batch_pdfs",
    )
    if batch_pdfs:
        st.info(f"✓ {len(batch_pdfs)} PDF(s) uploaded")

    st.divider()

    # ---- 4. Run predictions -----------------------------------------------
    has_any = json_items or batch_images or batch_pdfs
    if st.button("🚀 Run Batch", type="primary", use_container_width=True, key="batch_btn", disabled=not has_any):
        if not HAS_HTTPX:
            st.error("httpx not installed")
            return

        all_results: list = []
        errors: list = []

        # -- 4a. JSON sensor batch via /predict/batch -----------------------
        if json_items:
            with st.spinner(f"Running sensor batch ({len(json_items)} items)…"):
                try:
                    with httpx.Client(timeout=60) as c:
                        r = c.post(f"{API_URL}/predict/batch", json={"items": json_items})
                    if r.status_code == 200:
                        preds = r.json().get("predictions", [])
                        for p in preds:
                            p["_source"] = "sensor_json"
                        all_results.extend(preds)
                    else:
                        errors.append(f"Sensor batch API error {r.status_code}: {r.text[:200]}")
                except Exception as e:
                    errors.append(f"Sensor batch error: {e}")

        # -- 4b. Image batch via /predict/multimodal (one per image) --------
        if batch_images:
            with st.spinner(f"Running image predictions ({len(batch_images)} images)…"):
                progress = st.progress(0, text="Processing images…")
                try:
                    with httpx.Client(timeout=60) as c:
                        for i, img in enumerate(batch_images):
                            try:
                                files = [("images", (img.name, img.getvalue(), "image/png"))]
                                data = {"asset_id": f"img_{img.name}"}
                                resp = c.post(f"{API_URL}/predict/multimodal", data=data, files=files)
                                if resp.status_code == 200:
                                    result = resp.json()
                                    result["_source"] = f"image: {img.name}"
                                    all_results.append(result)
                                else:
                                    errors.append(f"Image {img.name}: API error {resp.status_code}")
                            except Exception as e:
                                errors.append(f"Image {img.name}: {e}")
                            progress.progress((i + 1) / len(batch_images), text=f"Image {i+1}/{len(batch_images)}")
                except Exception as e:
                    errors.append(f"Image batch error: {e}")

        # -- 4c. PDF batch via /predict/multimodal (one per PDF) ------------
        if batch_pdfs:
            with st.spinner(f"Running PDF predictions ({len(batch_pdfs)} PDFs)…"):
                progress = st.progress(0, text="Processing PDFs…")
                try:
                    with httpx.Client(timeout=60) as c:
                        for i, pdf in enumerate(batch_pdfs):
                            try:
                                files = [("pdfs", (pdf.name, pdf.getvalue(), "application/pdf"))]
                                data = {"asset_id": f"pdf_{pdf.name}"}
                                resp = c.post(f"{API_URL}/predict/multimodal", data=data, files=files)
                                if resp.status_code == 200:
                                    result = resp.json()
                                    result["_source"] = f"pdf: {pdf.name}"
                                    all_results.append(result)
                                else:
                                    errors.append(f"PDF {pdf.name}: API error {resp.status_code}")
                            except Exception as e:
                                errors.append(f"PDF {pdf.name}: {e}")
                            progress.progress((i + 1) / len(batch_pdfs), text=f"PDF {i+1}/{len(batch_pdfs)}")
                except Exception as e:
                    errors.append(f"PDF batch error: {e}")

        # -- 5. Display results ---------------------------------------------
        if errors:
            st.warning(f"⚠️ {len(errors)} error(s):\n" + "\n".join(errors))

        if all_results:
            st.success(f"✅ {len(all_results)} prediction(s) returned")
            df = pd.DataFrame(all_results)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Avg Prob", f"{df['failure_probability'].mean():.3f}")
            with c2:
                st.metric("Avg Conf", f"{df['fault_confidence'].mean():.3f}")
            with c3:
                st.metric("Avg ms", f"{df['inference_ms'].mean():.0f}")
            with c4:
                st.metric("High Risk", f"{(df['failure_probability'] > 0.7).sum()}/{len(df)}")

            display_cols = ["asset_id", "failure_probability", "fault_confidence", "top_signals", "inference_ms"]
            if "_source" in df.columns:
                display_cols.insert(1, "_source")
            st.dataframe(df[display_cols], use_container_width=True)
            st.download_button(
                "📥 Results JSON",
                json.dumps(all_results, indent=2),
                "batch_results.json",
                "application/json",
                key="batch_results_dl",
            )
        elif not errors:
            st.info("No inputs provided.")


# ---------- main -----------------------------------------------------------

def main():
    st.title("🔧 Pump Fault Risk Prediction")
    st.markdown("Multimodal predictive service for pump fault risk assessment")

    # Sidebar
    with st.sidebar:
        st.header("🏥 Service Status")
        health = check_health()
        if health:
            st.success(f"✅ {health.get('status', 'unknown')}")
            st.info(f"Model: {health.get('model_version', '?')}")
            st.info(f"Uptime: {health.get('uptime_s', 0):.0f}s")
        else:
            st.error("❌ API not reachable")
            st.warning(f"Trying: {API_URL}")

        st.divider()

        # Show artifact status
        st.markdown("### 🧩 Artifacts")
        artifacts_dir = Path("artifacts")
        _artifact_checks = {
            "sensor_baseline.pkl": "Sensor-only LightGBM (260-dim)",
            "joint_sensor_image.pkl": "Joint Sensor+Image LightGBM (772-dim)",
            "transformer_fusion_trained.pt": "Trained Transformer Fusion",
        }
        for fname, desc in _artifact_checks.items():
            exists = (artifacts_dir / fname).exists()
            icon = "✅" if exists else "⬜"
            st.markdown(f"{icon} **{fname}**  \n{desc}")

        st.divider()
        st.markdown(
            """
### 📖 Modes
- **Baseline** – sensor-only LightGBM model (NORMAL / RECOVERING)
- **Multimodal** – sensor + image (CLIP ViT-B/32) with Transformer fusion
- **Batch** – bulk prediction via JSON, images, and/or PDFs

### 🔗 Joint Training
Sensor data and pump images are **joined** on `serial_number`
and trained together (NORMAL↔Normal, RECOVERING↔Corroded).
When both sensor and image inputs are present the API automatically
uses the joint model for improved predictions.
"""
        )

    # Tabs
    tab_bl, tab_mm, tab_batch = st.tabs(
        ["📊 Baseline", "🧠 Multimodal", "📦 Batch"]
    )
    with tab_bl:
        baseline_tab()
    with tab_mm:
        multimodal_tab()
    with tab_batch:
        batch_prediction_tab()


if __name__ == "__main__":
    main()
