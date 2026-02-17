"""
🧪 Live Prediction — Interactive Demo

Supports three modes:
- Sensor Only (baseline LightGBM)
- Multimodal (sensor + image via CLIP + Transformer fusion)
- Batch (bulk JSON or file uploads)
"""

import json
import time
from pathlib import Path

import numpy as np
import streamlit as st

from _shared import (
    API_URL,
    PROJECT_ROOT,
    inject_css,
    render_sidebar,
    demo_script,
    api_predict_sensor,
    api_predict_multimodal,
    api_predict_batch,
    api_predict,
    display_prediction_result,
    display_batch_results,
    get_sample_sensor,
    load_sample_data,
    _sanitize_records,
    HAS_HTTPX,
    HAS_PLOTLY,
    make_gauge,
)

st.set_page_config(page_title="Live Prediction", page_icon="🧪", layout="wide")
inject_css()
render_sidebar()

# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1565C0 0%, #42A5F5 100%);
                color: white; padding: 1.2rem 2rem; border-radius: 0.8rem;
                margin-bottom: 1rem;">
        <h2 style="margin:0;">🧪 Live Prediction Demo</h2>
        <p style="margin:0.3rem 0 0 0; opacity:0.9;">
            Upload sensor data, images, or both — get instant fault risk predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

demo_script(
    "Now I'll show you the system in action. We have three ways to interact: "
    "sensor-only predictions, multimodal (sensor + image), and batch processing. "
    "Let me start with a quick sensor prediction using sample data."
)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_sensor, tab_multi, tab_batch = st.tabs([
    "📊 Sensor Prediction",
    "🧠 Multimodal (Sensor + Image)",
    "📦 Batch Prediction",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Sensor-only baseline
# ──────────────────────────────────────────────────────────────────────────────
with tab_sensor:
    st.markdown("##### Use the sensor-only LightGBM model to predict fault risk from 52 sensor channels.")

    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.markdown("**Input Configuration**")
        asset_id = st.text_input("Asset ID", value="PUMP-001", key="s_asset")

        # ── Sample data buttons ──
        st.markdown("**Quick-Fill Sample Data**")
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("🟢 Load Normal Sample", use_container_width=True, key="s_normal"):
                st.session_state["s_json_input"] = json.dumps(
                    _sanitize_records(get_sample_sensor("normal", 5)), indent=2
                )
        with bc2:
            if st.button("🟡 Load At-Risk Sample", use_container_width=True, key="s_recover"):
                st.session_state["s_json_input"] = json.dumps(
                    _sanitize_records(get_sample_sensor("recovering", 5)), indent=2
                )

        # ── JSON input ──
        default_json = st.session_state.get(
            "s_json_input",
            json.dumps(_sanitize_records(get_sample_sensor("normal", 3)), indent=2),
        )

        st.caption("JSON array of dicts — keys: `sensor_00` through `sensor_51`")

        raw_json = st.text_area(
            "Sensor Window JSON",
            value=default_json,
            height=250,
            key="s_json_area",
            label_visibility="collapsed",
        )

        # Parse
        sensor_window = []
        try:
            sensor_window = json.loads(raw_json)
            if isinstance(sensor_window, dict):
                sensor_window = [sensor_window]
            st.success(f"✓ {len(sensor_window)} reading(s) parsed")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

        predict_clicked = st.button(
            "🚀 Predict", type="primary", use_container_width=True, key="s_predict"
        )

    with col_output:
        if predict_clicked and sensor_window:
            with st.spinner("Running sensor prediction…"):
                t0 = time.time()
                result = api_predict_sensor(sensor_window, asset_id)
                wall_ms = (time.time() - t0) * 1000

            if result:
                st.markdown("**Prediction Result**")
                display_prediction_result(result)
                st.caption(f"Wall-clock: {wall_ms:.0f} ms (includes network)")
        elif predict_clicked:
            st.warning("Enter valid sensor data first.")
        else:
            st.info("👈 Fill in sensor data and click **Predict** to see results here.")

            st.markdown(
                """
                **What happens behind the scenes:**
                1. Your sensor readings are sent to the API
                2. 260 statistical features are extracted (mean, std, min, max, range per sensor)
                3. A LightGBM model produces a failure probability
                4. Top contributing signals are identified
                5. A human-readable explanation is generated
                """
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Multimodal (sensor + image)
# ──────────────────────────────────────────────────────────────────────────────
with tab_multi:
    st.markdown("##### Combine sensor data with inspection images for improved predictions using Transformer fusion.")

    demo_script(
        "The multimodal mode is where our system really shines. "
        "When we provide both sensor data AND an image, the trained "
        "TransformerCrossModalFusion layer performs cross-attention between "
        "the two data types — giving us richer predictions."
    )

    col_in, col_out = st.columns([1, 1])

    with col_in:
        mm_asset = st.text_input("Asset ID", value="PUMP-MM-001", key="mm_asset")

        # ── Images ──
        st.markdown("**📷 Inspection Images** (any combination)")
        images = st.file_uploader(
            "Upload pump images",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            accept_multiple_files=True,
            key="mm_images",
            label_visibility="collapsed",
        )
        if images:
            img_cols = st.columns(min(len(images), 3))
            for i, img in enumerate(images):
                with img_cols[i % 3]:
                    st.image(img, caption=img.name, use_container_width=True)

        # ── PDFs ──
        st.markdown("**📄 PDF Reports** (optional)")
        pdfs = st.file_uploader(
            "Upload PDF inspection reports",
            type=["pdf"],
            accept_multiple_files=True,
            key="mm_pdfs",
            label_visibility="collapsed",
        )
        if pdfs:
            for p in pdfs:
                st.caption(f"📎 {p.name} ({len(p.getvalue()) / 1024:.0f} KB)")

        # ── Sensor data (optional) ──
        st.markdown("**📟 Sensor Data** (optional)")
        include_sensor = st.checkbox("Include sensor readings", value=False, key="mm_inc_sensor")
        sensor_json_str = None
        if include_sensor:
            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("🟢 Normal Sample", key="mm_s_normal", use_container_width=True):
                    st.session_state["mm_s_input"] = json.dumps(
                        _sanitize_records(get_sample_sensor("normal", 3)), indent=2
                    )
            with bc2:
                if st.button("🟡 At-Risk Sample", key="mm_s_risk", use_container_width=True):
                    st.session_state["mm_s_input"] = json.dumps(
                        _sanitize_records(get_sample_sensor("recovering", 3)), indent=2
                    )

            s_raw = st.text_area(
                "Sensor JSON",
                value=st.session_state.get("mm_s_input", "[]"),
                height=120,
                key="mm_s_area",
                label_visibility="collapsed",
            )
            try:
                sw = json.loads(s_raw)
                if sw:
                    sensor_json_str = json.dumps(_sanitize_records(sw if isinstance(sw, list) else [sw]))
                    st.success(f"✓ {len(sw) if isinstance(sw, list) else 1} reading(s)")
            except json.JSONDecodeError:
                st.error("Invalid JSON")

        # ── Audio / Video (not implemented) ──
        with st.expander("🔇 Audio / Video (not enabled)"):
            st.info(
                "Audio and video modalities are not implemented in this version. "
                "The architecture supports adding new encoders — see the "
                "Architecture documentation for the extension blueprint."
            )

        has_input = images or pdfs or sensor_json_str
        mm_predict = st.button(
            "🚀 Predict (Multimodal)", type="primary",
            use_container_width=True, key="mm_predict",
            disabled=not has_input,
        )

    with col_out:
        if mm_predict and has_input:
            with st.spinner("Running multimodal prediction…"):
                t0 = time.time()
                result = api_predict_multimodal(
                    asset_id=mm_asset,
                    sensor_json=sensor_json_str,
                    image_files=images if images else None,
                    pdf_files=pdfs if pdfs else None,
                )
                wall_ms = (time.time() - t0) * 1000

            if result:
                # Show what modalities were used
                used = []
                if sensor_json_str:
                    used.append("📟 Sensor")
                if images:
                    used.append(f"📷 {len(images)} Image(s)")
                if pdfs:
                    used.append(f"📄 {len(pdfs)} PDF(s)")
                st.markdown(f"**Modalities used:** {' · '.join(used)}")

                display_prediction_result(result)
                st.caption(f"Wall-clock: {wall_ms:.0f} ms")
        elif mm_predict:
            st.warning("Provide at least one input.")
        else:
            st.info("👈 Upload images, PDFs, or sensor data and click **Predict**.")

            st.markdown(
                """
                **How multimodal fusion works:**
                1. **Images** → CLIP ViT-B/32 extracts 512-dim visual embeddings
                2. **Sensor data** → 260 statistical features extracted
                3. Both are projected to a shared 256-dim space
                4. **TransformerCrossModalFusion** performs cross-attention
                5. A **GatedFusion** layer combines all signals adaptively
                6. The system works with **any subset** of modalities
                """
            )

            with st.expander("ℹ️ How joint training works"):
                st.markdown(
                    """
                    - Sensor data and image data are **joined by serial_number** (241 paired samples)
                    - Label mapping: `NORMAL` ↔ `Normal` (healthy) · `RECOVERING` ↔ `Corroded` (fault)
                    - The TransformerCrossModalFusion is trained **end-to-end** with BCE loss
                    - Cross-modal attention weights are learned, not random
                    - Both sensor-only and joint models achieve **AUC = 1.0**
                    """
                )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: Batch
# ──────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("##### Run predictions in bulk via the `/predict/batch` API endpoint.")

    # ── Upload or paste ──
    batch_method = st.radio(
        "Input method",
        ["Upload JSON file", "Paste JSON"],
        horizontal=True,
        key="batch_method",
    )

    batch_items = None

    if batch_method == "Upload JSON file":
        uf = st.file_uploader("Batch JSON (array of PredictionRequest objects)", type=["json"], key="b_upload")
        if uf:
            try:
                batch_items = json.loads(uf.read())
                if not isinstance(batch_items, list):
                    st.error("JSON must be an array.")
                    batch_items = None
                else:
                    st.success(f"✓ {len(batch_items)} item(s) loaded")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
    else:
        sample_batch = [
            {"asset_id": "pump_001", "timestamp": "2026-02-12T10:30:00Z",
             "sensor_window": [{"sensor_00": 2.44, "sensor_01": 46.3}], "image_refs": []},
            {"asset_id": "pump_002", "timestamp": "2026-02-12T11:00:00Z",
             "sensor_window": [{"sensor_00": 0.0, "sensor_04": 3.1}], "image_refs": []},
        ]
        raw = st.text_area(
            "Batch JSON",
            value=json.dumps(sample_batch, indent=2),
            height=200,
            key="b_paste",
        )
        try:
            batch_items = json.loads(raw)
            if not isinstance(batch_items, list):
                st.error("Must be a JSON array")
                batch_items = None
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

    if batch_items:
        with st.expander("Preview payload"):
            st.json(batch_items[:3])

    b_predict = st.button(
        "🚀 Run Batch", type="primary", use_container_width=True,
        key="b_predict", disabled=not batch_items,
    )

    if b_predict and batch_items:
        if len(batch_items) > 100:
            st.error("Maximum batch size is 100.")
        else:
            with st.spinner(f"Processing {len(batch_items)} items…"):
                results = api_predict_batch(batch_items)
            if results:
                st.success(f"✅ {len(results)} predictions returned")
                display_batch_results(results)

                # Download
                st.download_button(
                    "📥 Download Results (JSON)",
                    json.dumps(results, indent=2),
                    "batch_results.json",
                    "application/json",
                    key="b_download",
                )
