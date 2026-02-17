"""
🔧 Pump Fault Risk Prediction Service — Home / Overview

Entry point for the multi-page Streamlit demo.
Run:  streamlit run app/Home.py
"""

import streamlit as st
from pathlib import Path

# Shared helpers (Streamlit adds app/ to sys.path automatically)
from _shared import inject_css, render_sidebar, check_health, demo_script, PROJECT_ROOT

# ── Page config ──
st.set_page_config(
    page_title="Pump Fault Risk Predictor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_sidebar()

# ── Demo script ──
demo_script(
    "Welcome to the Pump Fault Risk Prediction demo. This system helps maintenance "
    "teams detect pumps that are at risk of failure <b>before</b> they break down, "
    "using real-time sensor data and optional inspection images. Let me walk you through it."
)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1565C0 0%, #42A5F5 100%);
                color: white; padding: 2rem 2.5rem; border-radius: 1rem;
                margin-bottom: 1.5rem;">
        <h1 style="margin:0; font-size:2.2rem;">🔧 Pump Fault Risk Prediction</h1>
        <p style="margin:0.5rem 0 0 0; font-size:1.15rem; opacity:0.92;">
            Multimodal ML system that predicts pump failure risk from
            <b>sensor telemetry</b> and <b>inspection images</b> — in real time.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM → SOLUTION → VALUE
# ══════════════════════════════════════════════════════════════════════════════

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        ### ❗ The Problem
        Unplanned pump failures cause **costly downtime** and safety hazards.
        Maintenance teams rely on manual inspections and fixed schedules,
        missing early warning signs hidden in sensor telemetry.
        """
    )

with c2:
    st.markdown(
        """
        ### 💡 Our Solution
        A **multimodal ML pipeline** that continuously analyses up to
        52 sensor channels plus inspection images to produce a
        real-time **failure probability** with explainable signals —
        enabling predictive maintenance.
        """
    )

with c3:
    st.markdown(
        """
        ### 📈 The Value
        - **Early detection** — days before failure
        - **Reduced downtime** — targeted, not scheduled, maintenance
        - **Explainable outputs** — top contributing signals
        - **Sub-10 ms inference** — fits real-time SCADA loops
        """
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# WHAT WE BUILT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">What We Built</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([3, 2])

with col_a:
    st.markdown(
        """
        | Component | Technology | Purpose |
        |:----------|:-----------|:--------|
        | **Sensor Model** | LightGBM (260 features) | Baseline fault classifier |
        | **Image Encoder** | CLIP ViT-B/32 (512-dim) | Zero-shot fault detection from photos |
        | **Joint Model** | LightGBM (772-dim) | Combined sensor + image input |
        | **Fusion Layer** | Trained TransformerCrossModalFusion | Cross-modal attention |
        | **API** | FastAPI + Uvicorn | REST endpoints for inference |
        | **UI** | Streamlit (this app) | Interactive demo & reporting |
        """
    )

with col_b:
    st.markdown(
        """
        ```
        Sensor Data (52 channels)
              │
              ├─► Feature Engineering (260-dim)
              │        │
              │        ├─► LightGBM Sensor Model
              │        │
        Images ─► CLIP Encoder (512-dim)
              │        │
              │        ├─► LightGBM Joint Model (772-dim)
              │        │
              └────────┴─► TransformerFusion (256-dim)
                              │
                              ▼
                     Failure Probability
                     + Top Signals
                     + Explanation
        ```
        """
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# HOW TO USE THIS DEMO
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">How to Use This Demo</div>', unsafe_allow_html=True)

steps = [
    ("1️⃣", "**Live Prediction**", "Upload sensor data or images and get instant risk predictions with confidence scores and explainable signals."),
    ("2️⃣", "**Model & Data**", "Explore the pipeline architecture, see dataset summaries, and understand how predictions are produced."),
    ("3️⃣", "**Evaluation**", "Review model performance metrics, ablation experiments, and latency profiling results."),
    ("4️⃣", "**Deployment & Scale**", "See load test results, throughput benchmarks, and the recommended production architecture."),
]

cols = st.columns(4)
for i, (icon, title, desc) in enumerate(steps):
    with cols[i]:
        st.markdown(
            f"""<div class="info-card">
            <div style="font-size:1.8rem; text-align:center;">{icon}</div>
            <div style="text-align:center; font-weight:600; margin:0.3rem 0;">{title}</div>
            <div style="font-size:0.88rem; color:#555;">{desc}</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM STATUS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)

health = check_health()

c1, c2, c3, c4 = st.columns(4)

with c1:
    if health:
        st.metric("API Status", "✅ Online")
    else:
        st.metric("API Status", "❌ Offline")

with c2:
    st.metric("Model Version", health.get("model_version", "—") if health else "—")

with c3:
    artifacts_dir = PROJECT_ROOT / "artifacts"
    n_artifacts = sum(1 for f in ["sensor_baseline.pkl", "joint_sensor_image.pkl", "transformer_fusion_trained.pt"]
                      if (artifacts_dir / f).exists())
    st.metric("Loaded Models", f"{n_artifacts}/3")

with c4:
    st.metric("Uptime", f"{health.get('uptime_s', 0):.0f}s" if health else "—")

# ── Environment info ──
with st.expander("🔧 Environment Details"):
    import sys
    import platform
    st.markdown(
        f"""
        | Property | Value |
        |:---------|:------|
        | Python | `{sys.version.split()[0]}` |
        | Platform | `{platform.platform()}` |
        | API URL | `{st.session_state.get('api_url', 'http://localhost:8000')}` |
        | Project Root | `{PROJECT_ROOT}` |
        | Artifacts Dir | `{artifacts_dir}` |
        """
    )

# ── Footer ──
st.divider()
st.caption(
    "Pump Fault Risk Prediction Service · v1.0.0 · "
    "Built with FastAPI, LightGBM, CLIP ViT-B/32, and TransformerCrossModalFusion · "
    "Navigate using the sidebar ⬅️"
)
