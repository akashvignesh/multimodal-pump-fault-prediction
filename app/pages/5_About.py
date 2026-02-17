"""
ℹ️ About — AI usage declaration, documentation links, and project information.
"""

import streamlit as st
from pathlib import Path

from _shared import inject_css, render_sidebar, demo_script, PROJECT_ROOT

st.set_page_config(page_title="About & AI Usage", page_icon="ℹ️", layout="wide")
inject_css()
render_sidebar()

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1565C0 0%, #42A5F5 100%);
                color: white; padding: 1.2rem 2rem; border-radius: 0.8rem;
                margin-bottom: 1rem;">
        <h2 style="margin:0;">ℹ️ About & AI Usage</h2>
        <p style="margin:0.3rem 0 0 0; opacity:0.9;">
            Transparency on AI-assisted development and project documentation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

demo_script(
    "We believe in transparency about AI usage. This page shows exactly which "
    "parts of the system were AI-assisted and how we verified correctness. "
    "Every claim is backed by automated tests and reproducible benchmarks."
)

# ══════════════════════════════════════════════════════════════════════════════
# AI USAGE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">AI-Assisted Development</div>', unsafe_allow_html=True)

st.markdown(
    """
    This project was developed with **GitHub Copilot (Claude)** as an AI coding
    assistant. Below is a summary of AI vs. human contributions:

    | Component | AI Contribution | Human Contribution |
    |:----------|:---------------|:-------------------|
    | **Model architecture** | Code generation for modules | Architecture decisions, hyperparameter choices |
    | **Feature engineering** | Implementation of extraction functions | Design of 5-stat × 52-sensor scheme |
    | **Training scripts** | Boilerplate, Optuna integration | Training strategy, validation design |
    | **API endpoints** | Route handlers, schema definitions | API design, error handling strategy |
    | **Preprocessing** | Anomaly detection functions | Threshold tuning, anomaly signal design |
    | **Fusion layer** | PyTorch nn.Module implementation | Fusion strategy (hybrid Transformer + Gated) |
    | **Documentation** | All .md files (draft) | Review, accuracy verification |
    | **Performance optimization** | NumPy rewrite of hot-path functions | Profiling, bottleneck identification |
    | **This Streamlit app** | Page layouts, component code | UX design, demo narrative |
    """
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION METHODS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">How We Verified Correctness</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        #### Automated Verification
        | Method | Coverage |
        |:-------|:---------|
        | `pytest tests/ -v` | 10 tests — health, prediction, preprocessing |
        | Live endpoint testing | curl + httpx against running API |
        | Load testing (3 levels) | 5/25/75 concurrent users × 20s — zero errors |
        | Cross-validation | 5-fold stratified CV on all models |
        """
    )

with col2:
    st.markdown(
        """
        #### Manual Verification
        | Method | Scope |
        |:-------|:------|
        | Code review | All AI-generated code reviewed line-by-line |
        | Profiling | `time.perf_counter()` instrumentation on hot paths |
        | Ablation study | 12 experiments validating each architectural choice |
        | Edge case testing | NaN inputs, empty windows, missing modalities |
        """
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# WHAT WAS NOT AI-GENERATED
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">What Was NOT AI-Generated</div>', unsafe_allow_html=True)

st.markdown(
    """
    The following decisions were made entirely by the human engineer:

    1. **Problem framing** — binary classification (NORMAL vs RECOVERING) on pump sensor data
    2. **Dataset selection** — Kaggle pump sensor data + curated image mapping
    3. **Architecture design** — hybrid fusion (Transformer + Gated), not a single end-to-end model
    4. **Modality choice** — sensor + image (not audio/video, which lack training data)
    5. **CLIP frozen strategy** — justified by ablation (AUC=1.0 without fine-tuning)
    6. **Optimization targets** — identified pandas as bottleneck via profiling, not guessing
    7. **Label mapping** — confirmed NORMAL↔Normal, RECOVERING↔Corroded via data analysis
    8. **Cache sizes** — 1024 response + 2048 prediction entries based on memory budget
    9. **Deployment architecture** — ECS Fargate based on cost/scale tradeoffs
    """
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION INDEX
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Project Documentation</div>', unsafe_allow_html=True)

docs = [
    ("README.md", "Setup, usage, API reference, deployment"),
    ("docs/ARCHITECTURE.md", "Full technical architecture + transformer flow + extensibility"),
    ("docs/EVALUATION.md", "Model metrics, ablation results, error analysis, limitations"),
    ("docs/LOAD_SCALE.md", "Load test methodology, results, scaling strategy"),
    ("docs/DEMO_SCRIPT.md", "5-minute client walkthrough + talking points"),
    ("AI_USAGE.md", "AI usage declaration (this page, in detail)"),
    ("optimization_study.md", "Fine-tuning feasibility, deployment architecture, latency optimization"),
    ("evaluation_report.md", "Comprehensive model evaluation"),
    ("load_scale_report.md", "Load test analysis"),
    ("ablation_results.json", "Machine-readable experiment data (12 experiments)"),
    ("DATA_MANIFEST.md", "Dataset descriptions, schemas, licenses"),
]

for doc, desc in docs:
    full_path = PROJECT_ROOT / doc
    exists = full_path.exists()
    icon = "📄" if exists else "⬜"
    status = "" if exists else " *(not found)*"
    st.markdown(f"{icon} **{doc}**{status} — {desc}")

# ── Show full AI_USAGE.md ──
st.divider()
ai_usage_path = PROJECT_ROOT / "AI_USAGE.md"
if ai_usage_path.exists():
    with st.expander("📄 Full AI_USAGE.md"):
        st.markdown(ai_usage_path.read_text(encoding="utf-8"))

st.divider()
st.caption("Pump Fault Risk Prediction Service · v1.0.0 · Developed with GitHub Copilot (Claude)")
