"""Shared helpers for the Streamlit multi-page app.

This module provides:
- Theme / CSS injection
- Sidebar rendering (health, artifacts)
- API client wrappers
- Display helpers (gauges, result cards)
- Sample data loading
"""

import io
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_URL = os.environ.get("API_URL", "http://localhost:8000")
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # app/ -> project root

# ---------------------------------------------------------------------------
# Theme / CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ── Global ── */
.block-container { padding-top: 1.5rem; }
section[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f8f9fa 0%, #e8ecf1 100%);
    border: 1px solid #dee2e6;
    border-radius: 0.6rem;
    padding: 0.75rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
div[data-testid="stMetric"] label { font-weight: 600; color: #495057; }

/* ── Info / success / warning boxes ── */
div[data-testid="stAlert"] { border-radius: 0.5rem; }

/* ── Demo script box ── */
.demo-script {
    background: #fffde7;
    border-left: 4px solid #fbc02d;
    padding: 0.75rem 1rem;
    border-radius: 0 0.5rem 0.5rem 0;
    margin-bottom: 1rem;
    font-size: 0.92rem;
}
.demo-script strong { color: #f57f17; }

/* ── Section headers ── */
.section-header {
    background: linear-gradient(90deg, #1E88E5 0%, #42A5F5 100%);
    color: white;
    padding: 0.6rem 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0 1rem 0;
    font-weight: 600;
}

/* ── Cards ── */
.info-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
</style>
"""


def inject_css():
    """Inject custom CSS into the current page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render the shared sidebar with health and artifacts."""
    with st.sidebar:
        # ── Health ──
        st.markdown("### 🏥 API Status")
        health = check_health()
        if health:
            st.success(f"Online — {health.get('model_version', '?')}")
            st.caption(f"Uptime: {health.get('uptime_s', 0):.0f}s")
        else:
            st.error("API Offline")
            st.caption(f"Endpoint: {API_URL}")

        st.divider()

        # ── Artifacts ──
        st.markdown("### 🧩 Model Artifacts")
        artifacts_dir = PROJECT_ROOT / "artifacts"
        checks = {
            "sensor_baseline.pkl": "Sensor LightGBM",
            "joint_sensor_image.pkl": "Joint LightGBM",
            "transformer_fusion_trained.pt": "Transformer Fusion",
        }
        for fname, label in checks.items():
            exists = (artifacts_dir / fname).exists()
            st.markdown(f"{'✅' if exists else '⬜'} {label}")
        st.divider()

        # ── Navigation help ──
        st.markdown("### 📖 Pages")
        st.caption(
            "1. **Overview** — What & why\n"
            "2. **Live Prediction** — Interactive demo\n"
            "3. **Model & Data** — Architecture\n"
            "4. **Evaluation** — Metrics & ablations\n"
            "5. **Deployment** — Scale & ops\n"
            "6. **About** — AI usage & docs"
        )


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5)
def check_health() -> Optional[dict]:
    if not HAS_HTTPX:
        return None
    try:
        with httpx.Client(timeout=5) as c:
            r = c.get(f"{API_URL}/health")
            return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _sanitize_records(records: list) -> list:
    """Replace NaN / Inf with None for JSON serialization."""
    cleaned = []
    for row in records:
        cleaned.append(
            {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
             for k, v in row.items()}
        )
    return cleaned


def api_predict(payload: dict) -> Optional[dict]:
    """POST /predict with a full PredictionRequest payload."""
    if not HAS_HTTPX:
        st.error("httpx is not installed.")
        return None
    try:
        with httpx.Client(timeout=30) as c:
            r = c.post(f"{API_URL}/predict", json=payload)
            if r.status_code == 200:
                return r.json()
            st.error(f"API {r.status_code}: {r.text[:300]}")
            return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def api_predict_sensor(sensor_window: list, asset_id: str = "demo_pump") -> Optional[dict]:
    """Sensor-only prediction."""
    payload = {
        "asset_id": asset_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sensor_window": _sanitize_records(sensor_window),
        "image_refs": [],
    }
    return api_predict(payload)


def api_predict_multimodal(
    *,
    asset_id: str = "demo_pump",
    sensor_json: Optional[str] = None,
    image_files=None,
    pdf_files=None,
) -> Optional[dict]:
    """POST /predict/multimodal (file upload endpoint)."""
    if not HAS_HTTPX:
        st.error("httpx is not installed.")
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
            r = c.post(
                f"{API_URL}/predict/multimodal",
                data=data,
                files=files if files else None,
            )
            if r.status_code == 200:
                return r.json()
            st.error(f"API {r.status_code}: {r.text[:300]}")
            return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def api_predict_batch(items: list) -> Optional[list]:
    """POST /predict/batch."""
    if not HAS_HTTPX:
        st.error("httpx is not installed.")
        return None
    try:
        with httpx.Client(timeout=60) as c:
            r = c.post(f"{API_URL}/predict/batch", json={"items": items})
            if r.status_code == 200:
                return r.json().get("predictions", [])
            st.error(f"API {r.status_code}: {r.text[:300]}")
            return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def make_gauge(value: float, title: str, color_ranges=None):
    """Create a Plotly gauge figure (returns None if plotly unavailable)."""
    if not HAS_PLOTLY:
        return None
    if color_ranges is None:
        color_ranges = [
            {"range": [0, 0.3], "color": "#2ecc71"},
            {"range": [0.3, 0.7], "color": "#f39c12"},
            {"range": [0.7, 1.0], "color": "#e74c3c"},
        ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "", "font": {"size": 36}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1},
            "bar": {"color": "#2c3e50"},
            "steps": color_ranges,
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 0.7},
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def confidence_gauge(value: float, title: str = "Confidence"):
    return make_gauge(value, title, [
        {"range": [0, 0.4], "color": "#e74c3c"},
        {"range": [0.4, 0.7], "color": "#f39c12"},
        {"range": [0.7, 1.0], "color": "#2ecc71"},
    ])


def display_prediction_result(result: dict, show_raw: bool = True):
    """Unified prediction result display (works for baseline & multimodal)."""
    fp = result.get("failure_probability", 0)
    conf = result.get("fault_confidence", 0)
    label = "RECOVERING (At Risk)" if fp > 0.5 else "NORMAL (Healthy)"
    color = "🔴" if fp > 0.7 else "🟡" if fp > 0.3 else "🟢"

    # ── Gauges row ──
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        fig = make_gauge(fp, "Failure Probability")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.metric("Failure Probability", f"{fp:.2%}")
    with c2:
        fig = confidence_gauge(conf)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.metric("Confidence", f"{conf:.2%}")
    with c3:
        st.metric("Status", f"{color} {label.split('(')[0].strip()}")
        st.metric("Latency", f"{result.get('inference_ms', '?')} ms")
        st.metric("Model", result.get("model_version", "?"))

    # ── Explanation ──
    explanation = result.get("explanation", "")
    if explanation:
        st.info(f"**Why this prediction:** {explanation}")

    # ── Top signals ──
    signals = result.get("top_signals", [])
    if signals:
        st.markdown("**Contributing Signals**")
        cols = st.columns(min(len(signals), 5))
        for i, sig in enumerate(signals):
            with cols[i % 5]:
                icon = "🔴" if fp > 0.7 else "🟡" if fp > 0.3 else "🟢"
                st.markdown(
                    f"<div style='background:#f8f9fa;border:1px solid #dee2e6;"
                    f"border-radius:0.4rem;padding:0.5rem;text-align:center;'>"
                    f"{icon}<br><small>{sig.replace('_', ' ').title()}</small></div>",
                    unsafe_allow_html=True,
                )

    # ── Raw JSON ──
    if show_raw:
        with st.expander("📄 Raw JSON Response"):
            st.json(result)


def display_batch_results(results: list):
    """Display batch prediction summary + table."""
    import pandas as pd

    normal = sum(1 for r in results if r.get("failure_probability", 0) <= 0.5)
    at_risk = len(results) - normal

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total", len(results))
    with c2:
        st.metric("🟢 Normal", normal)
    with c3:
        st.metric("🟡 At Risk", at_risk)
    with c4:
        avg_fp = np.mean([r.get("failure_probability", 0) for r in results])
        st.metric("Avg Risk", f"{avg_fp:.2%}")

    rows = []
    for i, r in enumerate(results):
        fp = r.get("failure_probability", 0)
        rows.append({
            "#": i + 1,
            "Asset": r.get("asset_id", "?"),
            "Risk": f"{fp:.4f}",
            "Status": "🟡 RECOVERING" if fp > 0.5 else "🟢 NORMAL",
            "Confidence": f"{r.get('fault_confidence', 0):.2%}",
            "Latency (ms)": r.get("inference_ms", "?"),
            "Top Signal": (r.get("top_signals") or ["—"])[0],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

@st.cache_data
def load_sample_data() -> dict:
    """Load sample data from artifacts/sample_data.json."""
    path = PROJECT_ROOT / "artifacts" / "sample_data.json"
    if path.exists():
        raw = path.read_text(encoding="utf-8")
        # Handle NaN in JSON (not standard)
        raw = raw.replace(": NaN", ": null").replace(":NaN", ":null")
        return json.loads(raw)
    return {"normal": [], "recovering": []}


@st.cache_data
def load_ablation_results() -> dict:
    """Load ablation_results.json."""
    path = PROJECT_ROOT / "ablation_results.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_load_test_results() -> dict:
    """Load saved load test JSON results."""
    out = {}
    for key in ("load_test_results_before", "load_test_results_after", "load_test_results"):
        path = PROJECT_ROOT / "artifacts" / f"{key}.json"
        if path.exists():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    return out


def get_sample_sensor(kind: str = "normal", n: int = 5) -> list:
    """Return n sample sensor records of the given kind."""
    data = load_sample_data()
    records = data.get(kind, [])
    return records[:n] if records else _fallback_sensor(n)


def _fallback_sensor(n: int = 5) -> list:
    """Generate synthetic sensor data as fallback."""
    np.random.seed(42)
    return [
        {f"sensor_{i:02d}": round(float(45 + np.random.normal(0, 5)), 2) for i in range(52)}
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Demo script helper
# ---------------------------------------------------------------------------

def demo_script(text: str):
    """Render a 'what to say to the client' box."""
    st.markdown(
        f'<div class="demo-script"><strong>🎬 Demo Script:</strong> {text}</div>',
        unsafe_allow_html=True,
    )
