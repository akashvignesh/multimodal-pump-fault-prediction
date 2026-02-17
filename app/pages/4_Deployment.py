"""
🚀 Deployment & Scale — Load test results, architecture, and operational concerns.
"""

import json
import streamlit as st
import pandas as pd

from _shared import (
    inject_css, render_sidebar, demo_script,
    PROJECT_ROOT, load_load_test_results, HAS_PLOTLY,
)

st.set_page_config(page_title="Deployment & Scale", page_icon="🚀", layout="wide")
inject_css()
render_sidebar()

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1565C0 0%, #42A5F5 100%);
                color: white; padding: 1.2rem 2rem; border-radius: 0.8rem;
                margin-bottom: 1rem;">
        <h2 style="margin:0;">🚀 Deployment & Scale</h2>
        <p style="margin:0.3rem 0 0 0; opacity:0.9;">
            Production architecture, load test results, and scaling strategy.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

demo_script(
    "Let me walk you through how this system performs under load and how "
    "we'd deploy it in production. We tested with up to 75 concurrent users "
    "and achieved 385 requests/second at light load with sub-10ms latency."
)

# ══════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Recommended Production Architecture</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown(
        """
        ```
        ┌──────────────┐      ┌─────────────────────────┐
        │  SCADA / PLCs  │────▶│   Application Load       │
        │  Sensor feeds  │     │   Balancer (ALB)         │
        └──────────────┘      └───────────┬───────────────┘
                                          │
        ┌──────────────┐      ┌───────────▼───────────────┐
        │  Streamlit UI  │────▶│   ECS Fargate Cluster     │
        │  (:8501)       │     │                           │
        └──────────────┘      │   ┌─────────────────────┐ │
                               │   │ pump-risk-api        │ │
        ┌──────────────┐      │   │ 2 vCPU / 4 GB RAM   │ │
        │  Monitoring    │◀────│   │ × 2–10 containers   │ │
        │  CloudWatch    │     │   └─────────────────────┘ │
        └──────────────┘      └───────────┬───────────────┘
                                          │
                               ┌──────────▼──────────────┐
                               │  Redis (shared cache)    │
                               │  PostgreSQL (audit logs)  │
                               └──────────────────────────┘
        ```
        """
    )

with col2:
    st.markdown(
        """
        | Component | Spec |
        |:----------|:-----|
        | **Compute** | ECS Fargate (2 vCPU / 4 GB per task) |
        | **Auto-scaling** | 2–10 tasks, CPU target 70% |
        | **Load Balancer** | ALB with health check on `/health` |
        | **Cache** | Redis ElastiCache (prediction dedup) |
        | **Database** | PostgreSQL RDS (audit log) |
        | **Model Storage** | S3 (versioned artifacts) |
        | **Monitoring** | CloudWatch + Prometheus |
        | **Est. Cost** | ~$198/mo steady-state (2 tasks) |
        """
    )

# Architecture diagram (Mermaid)
with st.expander("📊 Deployment Diagram (Mermaid source)"):
    st.code(
        """
flowchart TB
    subgraph External
        SCADA["SCADA / PLCs"]
        UI["Streamlit UI"]
    end

    subgraph AWS["AWS Cloud"]
        ALB["Application Load Balancer"]

        subgraph ECS["ECS Fargate Cluster"]
            T1["Task 1: pump-risk-api\\n2 vCPU / 4 GB"]
            T2["Task 2: pump-risk-api\\n2 vCPU / 4 GB"]
            TN["Task N (auto-scaled)"]
        end

        REDIS["Redis ElastiCache\\n(prediction cache)"]
        PG["PostgreSQL RDS\\n(audit log)"]
        S3["S3 (model artifacts)"]
        CW["CloudWatch\\n(metrics + alarms)"]
    end

    SCADA --> ALB
    UI --> ALB
    ALB --> T1
    ALB --> T2
    ALB --> TN
    T1 --> REDIS
    T2 --> REDIS
    T1 --> PG
    T1 --> S3
    ECS --> CW
        """,
        language="mermaid",
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD TEST RESULTS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Load Test Results</div>', unsafe_allow_html=True)

lt = load_load_test_results()

# Display before/after tables
before_data = [
    {"Level": "Light (5 users)", "Throughput": "395/s", "p50": "10.2 ms", "p95": "18.8 ms", "p99": "30.8 ms"},
    {"Level": "Medium (25 users)", "Throughput": "310/s", "p50": "52.0 ms", "p95": "123.3 ms", "p99": "2,043 ms"},
    {"Level": "Heavy (75 users)", "Throughput": "183/s", "p50": "181.7 ms", "p95": "2,130 ms", "p99": "2,295 ms"},
]

after_data = [
    {"Level": "Light (5 users)", "Throughput": "385/s", "p50": "9.6 ms", "p95": "21.9 ms", "p99": "50.6 ms"},
    {"Level": "Medium (25 users)", "Throughput": "351/s", "p50": "47.2 ms", "p95": "111.9 ms", "p99": "536 ms"},
    {"Level": "Heavy (75 users)", "Throughput": "230/s", "p50": "252.8 ms", "p95": "1,091 ms", "p99": "2,285 ms"},
]

tab_before, tab_after, tab_compare = st.tabs(["Before Optimization", "After Optimization", "Comparison"])

with tab_before:
    st.dataframe(pd.DataFrame(before_data), use_container_width=True, hide_index=True)

with tab_after:
    st.dataframe(pd.DataFrame(after_data), use_container_width=True, hide_index=True)

with tab_compare:
    compare_data = [
        {"Metric": "Medium throughput", "Before": "310/s", "After": "351/s", "Change": "+13%"},
        {"Metric": "Heavy throughput", "Before": "183/s", "After": "230/s", "Change": "+26%"},
        {"Metric": "Medium p99", "Before": "2,043 ms", "After": "536 ms", "Change": "−74%"},
        {"Metric": "Heavy p95", "Before": "2,130 ms", "After": "1,091 ms", "Change": "−49%"},
    ]
    st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)

# ── Charts ──
if HAS_PLOTLY:
    import plotly.graph_objects as go

    col_tp, col_lat = st.columns(2)

    with col_tp:
        fig = go.Figure()
        levels = ["Light (5)", "Medium (25)", "Heavy (75)"]
        fig.add_trace(go.Bar(name="Before", x=levels, y=[395, 310, 183], marker_color="#e74c3c"))
        fig.add_trace(go.Bar(name="After", x=levels, y=[385, 351, 230], marker_color="#2ecc71"))
        fig.update_layout(title="Throughput (req/s)", barmode="group", height=320, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_lat:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Before p95", x=levels, y=[18.8, 123.3, 2130], marker_color="#e74c3c"))
        fig.add_trace(go.Bar(name="After p95", x=levels, y=[21.9, 111.9, 1091], marker_color="#2ecc71"))
        fig.update_layout(title="p95 Latency (ms)", barmode="group", height=320, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION DETAILS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Optimization: What We Changed</div>', unsafe_allow_html=True)

st.markdown(
    """
    | Optimization | Before | After | Impact |
    |:-------------|:-------|:------|:-------|
    | **Pandas → NumPy** (`extract_features`) | 5.70 ms | 0.36 ms | **16× faster** |
    | **Pandas → NumPy** (`compute_anomalies`) | 6.14 ms | 0.29 ms | **21× faster** |
    | **Prediction caching** (MD5 → LRU 2048) | Per-request | Cached | ~3× on repeats |
    | **Response caching** (1024 entries) | None | OrderedDict LRU | Eliminates recomputation |
    | **Pre-computed feature importance** | Per-request SHAP | Startup-once | Major latency drop |
    | **Thread pool** (16 workers) | 4 workers | min(16, cpu+4) | Better parallelism |
    """
)

st.info(
        "**Key takeaway:** The biggest win was removing pandas from the hot path. "
        "Pandas DataFrames have high construction overhead per request. By rewriting "
        "core functions with pure NumPy, we reduced per-request cost from ~12ms to ~1ms."
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# RESOURCE UTILIZATION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Resource Utilization</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        #### Memory Breakdown
        | Component | RAM |
        |:----------|:----|
        | CLIP ViT-B/32 | ~580 MB |
        | Python runtime | ~100 MB |
        | LightGBM models | ~5 MB |
        | Caches | ~20 MB |
        | **Total** | **~730 MB** |
        """
    )

with col2:
    st.markdown(
        """
        #### CPU Profile
        | Phase | CPU % |
        |:------|:------|
        | Startup (model load) | 100% (45s) |
        | Light load | 20–25% |
        | Medium load | 35–45% |
        | Heavy load | 60–80% |
        """
    )

with col3:
    st.markdown(
        """
        #### Scaling Projections
        | Workers | Throughput | RAM |
        |:--------|:----------|:----|
        | 1 | ~385/s | 730 MB |
        | 2 | ~700/s | 1.4 GB |
        | 4 | ~1,200/s | 2.9 GB |
        | 4 × 3 nodes | ~3,500/s | 8.7 GB |
        """
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# OPERATIONAL CONCERNS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Operational Concerns</div>', unsafe_allow_html=True)

st.markdown(
    """
    | Concern | Current State | Production Recommendation |
    |:--------|:-------------|:--------------------------|
    | **Logging** | Structured Python logging to stdout | Ship to CloudWatch / ELK; add request_id tracing |
    | **Monitoring** | Health endpoint + uptime counter | Prometheus metrics: latency histogram, error rate, model drift |
    | **Caching** | In-process OrderedDict LRU | Redis ElastiCache for cross-instance dedup |
    | **Model warmup** | CLIP loads at startup (~45s) | Readiness probe; rolling deployment; pre-warm in CI |
    | **Model versioning** | `model_version` field in response | S3 versioned artifacts + blue/green deploy |
    | **Cost** | ~$0 (local dev) | ~$198/mo on Fargate (2 tasks steady-state) |
    | **Security** | No auth | API Gateway + JWT auth; VPC internal only for SCADA traffic |
    """
)

with st.expander("📄 Full Load & Scale Report"):
    ls_path = PROJECT_ROOT / "load_scale_report.md"
    if ls_path.exists():
        st.markdown(ls_path.read_text(encoding="utf-8"))
    else:
        st.warning("load_scale_report.md not found.")

with st.expander("🔧 How to Run Load Tests"):
    st.code(
        """
# Start the API server first
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Run load test (3 levels: 5/25/75 concurrent users × 20s each)
python scripts/load_test.py          # saves to artifacts/load_test_results_before.json
python scripts/load_test.py --after  # saves to artifacts/load_test_results_after.json
        """,
        language="bash",
    )

st.caption("See load_scale_report.md and optimization_study.md for complete analysis.")
