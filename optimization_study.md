# Optimization Study – Pump Fault Risk Service

> **Author:** Engineering team  
> **Date:** 2026-02-17  
> **Service version:** v1.0.0  

---

## Table of Contents

1. [Fine-Tuning Feasibility](#1-fine-tuning-feasibility)
2. [Structured Reasoning & Ensemble Strategies](#2-structured-reasoning--ensemble-strategies)
3. [Deployment Architecture](#3-deployment-architecture)
4. [Database & State Connections](#4-database--state-connections)
5. [Latency Bottleneck Mitigations](#5-latency-bottleneck-mitigations)

---

## 1 Fine-Tuning Feasibility

### 1.1 Context

The service uses two pre-trained foundation models:

| Model | Parameters | Role | Currently |
|-------|-----------|------|-----------|
| **CLIP ViT-B/32** | ~151 M | Image encoder (512-dim embeddings) | Frozen — used as feature extractor |
| **LightGBM** (sensor baseline) | ~260 leaves | Sensor-only classifier (260-dim) | Fully trained from scratch |
| **LightGBM** (joint) | ~772 leaves | Joint sensor+image classifier (772-dim) | Fully trained from scratch |
| **TransformerCrossModalFusion** | ~530 K | Cross-modal attention (256-dim, 2 layers, 4 heads) | End-to-end trained on 241 paired samples |

### 1.2 Experiment: Fine-Tuning CLIP on Pump Images

**Hypothesis:** Fine-tuning the last few layers of CLIP on our domain-specific pump images (Normal vs Corroded) could improve image-feature quality and downstream fusion accuracy.

**Setup:**
- Dataset: 241 paired images (120 Normal, 121 Corroded) from `data/multimodal_model/images/`
- Strategy: Linear probing (freeze all CLIP layers, train a classification head on CLIP embeddings) vs. full fine-tuning of the last 2 transformer blocks
- Evaluation: 5-fold stratified CV, metric = ROC-AUC

**Results (from `ablation_results.json`, experiment `clip_finetune`):**

| Strategy | ROC-AUC (5-fold CV) | Training Time | Embedding Drift |
|----------|:-------------------:|:-------------:|:---------------:|
| Frozen CLIP (current) | **1.0000** | 0 (inference only) | None |
| Linear probe on CLIP | 1.0000 | ~30 s | None |
| Fine-tune last 2 blocks | 1.0000 | ~5 min | Moderate |

**Analysis:**
The frozen CLIP embeddings already achieve perfect separation on this binary task (Normal vs Corroded images are visually distinct — clean metal vs oxidized/rusted surfaces). Fine-tuning yields **no measurable benefit** because the task is already solved by the zero-shot visual features.

**Recommendation:** **Do NOT fine-tune CLIP** for the current task and dataset.

- **Cost:** Fine-tuning adds ~600 MB of optimizer state to GPU memory, requires careful learning-rate scheduling to avoid catastrophic forgetting, and complicates the deployment (must version and ship fine-tuned weights instead of using the public checkpoint).
- **Risk:** With only 241 images, fine-tuning the 151 M-parameter vision transformer risks severe overfitting, even with aggressive regularization. The current approach (frozen encoder + trained fusion head) is safer.
- **When to revisit:** Fine-tuning becomes worthwhile when:
  1. The dataset grows to ≥1,000 labelled images with a richer label taxonomy (e.g., 5+ fault types — cavitation, seal failure, bearing wear, not just "corroded")
  2. Domain shift is detected (e.g., CLIP embeddings no longer separate new pump image types)
  3. The classification task gets harder (AUC drops below ~0.95)

### 1.3 Experiment: TransformerCrossModalFusion Training

The TransformerCrossModalFusion **was trained** end-to-end on 241 paired sensor+image samples (see `scripts/train_joint_multimodal.py`). This is a genuine fine-tuning success story:

| Configuration | ROC-AUC | F1 | Notes |
|:-------------|:-------:|:--:|:------|
| Random init (no training) | 0.50 | 0.50 | Coin flip — meaningless output |
| Trained (30 epochs, BCE loss) | **1.0000** | **1.0000** | Converges by epoch 5 |

The trained weights are saved to `artifacts/transformer_fusion_trained.pt` and auto-loaded at startup. This shows that **fine-tuning the fusion head is both feasible and essential** — without training, the transformer's cross-modal attention adds only noise.

---

## 2 Structured Reasoning & Ensemble Strategies

### 2.1 Current Architecture (Hybrid Fusion)

The production system uses a **hybrid ensemble** of two complementary fusion strategies:

```
Sensor Model ──────────┬──▶ GatedFusion ────────────────┐
                       │                                 │
Image Encoder ─────────┤                                 ├──▶ Adaptive Blend ──▶ Final Prediction
                       │                                 │
Joint Model (LightGBM) ┤                                 │
                       │                                 │
TransformerFusion ─────┴──▶ Cross-Modal Attention ──────┘
```

**Adaptive blend formula:**
```
transformer_weight = min(0.10 + n_modalities × 0.125, 0.40)
final_prob = (1 - w) × gated_prob + w × transformer_prob
```

### 2.2 Ablation: Strategy Comparison

We compared three ensemble/fusion strategies on the 241-sample dataset using 5-fold stratified CV:

| Strategy | ROC-AUC | F1 | p50 Latency | Complexity |
|:---------|:-------:|:--:|:-----------:|:----------:|
| **A. Sensor-only LightGBM** (260-dim) | 1.0000 | 1.0000 | 0.6 ms | Low |
| **B. Joint LightGBM** (772-dim, sensor+CLIP) | 1.0000 | 1.0000 | 0.7 ms | Low |
| **C. Hybrid Fusion** (current production) | 1.0000 | 1.0000 | 1.2 ms | Medium |

**Interpretation:**
All three strategies achieve perfect AUC on the current dataset. This makes it impossible to discriminate between them on accuracy alone. Instead, we evaluate on secondary criteria:

| Criterion | A (Sensor-only) | B (Joint LightGBM) | C (Hybrid) |
|:----------|:---------------:|:-------------------:|:----------:|
| Handles image-only input | ❌ | ❌ | ✅ |
| Cross-modal reasoning | ❌ | Implicit (concatenation) | ✅ (attention) |
| Graceful degradation | Sensor required | Both required | Any subset works |
| Latency overhead | Baseline | +0.1 ms | +0.6 ms |
| Extensibility (new modalities) | Low | Moderate | High (add projector) |

**Recommendation:** Keep **Strategy C (Hybrid Fusion)** as production default because:
1. It is the **only strategy that handles arbitrary modality subsets** — sensor-only, image-only, or both.
2. The transformer fusion provides a natural extension point for future modalities (text logs, audio, vibration spectra).
3. The latency overhead (+0.6 ms) is negligible vs the network + serialization cost in a real deployment.

**When to switch to simpler strategies:**
- If latency budget is extremely tight (<1 ms total) and only sensor data is available, Strategy A is optimal.
- If the dataset grows and accuracy between strategies diverges, ablation should be re-run.

### 2.3 Ablation: Component-Level Contribution

Disabling each component individually to measure marginal contribution:

| Configuration | ROC-AUC | Δ AUC | Notes |
|:-------------|:-------:|:-----:|:------|
| Full system (C) | 1.0000 | — | Baseline |
| Disable TransformerFusion (gated only) | 1.0000 | 0.0 | Gated fusion alone is sufficient |
| Disable JointModel (no joint LightGBM) | 1.0000 | 0.0 | Sensor baseline + fusion is enough |
| Disable image encoder (sensor only) | 1.0000 | 0.0 | Sensor features dominate |
| Disable sensor model (image only) | 0.9917 | −0.0083 | Image alone slightly worse |
| Disable caching | 1.0000 | 0.0 | No accuracy impact; latency impact only |

**Key insight:** On the current 241-sample dataset with perfect sensor NaN-pattern separation, the sensor modality alone is sufficient for classification. The image modality becomes critical only when sensor data is degraded or unavailable (image-only AUC = 0.9917 shows CLIP's zero-shot visual features are strong but not perfect).

### 2.4 Error Analysis

With perfect AUC, there are no classification errors to analyze on the training/test set. However, we identified **robustness edge cases** via synthetic perturbation:

| Perturbation | Sensor-only AUC | Hybrid AUC | Delta |
|:------------|:---------------:|:----------:|:-----:|
| None (baseline) | 1.0000 | 1.0000 | 0.0 |
| 50% sensor values set to NaN | 0.9916 | 0.9958 | +0.0042 |
| Gaussian noise σ=0.5 on sensors | 0.9833 | 0.9875 | +0.0042 |
| Swap 10% of image labels | 1.0000 | 0.9960 | −0.0040 |

The hybrid system shows slight resilience to sensor degradation (the image modality compensates), but slight vulnerability to mislabelled images. This validates the higher sensor weight (0.6) in the GatedFusion default configuration.

---

## 3 Deployment Architecture

### 3.1 Recommended Production Topology

```mermaid
flowchart TB
    subgraph Internet
        Client["Client / Monitoring System"]
    end

    subgraph AWS["AWS Region (us-east-1)"]
        ALB["Application Load Balancer\n(TLS termination, health checks)"]

        subgraph ECS["ECS Cluster (Fargate)"]
            subgraph APIService["API Service (auto-scaled)"]
                API1["Container 1\nuvicorn × 2 workers\n2 vCPU · 4 GB RAM"]
                API2["Container 2\nuvicorn × 2 workers\n2 vCPU · 4 GB RAM"]
                APIn["Container N\n(auto-scale 2–10)"]
            end

            subgraph UIService["Streamlit UI"]
                UI1["Container\nstreamlit\n0.5 vCPU · 1 GB"]
            end
        end

        ECR["ECR\n(Container Registry)"]
        S3["S3\n(Model Artifacts\n+ Training Data)"]
        CW["CloudWatch\n(Metrics + Logs)"]
        RDS["RDS PostgreSQL\n(Optional: prediction\nlog / audit trail)"]
        ElastiCache["ElastiCache Redis\n(Optional: shared\nprediction cache)"]
    end

    Client -->|HTTPS| ALB
    ALB -->|/predict, /predict/batch| APIService
    ALB -->|:8501| UIService
    APIService -->|pull images| ECR
    APIService -->|read artifacts| S3
    APIService -->|metrics, logs| CW
    APIService -.->|optional| RDS
    APIService -.->|optional| ElastiCache
    UIService -->|HTTP| APIService
```

### 3.2 Scaling Strategy

| Dimension | Approach | Justification |
|:----------|:---------|:-------------|
| **Horizontal** | ECS auto-scaling (2–10 containers), target: 70% CPU | The Python GIL limits single-process concurrency; horizontal scaling is the primary lever |
| **Vertical** | 2 vCPU / 4 GB per container | CLIP model consumes ~730 MB RAM; 4 GB gives comfortable headroom. 2 vCPU allows 2 uvicorn workers per container |
| **Workers** | 2 uvicorn workers per container | 2 workers × 10 containers = 20 parallel inference processes max |
| **Warm-up** | Pre-load CLIP model in container ENTRYPOINT; readiness probe on `/health` | CLIP model load takes ~45 s; ALB shouldn't route traffic until ready |

### 3.3 Cost Estimate (AWS Fargate, us-east-1)

| Resource | Spec | Monthly Est. |
|:---------|:-----|:-------------|
| 2× API containers (steady-state) | 2 vCPU / 4 GB × 720 h | ~$145 |
| 1× UI container | 0.5 vCPU / 1 GB × 720 h | ~$18 |
| ALB | 1 LCU average | ~$25 |
| S3 (artifacts, ~200 MB) | Standard | ~$0.01 |
| ECR (container image, ~2 GB) | Standard | ~$0.20 |
| CloudWatch (logs + metrics) | — | ~$10 |
| **Total (steady-state)** | | **~$198/mo** |

Burst scaling to 10 containers would cost ~$725/mo if sustained.

### 3.4 CI/CD Pipeline

```
GitHub Push → GitHub Actions → Build Docker Image → Push to ECR →
                                  ↓
                            Run pytest (10 tests)
                                  ↓
                            Run load_test.py (regression gate: p95 < 200 ms)
                                  ↓
                            ECS Rolling Deployment (blue/green)
```

---

## 4 Database & State Connections

### 4.1 Current State

The service is **stateless by design** — all models are loaded from disk artifacts at startup and held in memory. No database is required for core inference.

### 4.2 When to Add a Database

| Use Case | Database | Schema | Justification |
|:---------|:---------|:-------|:-------------|
| **Prediction audit log** | PostgreSQL (RDS) | `(id, asset_id, timestamp, fp, fc, signals, inference_ms, model_version)` | Regulatory compliance, model monitoring, detecting data drift |
| **Shared prediction cache** | Redis (ElastiCache) | Key: MD5(sensor_window), Value: PredictionResponse JSON, TTL: 300 s | When multiple API containers need cache coherence; current in-process OrderedDict cache doesn't share across containers |
| **Model registry** | S3 + DynamoDB | `(model_name, version, s3_path, metrics_json, deployed_at)` | Track model lineage across versions; support A/B testing and rollback |
| **Sensor time-series** | TimescaleDB or InfluxDB | `(asset_id, timestamp, sensor_00..51)` | If the service needs to maintain sliding windows itself rather than receiving them in the request |

### 4.3 Connection Architecture (if PostgreSQL added)

```python
# src/db.py (example — not implemented)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@rds-host:5432/pumpdb",
    pool_size=10,
    max_overflow=5,
    pool_recycle=3600,
)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
```

**Key considerations:**
- Use **async drivers** (`asyncpg`) to avoid blocking the FastAPI event loop
- Pool size should match uvicorn worker count (2 workers × 10 pool = 20 connections max per container)
- Write prediction logs **asynchronously** (fire-and-forget) to avoid adding latency to the inference path
- Use connection health checks and automatic reconnection to handle RDS failovers

### 4.4 Redis Cache Architecture (if deployed multi-container)

```python
# src/cache.py (example — not implemented)
import redis.asyncio as redis

pool = redis.ConnectionPool.from_url(
    "redis://elasticache-host:6379/0",
    max_connections=20,
    decode_responses=True,
)

async def get_cached(key: str) -> Optional[dict]:
    r = redis.Redis(connection_pool=pool)
    raw = await r.get(f"predict:{key}")
    return json.loads(raw) if raw else None

async def set_cached(key: str, response: dict, ttl: int = 300):
    r = redis.Redis(connection_pool=pool)
    await r.setex(f"predict:{key}", ttl, json.dumps(response))
```

**Decision:** For the current single-container deployment, the in-process `OrderedDict` cache is optimal (zero latency, zero network overhead). Redis should be introduced only when the system scales beyond a single container and cache hit rates are measured to justify the added infrastructure.

---

## 5 Latency Bottleneck Mitigations

### 5.1 Identified Bottlenecks

We profiled the per-request hot path (sensor-only predict, single-threaded, 1,000 iterations) before and after optimization:

| Component | Before (pandas) | After (numpy) | Speedup | % of Hot Path (Before) |
|:----------|:---------------:|:-------------:|:-------:|:----------------------:|
| `compute_sensor_anomalies()` | 6.14 ms | 0.29 ms | **21×** | 50.7% |
| `extract_features()` | 5.70 ms | 0.36 ms | **16×** | 47.1% |
| `model.predict()` (LightGBM) | 0.28 ms | 0.28 ms | 1× | 2.3% |
| Cache key (MD5) | 0.03 ms | 0.03 ms | 1× | <1% |
| `_generate_explanation()` | 0.004 ms | 0.004 ms | 1× | <1% |
| **Total hot-path** | **~12.1 ms** | **~0.95 ms** | **13×** | |

### 5.2 Root Cause

Both `extract_features()` and `compute_sensor_anomalies()` previously created a `pandas.DataFrame` on every request, then used `pd.to_numeric()` and `np.polyfit()` per sensor column. For a 3-row × 11-column sensor window:

- `pd.DataFrame(sensor_window)` construction: ~3 ms (type inference, index creation, column alignment)
- `pd.to_numeric()` per column: ~0.5 ms × 11 columns = ~5.5 ms
- `np.polyfit()` per column: ~0.2 ms × 11 columns = ~2.2 ms

**98% of per-request CPU cost was pandas overhead**, not actual computation.

CPU utilisation stayed below 15% even at 75 concurrent users, confirming the bottleneck was **per-request compute overhead** (making each request slow), not overall CPU saturation. At high concurrency, slow individual requests cause queueing in the single async event loop, which is why p95/p99 latencies exploded from ~100 ms to ~2,100 ms.

### 5.3 Optimizations Implemented

#### 5.3.1 Pure-NumPy `extract_features()` (risk_model.py)

**Before:**
```python
df = pd.DataFrame(sensor_window)
for col in sensor_cols:
    series = pd.to_numeric(df[col], errors='coerce')
    features.extend([series.mean(), series.std(), ...])
```

**After:**
```python
for i in range(self.num_sensors):
    col = f"sensor_{i:02d}"
    vals = [float(row[col]) for row in sensor_window if col in row and row[col] is not None]
    arr = np.array(vals, dtype=np.float64)
    features[offset] = arr.mean()
    features[offset + 1] = arr.std(ddof=0)
    ...
```

**Impact:** 5.70 ms → 0.36 ms (16× faster). Eliminates DataFrame construction, type inference, and column alignment overhead entirely.

#### 5.3.2 Pure-NumPy `compute_sensor_anomalies()` (preprocessing.py)

**Before:**
```python
df = pd.DataFrame(sensor_window)
for col in sensor_cols:
    vals = pd.to_numeric(df[col], errors='coerce').dropna()
    slope = np.polyfit(range(len(vals)), vals, 1)[0]  # least squares
```

**After:**
```python
for col in sensor_cols:
    vals = [float(row[col]) for row in sensor_window ...]
    arr = np.array(vals, dtype=np.float64)
    # Dot-product slope (no matrix solve):
    x = np.arange(n, dtype=np.float64)
    slope = np.dot(x - x_mean, arr - mean) / np.dot(x - x_mean, x - x_mean)
```

**Impact:** 6.14 ms → 0.29 ms (21× faster). Replaces `np.polyfit` (which calls `np.linalg.lstsq` internally — O(n²) matrix operations) with a closed-form dot-product slope formula that is O(n).

#### 5.3.3 Expanded Thread Pool (orchestrator.py)

**Before:** `ThreadPoolExecutor(max_workers=4)`
**After:** `ThreadPoolExecutor(max_workers=min(16, os.cpu_count() + 4))`

Allows more concurrent blocking operations (sensor model inference, image encoding) to proceed in parallel when the event loop dispatches them.

### 5.4 Load Test Results (Before vs After)

| Level | Metric | Before | After | Δ |
|:------|:-------|:------:|:-----:|:--|
| Light (5 users) | Throughput | 395 /s | 385 /s | −3% (within noise) |
| Light | p50 | 10.2 ms | 9.6 ms | −6% |
| **Medium (25 users)** | **Throughput** | **310 /s** | **351 /s** | **+13%** |
| Medium | p95 | 123 ms | 112 ms | −9% |
| Medium | p99 | 2,043 ms | 536 ms | **−74%** |
| **Heavy (75 users)** | **Throughput** | **183 /s** | **230 /s** | **+26%** |
| Heavy | p95 | 2,130 ms | 1,091 ms | **−49%** |
| Heavy | p99 | 2,295 ms | 2,285 ms | −0.4% |

### 5.5 Further Optimization Opportunities (Not Yet Implemented)

| Optimization | Expected Impact | Effort | Trade-off |
|:------------|:---------------|:------:|:----------|
| **Multiple uvicorn workers** (`--workers 4`) | 2–4× throughput at heavy load | Low | 4× memory (~3 GB total for CLIP) |
| **ONNX Runtime for LightGBM** | ~2× faster model.predict() (0.28→0.14 ms) | Medium | Additional dependency, model export step |
| **ONNX Runtime for CLIP** | ~3× faster image encoding | Medium | Export complexity, potential accuracy delta |
| **Batch inference** (accumulate requests) | Higher GPU utilization | Medium | Adds latency for first request in batch |
| **Response streaming** | Lower TTFB | Low | Client must handle streaming |
| **Pre-computed sensor baselines per asset** | Skip redundant feature extraction | Low | Requires asset registry |

### 5.6 Remaining Scaling Constraint

The **single Uvicorn worker + Python GIL** remains the primary scaling constraint. At 75 concurrent users, the event loop serializes HTTP parsing and response writes, causing tail-latency growth even with sub-millisecond compute. Solutions:

1. **Multiple workers** (`--workers N`): Immediate 2–4× improvement. Each worker has its own GIL and event loop.
2. **Async-native inference**: Replace blocking LightGBM predict with an async ONNX Runtime session.
3. **Horizontal scaling**: Run multiple containers behind a load balancer (see Section 3).

---

## Appendix A: Reproduction Commands

```bash
# 1. Train all models
python scripts/train_baseline.py
python scripts/train_joint_multimodal.py --epochs 30

# 2. Start the server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1

# 3. Run load tests
python scripts/load_test.py              # saves to artifacts/load_test_results_before.json
python scripts/load_test.py --after      # saves to artifacts/load_test_results_after.json

# 4. Run micro-benchmark (per-function profiling)
python scripts/benchmark_latency.py      # per-function timing (1000 iterations)

# 5. Run ablation experiments
python scripts/run_ablations.py          # generates ablation_results.json
```

## Appendix B: Hardware

All benchmarks were run on a local Windows machine:
- CPU: Multi-core (exact model varies by dev machine)
- RAM: 16+ GB
- GPU: None (CPU-only inference)
- Python: 3.14.2
- OS: Windows

Network latency is excluded from latency measurements (client and server on localhost).
