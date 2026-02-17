# Load & Scale — Pump Fault Risk Prediction Service

> **Version:** v1.0.0  
> **Date:** 2026-02-17  

---

## 1 Test Methodology

| Parameter | Value |
|:----------|:------|
| Server | Uvicorn, single worker, CPU-only |
| Endpoint | `POST /predict` (sensor-only, 3 readings × 11 sensors) |
| Traffic levels | Light (5), Medium (25), Heavy (75) concurrent users |
| Duration | 20 seconds per level |
| Tool | Custom `scripts/load_test.py` (httpx + ThreadPoolExecutor + psutil) |
| Warm-up | 3 requests before each level |

---

## 2 Results

### Before Optimization

| Level | Users | Total Reqs | Throughput | p50 | p95 | p99 |
|:------|:-----:|:----------:|:----------:|:---:|:---:|:---:|
| Light | 5 | 7,910 | 395/s | 10.2 ms | 18.8 ms | 30.8 ms |
| Medium | 25 | 6,226 | 310/s | 52.0 ms | 123.3 ms | 2,043 ms |
| Heavy | 75 | 3,759 | 183/s | 181.7 ms | 2,130 ms | 2,295 ms |

### After Optimization

| Level | Users | Total Reqs | Throughput | p50 | p95 | p99 |
|:------|:-----:|:----------:|:----------:|:---:|:---:|:---:|
| Light | 5 | 7,695 | 385/s | 9.6 ms | 21.9 ms | 50.6 ms |
| Medium | 25 | 7,040 | 351/s | 47.2 ms | 111.9 ms | 536 ms |
| Heavy | 75 | 4,683 | 230/s | 252.8 ms | 1,091 ms | 2,285 ms |

---

## 3 Improvement Summary

| Metric | Before | After | Change |
|:-------|:------:|:-----:|:------:|
| Medium throughput | 310/s | 351/s | **+13%** |
| Heavy throughput | 183/s | 230/s | **+26%** |
| Medium p99 | 2,043 ms | 536 ms | **−74%** |
| Heavy p95 | 2,130 ms | 1,091 ms | **−49%** |

---

## 4 What We Optimized

### 4.1 Pandas → NumPy (Core Bottleneck)

The hot-path functions `extract_features()` and `compute_sensor_anomalies()` were originally built on pandas DataFrames. Profiling revealed that DataFrame construction dominated per-request latency:

| Function | Before | After | Speedup |
|:---------|:------:|:-----:|:-------:|
| `extract_features()` | 5.70 ms | 0.36 ms | **16×** |
| `compute_sensor_anomalies()` | 6.14 ms | 0.29 ms | **21×** |
| **Total hot-path** | **12.10 ms** | **0.95 ms** | **13×** |

### 4.2 Prediction Caching

- MD5 hash of sensor features → LRU cache (2,048 entries)
- Response-level cache for identical requests (1,024 entries, 25% eviction)
- ~3× speedup for repeated queries

### 4.3 Pre-Computed Feature Importance

Replaced per-request SHAP TreeExplainer with startup-time `feature_importance(importance_type='gain')`.

### 4.4 Thread Pool Expansion

`ThreadPoolExecutor(workers=min(16, os.cpu_count() + 4))` — up from 4 fixed workers.

---

## 5 Resource Utilization

| Resource | Value |
|:---------|:------|
| RAM (steady-state) | ~730 MB |
| — CLIP ViT-B/32 | ~580 MB |
| — Python runtime | ~100 MB |
| — LightGBM + caches | ~50 MB |
| CPU (light load) | 20–25% |
| CPU (heavy load) | 60–80% |
| Startup time | ~45 s (CLIP download/load) |

---

## 6 Scaling Strategy

### Vertical (Single Machine)

| Workers | Est. Throughput | RAM Required |
|:--------|:---------------|:-------------|
| 1 | ~385/s | 730 MB |
| 2 | ~700/s | 1.4 GB |
| 4 | ~1,200/s | 2.9 GB |

### Horizontal (Multi-Node)

| Setup | Est. Throughput | Monthly Cost (Fargate) |
|:------|:---------------|:----------------------|
| 2 tasks × 2 vCPU | ~700/s | ~$198 |
| 4 tasks × 2 vCPU | ~1,400/s | ~$396 |
| 10 tasks × 2 vCPU | ~3,500/s | ~$990 |

Auto-scaling policy: target CPU 70%, min 2, max 10 tasks.

---

## 7 Recommendations

1. **Add Redis cache** — share prediction cache across instances
2. **Preload CLIP in CI** — reduce cold-start time
3. **GPU for image-heavy workloads** — CLIP on GPU is ~10× faster
4. **WebSocket for streaming** — push predictions to SCADA dashboards
5. **Rate limiting** — protect against burst traffic (token bucket at ALB)

---

## 8 Reproduction

```bash
# Start API server
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Run load test
python scripts/load_test.py           # saves artifacts/load_test_results_before.json
python scripts/load_test.py --after   # saves artifacts/load_test_results_after.json
```
