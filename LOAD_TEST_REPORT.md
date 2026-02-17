# Load Test Report – Pump Fault Risk Service

**Date:** 2026-02-17  
**Endpoint:** `POST /predict` (sensor-only, 3⨉11 sensor window)  
**Server:** Uvicorn single-worker, Python 3.14.2  
**Machine:** Windows, local  

---

## 1 Traffic Levels & Results

### Before Optimisation

| Level  | Users | Requests | Errors | Throughput | p50 (ms) | p95 (ms)  | p99 (ms)  | Avg CPU | RAM (MB) |
|--------|------:|:--------:|:------:|:----------:|:--------:|:---------:|:---------:|:-------:|:--------:|
| Light  |     5 |    7 910 |      0 |  395.2 /s  |    10.2  |     18.8  |     30.8  |   13 %  |    728   |
| Medium |    25 |    6 226 |      0 |  310.2 /s  |    52.0  |    123.3  |  2 043.1  |    7 %  |    729   |
| Heavy  |    75 |    3 759 |      0 |  183.2 /s  |   181.7  |  2 130.1  |  2 294.9  |    2 %  |    737   |

### After Optimisation

| Level  | Users | Requests | Errors | Throughput | p50 (ms) | p95 (ms)  | p99 (ms)  | Avg CPU | RAM (MB) |
|--------|------:|:--------:|:------:|:----------:|:--------:|:---------:|:---------:|:-------:|:--------:|
| Light  |     5 |    7 695 |      0 |  384.6 /s  |     9.6  |     21.9  |     50.6  |   12 %  |    726   |
| Medium |    25 |    7 040 |      0 |  351.0 /s  |    47.2  |    111.9  |    536.0  |    6 %  |    728   |
| Heavy  |    75 |    4 683 |      0 |  230.2 /s  |   252.8  |  1 090.8  |  2 285.4  |    6 %  |    736   |

---

## 2 Improvement Summary

| Metric               | Before  | After   | Change          |
|:---------------------|:--------|:--------|:----------------|
| **Medium throughput** |  310 /s |  351 /s | **+13 %**       |
| **Heavy throughput**  |  183 /s |  230 /s | **+26 %**       |
| **Medium p95**        | 123 ms  | 112 ms  | **−9 %**        |
| **Heavy p95**         | 2 130 ms| 1 091 ms| **−49 %**       |
| **Medium p99**        | 2 043 ms|  536 ms | **−74 %**       |
| **Heavy requests**    | 3 759   | 4 683   | **+25 %** more served |

---

## 3 Bottleneck Analysis

Micro-profiling of the per-request hot path (sensor-only predict, single-threaded, 1 000 iterations):

| Component                 | Before (pandas) | After (numpy) | Speedup |
|:--------------------------|:---------------:|:-------------:|:-------:|
| `compute_sensor_anomalies()` |     6.14 ms     |     0.29 ms   | **21×** |
| `extract_features()`         |     5.70 ms     |     0.36 ms   | **16×** |
| `model.predict()` (LightGBM) |     0.28 ms     |     0.28 ms   |   1×    |
| Cache key (MD5)              |     0.03 ms     |     0.03 ms   |   1×    |
| `_generate_explanation()`    |     0.004 ms    |     0.004 ms  |   1×    |
| **Total hot-path**           |  **~12.1 ms**   |  **~0.95 ms** | **13×** |

**Root cause:** Both `extract_features()` and `compute_sensor_anomalies()` created a
`pandas.DataFrame` on every request, then used `pd.to_numeric()` and `np.polyfit()`
per sensor column. For a 3-row × 11-column sensor window, pandas object construction
dominated compute time — accounting for **~98 %** of the per-request CPU cost.

CPU utilisation stayed below 15 % even at 75 users, confirming the bottleneck was
**per-request compute overhead** (making each request slow), not overall CPU saturation.
At high concurrency, slow individual requests cause queueing in the single async event
loop, which is why p95/p99 latencies exploded.

---

## 4 Optimisations Implemented

### 4.1 Pure-NumPy `extract_features()` (risk_model.py)
Replaced `pd.DataFrame(sensor_window)` → `pd.to_numeric()` per-column loop with a
direct dict-iteration + `np.array()` per sensor.  Eliminates DataFrame construction
and pandas type coercion overhead.  **5.70 ms → 0.36 ms (16×).**

### 4.2 Pure-NumPy `compute_sensor_anomalies()` (preprocessing.py)
Replaced pandas-based anomaly detection with:
- Direct dict-iteration instead of `pd.DataFrame`
- Dot-product slope calculation instead of `np.polyfit` (avoids lstsq overhead)
- Removed `pd.to_numeric()` calls

**6.14 ms → 0.29 ms (21×).**

### 4.3 Expanded Thread Pool (orchestrator.py)
Increased `ThreadPoolExecutor(max_workers=4)` → `min(16, os.cpu_count() + 4)`.
This allows more requests to be processed concurrently when the event loop dispatches
blocking work (sensor model inference, image encoding) to the thread pool.

### 4.4 Removed Unused Pandas Imports
Removed `import pandas as pd` from `risk_model.py` and `preprocessing.py`, reducing
module load time and memory footprint.

---

## 5 Resource Usage

| Metric     | Light | Medium | Heavy | Notes                            |
|:-----------|:-----:|:------:|:-----:|:---------------------------------|
| Avg CPU %  |  12   |   6    |   6   | Event-loop bound, not CPU-bound  |
| Peak CPU % |  33   |  20    |  14   | Low — single worker              |
| RAM (MB)   | 726   | 728    | 736   | Stable; dominated by CLIP model  |

RAM is constant at ~730 MB, dominated by the CLIP ViT-B/32 model weights loaded
at startup. No memory leaks observed across all traffic levels.

---

## 6 Remaining Bottleneck

The single Uvicorn worker + Python GIL remains the primary scaling constraint.
At 75 users, the event loop serialises HTTP parsing and response writes, causing
tail-latency growth even with sub-millisecond compute. Further improvements would
require **multiple workers** (`--workers N`) or an async-native inference path
(e.g., ONNX Runtime with async batching).
