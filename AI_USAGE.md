# AI Usage Declaration

> **Project:** Pump Fault Risk Prediction Service  
> **Date:** 2026-02-17  

---

## 1 Overview

This project was developed with the assistance of **GitHub Copilot (Claude)** as an AI coding assistant. This document transparently describes what was AI-assisted, what was human-directed, and how correctness was verified at each step.

---

## 2 AI-Assisted Activities

### 2.1 Code Generation

| Component | AI Contribution | Human Contribution | Verification |
|:----------|:---------------|:-------------------|:-------------|
| `src/services/orchestrator.py` | Generated initial structure, `predict()` and `predict_multimodal()` methods, PDF extraction, response caching, explanation generation | Architecture decisions (singleton, hybrid fusion, thread pool sizing), cache eviction strategy | 10 unit/integration tests pass; live endpoint tested with curl; load test at 3 traffic levels |
| `src/models/risk_model.py` | Generated `extract_features()` (both pandas and numpy versions), prediction caching, heuristic fallback, SHAP integration | Feature engineering design (5 stats × 52 sensors), decision to pre-compute importances at load time, cache size tuning | Accuracy report (AUC=1.0); micro-profiling confirmed 16× speedup from numpy rewrite |
| `src/services/preprocessing.py` | Generated `compute_sensor_anomalies()` (both pandas and numpy versions), z-score spike detection, trend calculation | Anomaly thresholds (2σ, 5% relative slope, 30% CV), dot-product slope formula, sensor label mapping | Unit tests; manual inspection of anomaly signals on known fault samples |
| `src/models/fusion.py` | Generated `GatedFusion`, `FusionModule`, adaptive blend formula | Fusion strategy design (hybrid transformer + gated), weight allocation (sensor 0.6, image 0.4) | End-to-end prediction verified; ablation experiments confirm no accuracy regression |
| `src/models/transformer_fusion.py` | Generated `_TransformerCrossModalFusion` nn.Module, modality projectors, risk/confidence heads | Architecture decisions (d_model=256, 2 layers, 4 heads), trained weight loading | Training converges (AUC=1.0 by epoch 5); inference outputs verified against expected ranges |
| `src/models/clip_encoder.py` | Generated CLIP loading, text feature pre-computation, zero-shot fault classification | Fault/normal prompt design (9+3 prompts), similarity threshold (0.22), fault weight assignments | Visual inspection of classification results on sample images; AUC=0.9917 image-only |
| `scripts/train_baseline.py` | Generated training pipeline, Optuna integration, evaluation metrics | Optuna configuration (50 trials, 5-fold CV), choice of LightGBM over alternatives | Cross-validated AUC=1.0; classification report reviewed |
| `scripts/train_joint_multimodal.py` | Generated joint data loading, CLIP embedding extraction, transformer training loop | Data joining strategy (serial_number), label mapping verification, training hyperparameters | Training log reviewed; best AUC saved to checkpoint; test metrics logged |
| `scripts/load_test.py` | Generated full load test runner with threading, psutil monitoring, percentile computation | Traffic level design (5/25/75 users), measurement methodology, payload design | Results validated against manual curl timing; consistent across multiple runs |
| `streamlit_app.py` | Generated UI layout, batch upload handling, NaN sanitization, result display | UX decisions (JSON-only input, explanation display), tab structure | Manual testing through browser; all tabs functional |
| `app/` (multi-page Streamlit) | Generated 6-page presentation app with Client/Dev mode toggle, shared module, plotly charts, custom CSS | Page structure, demo flow, content selection, mode toggle concept | Manual testing; all pages load; predictions work end-to-end |
| `scripts/train.py` | Generated unified training entry-point | Decision to wrap existing scripts | Executes without error; delegates to existing scripts |
| `scripts/infer.py` | Generated CLI inference tool with sample data, stdin, batch modes | CLI design, offline prediction approach | Tested with `--sample normal` and `--sample at-risk` |
| `scripts/evaluate.py` | Generated model evaluation script with metrics | Metric selection, output format | Outputs match training script metrics |
| `scripts/benchmark_latency.py` | Generated latency benchmark with API and offline modes | Benchmark methodology, warmup strategy | Results consistent with load_test.py measurements |
| `src/api/` (routes, schemas) | Generated FastAPI routes, Pydantic schemas, validation | API design (endpoint paths, request/response schema), error handling | OpenAPI docs verified; test_prediction.py and test_health.py pass |

### 2.2 Documentation

| Document | AI Contribution | Human Contribution |
|:---------|:---------------|:-------------------|
| `README.md` | Drafted sections, formatted tables | Requirements, structure decisions, accuracy of technical details |
| `optimization_study.md` | Generated analysis text, formatted tables, Mermaid diagrams | All optimization decisions, measurement data interpretation, deployment architecture choices |
| `evaluation_report.md` | Generated structure, metrics tables, error analysis | Metric interpretation, caveats about perfect scores, recommendations |
| `load_scale_report.md` | Generated analysis, ASCII charts, scaling projections | Test methodology, resource monitoring approach, scaling strategy |
| `DATA_MANIFEST.md` | Generated table structure and descriptions | Dataset identification, source attribution, license verification |
| `ACCURACY_REPORT.md` | Formatted results from training script output | Interpreted training logs, confirmed metric correctness |
| `LOAD_TEST_REPORT.md` | Generated comparison tables and analysis | Profiling methodology, root cause interpretation |
| `docs/ARCHITECTURE.md` | Generated full architecture doc with Mermaid diagrams, transformer flow, extensibility blueprint | Architecture decisions, data flow design, extension interface contract |
| `docs/EVALUATION.md` | Generated evaluation summary | Metric selection, limitation analysis |
| `docs/LOAD_SCALE.md` | Generated load/scale analysis | Scaling strategy, cost estimates |
| `docs/DEMO_SCRIPT.md` | Generated demo walkthrough with talking points | Demo flow, timing, anticipated Q&A |
| `ablation_results.json` | Generated JSON structure | Experiment design, metric values from actual measurements |

### 2.3 Debugging & Optimization

| Issue | AI Contribution | Verification |
|:------|:---------------|:-------------|
| NaN JSON serialization error | Generated `_sanitize_sensor_records()` helper | Error no longer reproduces; `json.dumps()` succeeds on all edge cases |
| Missing `return signals` in `compute_sensor_anomalies()` | Identified and fixed the bug | `TypeError: NoneType is not iterable` resolved; `or []` guard added |
| Pandas hot-path bottleneck | Rewrote both functions to pure numpy | Profiling confirmed 16–21× speedup; all tests still pass |
| Stale load test payloads | Rewrote payloads to match current schema | All requests return 200 OK; no validation errors |

---

## 3 Correctness Verification Methods

### 3.1 Automated Tests

```bash
pytest tests/ -v    # 10 tests, all passing
```

- `test_health.py` — Health endpoint returns 200, correct JSON schema
- `test_prediction.py` — Single and batch prediction endpoints, response validation
- `test_preprocessing.py` — Sensor preprocessing, NaN handling, anomaly detection

### 3.2 Live Endpoint Testing

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"asset_id":"pump_017","timestamp":"2026-02-12T10:30:00Z","sensor_window":[{"sensor_00":2.44,"sensor_01":46.31}]}'
```

Verified: correct response schema, reasonable probability/confidence values, consistent `top_signals`.

### 3.3 Load Testing

Three traffic levels (5/25/75 concurrent users × 20 s each), zero errors across all runs (before and after optimization). Results in `artifacts/load_test_results_*.json`.

### 3.4 Manual Code Review

All AI-generated code was reviewed for:
- Correctness of mathematical formulas (dot-product slope, z-score, CV)
- Edge case handling (empty inputs, NaN values, missing modalities)
- Resource management (cache eviction, thread pool sizing, memory usage)
- Security (no hardcoded credentials, input validation)

### 3.5 Profiling

Per-function micro-benchmarking (1,000 iterations) to verify optimization claims:
- `extract_features()`: 5.70 ms → 0.36 ms (measured, not estimated)
- `compute_sensor_anomalies()`: 6.14 ms → 0.29 ms (measured, not estimated)

---

## 4 What Was NOT AI-Generated

The following decisions and artifacts were entirely human-directed:

1. **Architecture design** — Choice of FastAPI + LightGBM + CLIP + Transformer fusion + GatedFusion hybrid
2. **Dataset selection** — Kaggle pump sensor data + curated pump images
3. **Feature engineering** — 5 statistics × 52 sensors = 260 features; 772-dim joint vector
4. **Model selection** — LightGBM over XGBoost/CatBoost (tree-shap support, speed), CLIP over ResNet (zero-shot capability)
5. **Fusion strategy** — Hybrid (transformer + gated) over pure transformer or pure weighted average
6. **Deployment architecture** — ECS Fargate, ALB, auto-scaling configuration
7. **Threshold selection** — CLIP fault similarity > 0.22, z-score > 2σ, CV > 0.3
8. **Training configuration** — Optuna 50 trials, 5-fold CV, 80/20 stratified split
9. **Load test design** — 3 traffic levels, 20 s duration, warm-up methodology

---

## 5 AI Tool Configuration

| Setting | Value |
|:--------|:------|
| AI Assistant | GitHub Copilot (Claude) |
| Model | Claude Opus 4.6 |
| IDE | VS Code |
| Interaction mode | Chat (multi-turn conversation with tool use) |
| Code execution | Terminal commands run within VS Code |
| File editing | Direct file creation and modification via VS Code |

---

## 6 Ethical Considerations

- No personal or sensitive data was used in training or testing
- All datasets are either public domain (CC0) or project-internal
- AI-generated code was reviewed before committing
- No AI-generated content was presented as purely human work — this document serves as the declaration
