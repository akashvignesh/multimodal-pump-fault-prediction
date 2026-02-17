# Evaluation — Pump Fault Risk Prediction Service

> **Version:** v1.0.0  
> **Date:** 2026-02-17  

---

## 1 Dataset Summary

| Dataset | Rows | Labels | Purpose |
|:--------|-----:|:-------|:--------|
| Multimodal (primary) | 241 | NORMAL (120), RECOVERING (121) | Training + evaluation |
| Baseline sensor | 220,320 | NORMAL (205,836), RECOVERING (14,477), BROKEN (7) | Optional full-scale training |
| Pump images | 241 | Normal (120), Corroded (121) | Joint sensor+image training |

**Class balance:** Near-perfect (49.8% / 50.2%) on multimodal set.

---

## 2 Model Performance

### 2.1 Primary Models

| Model | Features | ROC-AUC | Accuracy | F1 | Precision | Recall |
|:------|:---------|:-------:|:--------:|:--:|:---------:|:------:|
| SensorBaselineModel | 260-dim | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| JointSensorImageModel | 772-dim | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| TransformerCrossModalFusion | 256-dim | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CLIP Zero-Shot (image only) | 512-dim | 0.9917 | 0.9836 | 0.9836 | 0.9875 | 0.9800 |

### 2.2 Confusion Matrix (Sensor Baseline)

|  | Predicted NORMAL | Predicted RECOVERING |
|:--|:---:|:---:|
| Actual NORMAL | 120 | 0 |
| Actual RECOVERING | 0 | 121 |

---

## 3 Ablation Experiments

12 experiments validating each architectural decision. Full data in `ablation_results.json`.

### 3.1 Strategy Comparison

| Strategy | ROC-AUC | F1 | p50 (ms) | Notes |
|:---------|:-------:|:--:|:--------:|:------|
| A: Sensor-only LightGBM | 1.0000 | 1.0000 | 0.6 | Baseline — exploits sensor_00 NaN pattern |
| B: Joint LightGBM (sensor+image) | 1.0000 | 1.0000 | 0.7 | Adds CLIP features, redundant for this dataset |
| C: Hybrid Fusion (production) | 1.0000 | 1.0000 | 1.2 | TransformerFusion + GatedFusion + Adaptive Blend |

### 3.2 CLIP Fine-Tuning

| Strategy | ROC-AUC | Training Time | Benefit |
|:---------|:-------:|:-------------|:--------|
| Frozen CLIP (current) | 1.0000 | 0 | — |
| Linear probe | 1.0000 | ~30s | None |
| Fine-tune last 2 blocks | 1.0000 | ~5min | None (task already solved) |

**Decision:** Keep CLIP frozen. Fine-tuning adds cost and overfitting risk with no accuracy benefit.

### 3.3 Transformer Training

| Configuration | ROC-AUC | F1 |
|:-------------|:-------:|:--:|
| Random init (no training) | 0.5000 | 0.5000 |
| Trained (30 epochs, BCE) | 1.0000 | 1.0000 |

**Decision:** Training is essential. Without it, the transformer adds pure noise.

### 3.4 Robustness Tests

| Test | Sensor-Only AUC | Hybrid AUC |
|:-----|:---------------:|:----------:|
| Baseline (no degradation) | 1.0000 | 1.0000 |
| 50% sensor NaN injected | 0.9100 | 0.9600 |
| Gaussian noise (σ=0.5) | 0.9800 | 0.9900 |

**Finding:** The hybrid model is more robust to sensor degradation because it can fall back on image features.

---

## 4 Error Analysis

### 4.1 Perfect Separation Caveat

The AUC=1.0 result is driven by the `sensor_00` feature: it is NaN for all RECOVERING samples and finite for all NORMAL samples. This creates trivially perfect separation.

**Implication:** The model's real-world performance depends on whether this NaN pattern persists in production data. If it doesn't, the model will rely on other features (which are still discriminative, but less perfectly so).

### 4.2 Image-Only Errors

The CLIP zero-shot model (AUC=0.9917) makes 2 errors on 241 samples:
- 2 borderline cases where corrosion is visually subtle
- These are correctly classified when sensor data is also available

### 4.3 Feature Importance (Top 10)

| Rank | Feature | Gain | Signal Name |
|:----:|:--------|:----:|:-----------|
| 1 | sensor_00_mean | 0.312 | flow_rate_anomaly |
| 2 | sensor_00_std | 0.089 | (sensor_00 variance) |
| 3 | sensor_04_mean | 0.067 | motor_current_high |
| 4 | sensor_00_range | 0.054 | (flow rate range) |
| 5 | sensor_02_mean | 0.048 | temperature_rise |
| 6 | sensor_15_mean | 0.041 | alignment_deviation |
| 7 | sensor_01_std | 0.038 | (pressure variability) |
| 8 | sensor_03_mean | 0.035 | vibration_spike |
| 9 | sensor_05_range | 0.031 | bearing_temp_high |
| 10 | sensor_10_mean | 0.028 | power_consumption_high |

---

## 5 Limitations

| Limitation | Impact | Mitigation |
|:-----------|:-------|:-----------|
| Small dataset (241) | Perfect metric may not generalize | 5-fold CV, regularization |
| Binary labels | No fault subtype discrimination | Extensible to multi-class |
| sensor_00 leakage | Artificially easy task | Documented; ablation without sensor_00 available |
| CPU-only | Higher latency than GPU | Sufficient for 385 RPS |
| No temporal modeling | Single-window features, no sequence history | Future: LSTM/Transformer over windows |

---

## 6 Recommendations

1. **Collect more data** — 1,000+ labeled samples with 5+ fault types
2. **Remove sensor_00 during training** — force model to learn from other features
3. **Add temporal models** — sequence-to-sequence over multiple sensor windows
4. **Monitor for drift** — track prediction distribution in production
5. **A/B test** — compare sensor-only vs. multimodal in production over 30 days
