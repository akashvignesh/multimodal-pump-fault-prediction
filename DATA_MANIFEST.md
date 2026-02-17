# Dataset Manifest

This document describes all datasets used in the Pump Fault Risk Service, including source, license, modality, and data nature.

---

## 1. Baseline Sensor Dataset

| Field | Value |
|-------|-------|
| **Path** | `data/baseline_model/sensor_data/sensor.csv` |
| **Modality** | Sensor (time-series) |
| **Records** | 220,320 rows × 55 columns |
| **Schema** | `timestamp`, `sensor_00` … `sensor_51`, `machine_status` |
| **Labels** | NORMAL (205,836), RECOVERING (14,477), BROKEN (7) |
| **Source** | [Kaggle — Pump Sensor Data for Predictive Maintenance](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) |
| **License** | CC0: Public Domain |
| **Data Nature** | Real sensor telemetry from industrial pumps |
| **Key** | Indexed by row order (no explicit `asset_id`; `timestamp` column present) |
| **Notes** | BROKEN class has only 7 samples — collapsed into RECOVERING for binary classification. `sensor_15` contains NaN values (~5% missing). |

---

## 2. Multimodal Sensor Dataset

| Field | Value |
|-------|-------|
| **Path** | `data/multimodal_model/sensor_data.csv` |
| **Modality** | Sensor (time-series) |
| **Records** | 241 rows × 55 columns |
| **Schema** | `serial_number`, `timestamp`, `sensor_00` … `sensor_51`, `machine_status` |
| **Labels** | NORMAL, RECOVERING |
| **Source** | Subset of baseline sensor data, sampled to match image availability |
| **License** | CC0: Public Domain |
| **Data Nature** | Real sensor telemetry, one representative row per pump serial number |
| **Key** | `serial_number` (joins to image mapping) |

---

## 3. Image Dataset

| Field | Value |
|-------|-------|
| **Path** | `data/multimodal_model/images/` |
| **Modality** | Image (visual inspection) |
| **Records** | 241 images (.png, .jpg) |
| **Resolution** | Variable; preprocessed to 224×224 by CLIP processor |
| **Source** | Pump inspection photographs — sourced and curated for this project |
| **License** | Project-internal use |
| **Data Nature** | Real photographs of pump components (normal and corroded states) |
| **Classes** | Normal (healthy pump), Corroded (visible corrosion/degradation) |

---

## 4. Image Mapping

| Field | Value |
|-------|-------|
| **Path** | `data/multimodal_model/image_mapping.csv` |
| **Modality** | Metadata (joins sensor ↔ image) |
| **Records** | 241 rows × 5 columns |
| **Schema** | `serial_number`, `image_location`, `image_type`, `machine_status`, `source_image` |
| **Source** | Generated mapping file linking sensor readings to inspection images |
| **License** | Project-internal |
| **Data Nature** | Curated mapping — `serial_number` joins to sensor data, `image_location` points to image file |
| **Label Mapping** | `Normal` → NORMAL, `Corroded` → RECOVERING |

---

## 5. Pre-computed Artifacts

| Artifact | Description | Source |
|----------|-------------|--------|
| `artifacts/sensor_baseline.pkl` | Trained LightGBM baseline (260-dim sensor features) | `scripts/train_baseline.py` |
| `artifacts/joint_sensor_image.pkl` | Trained LightGBM joint model (772-dim sensor+CLIP features) | `scripts/train_joint_multimodal.py` |
| `artifacts/transformer_fusion_trained.pt` | Trained TransformerCrossModalFusion weights (d=256, 2 layers, 4 heads) | `scripts/train_joint_multimodal.py` |
| `artifacts/clip_image_embeddings.npy` | Pre-computed CLIP embeddings for 241 training images | `scripts/train_joint_multimodal.py` |
| `artifacts/sample_data.json` | Sample sensor windows (normal + recovering) for UI/testing | `extract_samples.py` |
| `artifacts/example_data.json` | Example sensor records for Streamlit demo | `extract_examples.py` |

---

## Modality Summary

| Modality | Available | Model Component | Embedding Dim |
|----------|-----------|-----------------|---------------|
| **Sensor** | ✅ 220K rows | LightGBM baseline | 260 (5 stats × 52 sensors) |
| **Image** | ✅ 241 images | CLIP ViT-B/32 (VLM) | 512 |
| **Text** | ❌ Not available | — | — |
| **Audio** | ❌ Not available | — | — |
| **Video** | ❌ Not available | — | — |

> **Note**: The project focuses on the two modalities (sensor + image) for which real data is available.
> The architecture (TransformerCrossModalFusion) supports additional modality projectors and can be
> extended to text, audio, and video when corresponding datasets become available.
