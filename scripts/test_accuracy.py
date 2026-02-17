"""Accuracy evaluation for baseline and multimodal sensor models."""
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add parent dir to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.config import settings
from src.models.risk_model import BaselineFullModel, SensorBaselineModel

print("=" * 80)
print("MODEL ACCURACY EVALUATION")
print("=" * 80)


# ---------------------------------------------------------------------------
# Test 1: BaselineFullModel (220K-row trained model)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("1. BASELINE FULL MODEL (v2.0.0-baseline)")
print("   Training Data: data/baseline_model/sensor_data/sensor.csv (220,320 rows)")
print("=" * 80)

baseline_full_model = BaselineFullModel()

# Load data
baseline_csv = settings.data_dir / "baseline_model" / "sensor_data" / "sensor.csv"
print(f"\nLoading data from: {baseline_csv}")

if not baseline_csv.exists():
    print(f"ERROR: File not found: {baseline_csv}")
else:
    try:
        df_baseline = pd.read_csv(baseline_csv, low_memory=False)
        print(f"Loaded {len(df_baseline)} rows × {len(df_baseline.columns)} cols")
        
        # Extract labels and sensor data
        y_true = df_baseline["machine_status"].values
        
        # Map labels to indices
        label_map = {"NORMAL": 0, "RECOVERING": 1, "BROKEN": 2}
        if isinstance(y_true[0], str):
            y_true_idx = np.array([label_map.get(l, 1) for l in y_true])
        else:
            y_true_idx = y_true
        
        print(f"Class distribution:\n{pd.Series(y_true_idx).value_counts().sort_index()}")
        
        # Predict on first 1000 samples for speed
        sample_size = min(1000, len(df_baseline))
        sample_idx = np.random.choice(len(df_baseline), sample_size, replace=False)
        
        predictions = []
        for idx in sample_idx:
            row = df_baseline.iloc[idx]
            sensor_dict = {
                f"sensor_{i:02d}": row.get(f"sensor_{i:02d}", 0) 
                for i in range(52)
            }
            result = baseline_full_model.predict([sensor_dict])
            predicted_label = result["label"]
            # Convert to index
            idx_map = {"NORMAL": 0, "RECOVERING": 1, "BROKEN": 2}
            predictions.append(idx_map.get(predicted_label, 0))
        
        y_true_sample = y_true_idx[sample_idx]
        y_pred = np.array(predictions)
        
        acc = accuracy_score(y_true_sample, y_pred)
        prec = precision_score(y_true_sample, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true_sample, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true_sample, y_pred, average="weighted", zero_division=0)
        
        print(f"\n  Evaluation on {sample_size} samples:")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1-score:  {f1:.4f}")
        
        print(f"\n  Confusion Matrix:")
        cm = confusion_matrix(y_true_sample, y_pred)
        print(cm)
        
        print(f"\n  Classification Report:")
        print(classification_report(y_true_sample, y_pred, 
                                   target_names=["NORMAL", "RECOVERING", "BROKEN"],
                                   labels=[0, 1, 2],
                                   zero_division=0))
    except Exception as e:
        print(f"ERROR: {e}")


# ---------------------------------------------------------------------------
# Test 2: SensorBaselineModel (241-row multimodal dataset)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("2. SENSOR BASELINE MODEL (multimodal, 241 samples)")
print("   Training Data: data/multimodal_model/sensor_data.csv (241 rows)")
print("=" * 80)

sensor_baseline_model = SensorBaselineModel()

# Load multimodal sensor data
multimodal_csv = settings.multimodal_data_dir / "sensor_data.csv"
print(f"\nLoading data from: {multimodal_csv}")

if not multimodal_csv.exists():
    print(f"ERROR: File not found: {multimodal_csv}")
else:
    try:
        df_mm = pd.read_csv(multimodal_csv)
        print(f"Loaded {len(df_mm)} rows × {len(df_mm.columns)} cols")
        
        # The multimodal dataset already has 'machine_status' column, no need to merge
        if "machine_status" in df_mm.columns:
            y_true = df_mm["machine_status"].values
            
            # Map string labels to numeric
            unique_labels = sorted(set(y_true[pd.notna(y_true)]))
            label_to_idx = {l: i for i, l in enumerate(unique_labels)}
            y_true_idx = np.array([label_to_idx.get(l, -1) if pd.notna(l) else -1 for l in y_true])
            
            # Filter out missing labels
            valid_idx = y_true_idx >= 0
            y_true_idx = y_true_idx[valid_idx]
            df_mm_valid = df_mm[valid_idx].reset_index(drop=True)
            
            print(f"Valid samples with labels: {len(y_true_idx)}")
            print(f"Unique labels: {unique_labels}")
            print(f"Class distribution:\n{pd.Series(y_true_idx).value_counts().sort_index()}")
            
            # Predict on all samples
            predictions = []
            for idx in range(len(df_mm_valid)):
                row = df_mm_valid.iloc[idx]
                sensor_dict = {
                    f"sensor_{i:02d}": row.get(f"sensor_{i:02d}", 0) 
                    for i in range(52)
                }
                prob, conf, signals = sensor_baseline_model.predict([sensor_dict])
                # High probability = fault (class 1), low = normal (class 0)
                # Threshold at 0.5
                pred_class = 1 if prob > 0.5 else 0
                predictions.append(pred_class)
            
            y_pred = np.array(predictions)
            
            # For binary classification
            if len(np.unique(y_true_idx)) == 2:
                acc = accuracy_score(y_true_idx, y_pred)
                prec = precision_score(y_true_idx, y_pred, zero_division=0)
                rec = recall_score(y_true_idx, y_pred, zero_division=0)
                f1 = f1_score(y_true_idx, y_pred, zero_division=0)
                try:
                    auc = roc_auc_score(y_true_idx, y_pred)
                except:
                    auc = None
                
                print(f"\n  Evaluation on {len(y_true_idx)} samples:")
                print(f"    Accuracy:  {acc:.4f}")
                print(f"    Precision: {prec:.4f}")
                print(f"    Recall:    {rec:.4f}")
                print(f"    F1-score:  {f1:.4f}")
                if auc:
                    print(f"    AUC:       {auc:.4f}")
            else:
                acc = accuracy_score(y_true_idx, y_pred)
                prec = precision_score(y_true_idx, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_true_idx, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_true_idx, y_pred, average="weighted", zero_division=0)
                
                print(f"\n  Evaluation on {len(y_true_idx)} samples:")
                print(f"    Accuracy:  {acc:.4f}")
                print(f"    Precision: {prec:.4f}")
                print(f"    Recall:    {rec:.4f}")
                print(f"    F1-score:  {f1:.4f}")
            
            print(f"\n  Confusion Matrix:")
            cm = confusion_matrix(y_true_idx, y_pred)
            print(cm)
            
            print(f"\n  Classification Report:")
            print(classification_report(y_true_idx, y_pred, zero_division=0))
        else:
            print("ERROR: 'machine_status' column not found in sensor data")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Baseline Full Model:
  - 220K training samples
  - Classes: NORMAL, RECOVERING, BROKEN
  - Reported during training: Accuracy=0.9992, AUC=1.0, F1=0.9974
  - Test run: evaluated on random 1000 samples

Sensor Baseline Model (multimodal):
  - 241 training samples  
  - Binary or multiclass based on data
  - Reported during training: AUC=1.0 (perfect fit on small dataset)
  - Test run: evaluated on all available samples
""")
print("=" * 80)
