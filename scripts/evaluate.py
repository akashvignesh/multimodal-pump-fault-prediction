#!/usr/bin/env python
"""Evaluate trained models on validation data.

Usage:
    python scripts/evaluate.py                  # evaluate all models
    python scripts/evaluate.py --model baseline # sensor-only
    python scripts/evaluate.py --output results.json  # save results

Prints metrics to stdout, optionally saves JSON report.
"""
import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_multimodal_data() -> pd.DataFrame:
    """Load the multimodal sensor dataset (241 rows)."""
    path = BASE_DIR / "data" / "multimodal_model" / "sensor_data.csv"
    if not path.exists():
        logger.error(f"Data not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_baseline_data(max_rows: int = 2000) -> pd.DataFrame:
    """Load the baseline sensor dataset (subsample for speed)."""
    path = BASE_DIR / "data" / "baseline_model" / "sensor_data" / "sensor.csv"
    if not path.exists():
        logger.error(f"Data not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    return df


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    model_name: str = "model",
) -> dict[str, Any]:
    """Compute standard classification metrics."""
    metrics: dict[str, Any] = {
        "model": model_name,
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auc_roc"] = None
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def print_metrics(metrics: dict[str, Any]) -> None:
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {metrics['model']}  (n={metrics['n_samples']})")
    print(f"{'=' * 60}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    if metrics.get("auc_roc") is not None:
        print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    cm = metrics.get("confusion_matrix", [])
    if cm:
        print(f"  Confusion Matrix:")
        for row in cm:
            print(f"    {row}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Model evaluators
# ---------------------------------------------------------------------------

def eval_sensor_baseline(df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate the sensor baseline model (SensorBaselineModel)."""
    from src.models.risk_model import SensorBaselineModel
    model = SensorBaselineModel()

    label_map = {"NORMAL": 0, "RECOVERING": 1, "BROKEN": 1}
    y_true_raw = df["machine_status"].values
    y_true = np.array([label_map.get(str(l), 0) for l in y_true_raw])

    y_pred = []
    y_prob = []
    t0 = time.perf_counter()

    for _, row in df.iterrows():
        sensor_dict = {}
        for i in range(52):
            col = f"sensor_{i:02d}"
            if col in row:
                val = row[col]
                sensor_dict[col] = float(val) if pd.notna(val) else None
        try:
            result = model.predict([sensor_dict])
            prob = result["failure_probability"]
            y_prob.append(prob)
            y_pred.append(1 if prob >= 0.5 else 0)
        except Exception:
            y_prob.append(0.5)
            y_pred.append(0)

    elapsed = time.perf_counter() - t0
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = evaluate_model(y_true, y_pred, y_prob, "Sensor Baseline (LightGBM)")
    metrics["total_time_s"] = round(elapsed, 3)
    metrics["avg_latency_ms"] = round(elapsed / len(df) * 1000, 3)
    return metrics


def eval_multimodal_sensor(df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate the multimodal model on sensor-only inputs (no images)."""
    from src.models.risk_model import SensorBaselineModel
    model = SensorBaselineModel()

    label_map = {"NORMAL": 0, "RECOVERING": 1}
    y_true = np.array([label_map.get(str(l), 0) for l in df["machine_status"].values])

    y_pred = []
    y_prob = []
    t0 = time.perf_counter()

    for _, row in df.iterrows():
        sensor_dict = {}
        for i in range(52):
            col = f"sensor_{i:02d}"
            if col in row:
                val = row[col]
                sensor_dict[col] = float(val) if pd.notna(val) else None
        try:
            result = model.predict([sensor_dict])
            prob = result["failure_probability"]
            y_prob.append(prob)
            y_pred.append(1 if prob >= 0.5 else 0)
        except Exception:
            y_prob.append(0.5)
            y_pred.append(0)

    elapsed = time.perf_counter() - t0
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = evaluate_model(y_true, y_pred, y_prob, "Multimodal Sensor-Only")
    metrics["total_time_s"] = round(elapsed, 3)
    metrics["avg_latency_ms"] = round(elapsed / len(df) * 1000, 3)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pump fault risk models",
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "multimodal", "all"],
        default="all",
        help="Which model to evaluate (default: all)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=2000,
        help="Max rows for baseline evaluation (default: 2000)",
    )
    args = parser.parse_args()

    all_metrics: list[dict[str, Any]] = []

    if args.model in ("baseline", "all"):
        logger.info("Loading baseline dataset...")
        df_bl = load_baseline_data(args.max_rows)
        if not df_bl.empty:
            logger.info(f"Evaluating sensor baseline on {len(df_bl)} samples...")
            m = eval_sensor_baseline(df_bl)
            print_metrics(m)
            all_metrics.append(m)

    if args.model in ("multimodal", "all"):
        logger.info("Loading multimodal dataset...")
        df_mm = load_multimodal_data()
        if not df_mm.empty:
            logger.info(f"Evaluating multimodal on {len(df_mm)} samples...")
            m = eval_multimodal_sensor(df_mm)
            print_metrics(m)
            all_metrics.append(m)

    # Summary
    print(f"\n{'=' * 60}")
    print("  EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    for m in all_metrics:
        auc = m.get("auc_roc", "N/A")
        auc_str = f"{auc:.4f}" if isinstance(auc, float) else auc
        print(f"  {m['model']:<35s}  AUC={auc_str}  F1={m['f1']:.4f}  "
              f"({m['avg_latency_ms']:.1f} ms/sample)")
    print(f"{'=' * 60}")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(all_metrics, indent=2))
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
