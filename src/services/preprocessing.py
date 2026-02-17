"""Preprocessing utilities for sensor and multimodal data."""
import logging
from typing import Dict, List, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def preprocess_sensor_window(
    sensor_window: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Preprocess sensor window data.
    
    - Handles missing values via forward-fill / zero-fill
    - Converts string values to float where possible
    - Removes all non-numeric keys
    
    Args:
        sensor_window: Raw sensor readings from request
        
    Returns:
        Cleaned sensor window
    """
    if not sensor_window:
        return sensor_window
    
    cleaned = []
    for reading in sensor_window:
        clean_reading = {}
        for key, val in reading.items():
            # Only process numeric sensor keys (skip metadata like timestamp, serial_number, etc.)
            if not key.startswith('sensor_'):
                continue
            
            if val is None:
                clean_reading[key] = 0.0
            elif isinstance(val, (int, float)):
                clean_reading[key] = float(val) if np.isfinite(val) else 0.0
            else:
                try:
                    clean_reading[key] = float(val)
                except (ValueError, TypeError):
                    clean_reading[key] = 0.0
        
        cleaned.append(clean_reading)
    
    return cleaned


def normalize_sensor_values(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> np.ndarray:
    """Normalize sensor features using z-score normalization.
    
    Args:
        features: Feature array
        mean: Optional pre-computed mean
        std: Optional pre-computed std
        
    Returns:
        Normalized features
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    
    return (features - mean) / std


def validate_sensor_data(sensor_window: List[Dict[str, Any]]) -> bool:
    """Validate that sensor window contains usable data.
    
    Args:
        sensor_window: Sensor readings to validate
        
    Returns:
        True if data is valid and usable
    """
    if not sensor_window:
        return False
    
    for reading in sensor_window:
        has_numeric = False
        for key, val in reading.items():
            if val is not None:
                try:
                    float(val)
                    has_numeric = True
                except (ValueError, TypeError):
                    pass
        if has_numeric:
            return True
    
    return False


# ------------------------------------------------------------------
# Derived sensor analytics (trend, rolling stats, anomaly indicators)
# ------------------------------------------------------------------

_SENSOR_LABELS = {
    "sensor_00": "flow_rate", "sensor_04": "motor_power", "sensor_06": "vibration",
    "sensor_07": "temperature_a", "sensor_08": "temperature_b", "sensor_09": "temperature_c",
    "sensor_10": "pressure_a", "sensor_11": "pressure_b",
}


def compute_sensor_anomalies(sensor_window: List[Dict[str, Any]]) -> List[str]:
    """Detect anomaly indicators from a sensor window.

    Pure-numpy implementation (no pandas) for speed.

    Computes per sensor column:
    - z-score spike: last reading > 2 sigma from window mean
    - trend (slope): rising/falling trend above threshold
    - high variance: coefficient of variation above threshold

    Returns list of human-readable anomaly signal strings.
    """
    if not sensor_window or len(sensor_window) < 2:
        return []

    # Collect sensor column names across all rows
    sensor_cols = sorted(
        {k for row in sensor_window for k in row if k.startswith("sensor_")}
    )
    if not sensor_cols:
        return []

    n_rows = len(sensor_window)
    signals: List[str] = []

    for col in sensor_cols:
        # Extract numeric values, skipping None / non-numeric
        vals: List[float] = []
        for row in sensor_window:
            v = row.get(col)
            if v is None:
                continue
            try:
                fv = float(v)
                if np.isfinite(fv):
                    vals.append(fv)
            except (TypeError, ValueError):
                continue
        if len(vals) < 2:
            continue

        arr = np.array(vals, dtype=np.float64)
        mean = arr.mean()
        std = arr.std(ddof=0)
        label = _SENSOR_LABELS.get(col, col)

        # 1. Z-score spike on last reading
        if std > 0:
            if abs(arr[-1] - mean) / std > 2.0:
                signals.append(f"{label}_zscore_spike")

        # 2. Trend – simple linear slope via dot-product (avoids np.polyfit overhead)
        n = len(arr)
        if n >= 3 and std > 0:
            x = np.arange(n, dtype=np.float64)
            x_mean = (n - 1) / 2.0
            slope = np.dot(x - x_mean, arr - mean) / np.dot(x - x_mean, x - x_mean)
            rel_slope = abs(slope) / (abs(mean) + 1e-9)
            if rel_slope > 0.05:
                direction = "rising" if slope > 0 else "falling"
                signals.append(f"{label}_trend_{direction}")

        # 3. High variance (coefficient of variation)
        if abs(mean) > 1e-6 and std / abs(mean) > 0.3:
            signals.append(f"{label}_high_variance")

    return signals