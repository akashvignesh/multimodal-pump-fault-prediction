"""Sensor-only baseline model for pump fault risk prediction.

Uses LightGBM with aggregated sensor features and SHAP for explainability.
Supports variable-length sensor windows through statistical aggregation.

Optimizations:
- Pre-computed global feature importances from LightGBM gain (avoids per-request SHAP)
- LRU cache on predict() keyed by hashed sensor_window
- SHAP explainer kept as optional fallback (disabled by default for speed)
"""
import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle

import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    shap = None

from src.config import settings

logger = logging.getLogger(__name__)


# Human-readable signal mappings for top contributing features
SIGNAL_MAPPINGS = {
    "sensor_00": "flow_rate_anomaly",
    "sensor_01": "pressure_drop", 
    "sensor_02": "temperature_rise",
    "sensor_03": "vibration_spike",
    "sensor_04": "motor_current_high",
    "sensor_05": "bearing_temp_high",
    "sensor_06": "seal_pressure_low",
    "sensor_07": "discharge_pressure_low",
    "sensor_08": "suction_pressure_low",
    "sensor_09": "rpm_deviation",
    "sensor_10": "power_consumption_high",
    "sensor_11": "acoustic_anomaly",
    "sensor_12": "cavitation_detected",
    "sensor_13": "oil_level_low",
    "sensor_14": "coolant_temp_high",
    "sensor_15": "alignment_deviation",
    "sensor_16": "inlet_temp_high",
    "sensor_17": "outlet_temp_high",
    "sensor_18": "humidity_high",
    "sensor_19": "ambient_temp_high",
    "sensor_20": "load_imbalance",
    "sensor_21": "torque_deviation",
    "sensor_22": "speed_fluctuation",
    "sensor_23": "energy_spike",
    "sensor_24": "noise_level_high",
    "sensor_25": "filter_clogged",
    "sensor_26": "valve_malfunction",
    "sensor_27": "leak_detected",
    "sensor_28": "corrosion_indicator",
    "sensor_29": "wear_detected",
    "sensor_30": "efficiency_drop",
    "sensor_31": "startup_anomaly",
    "sensor_32": "shutdown_anomaly",
    "sensor_33": "cycle_time_deviation",
    "sensor_34": "maintenance_overdue",
    "sensor_35": "calibration_drift",
    "sensor_36": "sensor_degradation",
    "sensor_37": "communication_error",
    "sensor_38": "data_quality_low",
    "sensor_39": "threshold_breach",
    "sensor_40": "trend_anomaly",
    "sensor_41": "pattern_deviation",
    "sensor_42": "baseline_drift",
    "sensor_43": "spike_detected",
    "sensor_44": "dip_detected",
    "sensor_45": "oscillation_detected",
    "sensor_46": "saturation_detected",
    "sensor_47": "range_exceeded",
    "sensor_48": "rate_of_change_high",
    "sensor_49": "cumulative_stress",
    "sensor_50": "thermal_stress",
    "sensor_51": "mechanical_stress",
}


class SensorBaselineModel:
    """Sensor-only baseline model using LightGBM with statistical aggregation.
    
    Features computed per sensor:
    - mean, std, min, max, range
    
    Optimizations:
    - Pre-computed global feature importances (LightGBM gain) at model load
    - Dict-based prediction cache keyed by hashed sensor_window JSON
    """
    
    _CACHE_MAX = 2048  # max cached predictions

    def __init__(self, model_path: Optional[Path] = None):
        self.model: Optional[lgb.Booster] = None
        self.explainer: Optional[Any] = None
        self.feature_names: List[str] = []
        self.num_sensors = settings.num_sensors
        self.model_path = model_path or (settings.artifacts_dir / "sensor_baseline.pkl")
        
        # Pre-computed feature importances (populated at load time)
        self._global_importances: Optional[np.ndarray] = None
        self._global_top_signals: Optional[List[str]] = None
        
        # Prediction cache: hash -> (prob, conf, signals)
        self._cache: Dict[str, Tuple[float, float, List[str]]] = {}
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model from disk if available."""
        if self.model_path and Path(self.model_path).exists():
            try:
                with open(self.model_path, 'rb') as f:
                    saved = pickle.load(f)
                self.model = saved.get('model')
                self.feature_names = saved.get('feature_names', [])
                logger.info(f"Loaded sensor baseline model from {self.model_path}")
                
                # Pre-compute global feature importances from LightGBM gain
                self._precompute_importances()
                
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                self.model = None
        else:
            logger.warning(f"Model file not found at {self.model_path}, using fallback")

    def _precompute_importances(self) -> None:
        """Pre-compute global feature importances from the trained model.
        
        Uses LightGBM's built-in feature_importance(importance_type='gain')
        which is O(1) at inference time (no SHAP needed per request).
        """
        if self.model is None:
            return
        try:
            gains = self.model.feature_importance(importance_type='gain')
            self._global_importances = gains.astype(np.float64)
            
            # Pre-compute the default top-k signals
            k = settings.top_k_signals
            top_indices = np.argsort(np.abs(gains))[-k:][::-1]
            
            signals = []
            for idx in top_indices:
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                    sensor_name = "_".join(feature_name.split("_")[:2])
                    signal = SIGNAL_MAPPINGS.get(sensor_name, feature_name)
                    if signal not in signals:
                        signals.append(signal)
            self._global_top_signals = signals[:k] if signals else ["unknown_signal"]
            logger.info(f"Pre-computed global feature importances (top: {self._global_top_signals})")
            
            # Also initialize SHAP explainer (optional, kept for detailed per-sample analysis)
            if HAS_SHAP:
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                    logger.info("SHAP explainer initialized (optional fallback)")
                except Exception as e:
                    logger.warning(f"SHAP explainer init failed (non-critical): {e}")
        except Exception as e:
            logger.warning(f"Could not pre-compute importances: {e}")
    
    @staticmethod
    def _hash_sensor_window(sensor_window: List[Dict[str, Any]]) -> str:
        """Compute a fast hash of sensor_window for caching."""
        raw = json.dumps(sensor_window, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()
    
    def extract_features(self, sensor_window: List[Dict[str, Any]]) -> np.ndarray:
        """Extract aggregated features from variable-length sensor window.

        Pure-numpy implementation (no pandas) for speed.

        Args:
            sensor_window: List of dicts with sensor readings

        Returns:
            1D numpy array of aggregated features (num_sensors * 5)
        """
        n_stats = 5  # mean, std, min, max, range
        if not sensor_window:
            return np.zeros(self.num_sensors * n_stats, dtype=np.float32)

        # Build column-index lookup once from the first row
        col_keys = {k for row in sensor_window for k in row if k.startswith("sensor_")}

        features = np.zeros(self.num_sensors * n_stats, dtype=np.float64)
        self.feature_names = []

        for i in range(self.num_sensors):
            col = f"sensor_{i:02d}"
            offset = i * n_stats
            self.feature_names.extend(
                [f"{col}_mean", f"{col}_std", f"{col}_min", f"{col}_max", f"{col}_range"]
            )
            if col not in col_keys:
                continue
            # Collect numeric values, coercing non-numeric to NaN
            vals = []
            for row in sensor_window:
                v = row.get(col)
                if v is None:
                    continue
                try:
                    fv = float(v)
                    if np.isfinite(fv):
                        vals.append(fv)
                except (TypeError, ValueError):
                    pass
            if not vals:
                continue
            arr = np.array(vals, dtype=np.float64)
            mean_val = arr.mean()
            std_val = arr.std(ddof=0) if len(arr) > 1 else 0.0
            min_val = arr.min()
            max_val = arr.max()
            features[offset] = mean_val
            features[offset + 1] = std_val
            features[offset + 2] = min_val
            features[offset + 3] = max_val
            features[offset + 4] = max_val - min_val

        return features.astype(np.float32)
    
    def predict(self, sensor_window: List[Dict[str, Any]]) -> Tuple[float, float, List[str]]:
        """Predict failure probability from sensor window.
        
        Uses dict-based LRU cache to avoid redundant computation on
        identical sensor windows.
        
        Args:
            sensor_window: List of sensor reading dicts
            
        Returns:
            Tuple of (failure_probability, confidence, top_signals)
        """
        # --- cache lookup ---
        cache_key = self._hash_sensor_window(sensor_window)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        
        features = self.extract_features(sensor_window)
        
        if self.model is None:
            result = self._heuristic_predict(features)
            self._put_cache(cache_key, result)
            return result
        
        # Reshape for prediction
        X = features.reshape(1, -1)
        
        # Get probability prediction
        prob = self.model.predict(X)[0]
        
        # Ensure probability is in [0, 1]
        prob = float(np.clip(prob, 0.0, 1.0))
        
        # Compute confidence based on prediction certainty
        confidence = float(abs(2 * prob - 1) * 0.8 + 0.2)  # Range [0.2, 1.0]
        
        # Get top signals – use pre-computed global importances (fast path)
        # Falls back to per-sample SHAP when global importances unavailable
        top_signals = self._get_top_signals(X, prob)
        
        result = (prob, confidence, top_signals)
        self._put_cache(cache_key, result)
        return result
    
    def _put_cache(self, key: str, value: Tuple[float, float, List[str]]) -> None:
        """Insert into cache, evicting oldest entries if full."""
        if len(self._cache) >= self._CACHE_MAX:
            # Evict ~25% oldest (dict is insertion-ordered in Python 3.7+)
            evict_count = self._CACHE_MAX // 4
            keys = list(self._cache.keys())[:evict_count]
            for k in keys:
                del self._cache[k]
        self._cache[key] = value
    
    def _heuristic_predict(self, features: np.ndarray) -> Tuple[float, float, List[str]]:
        """Fallback heuristic prediction when model is not available.
        
        Uses simple thresholding on aggregated statistics:
        - High variance/range indicates instability
        - Extreme mean values indicate anomalies
        """
        if len(features) == 0 or np.all(features == 0):
            return 0.5, 0.3, ["insufficient_data"]
        
        # Compute anomaly score based on deviations
        # Higher std and range values indicate more anomalies
        std_features = features[1::5]  # Every 5th starting at index 1
        range_features = features[4::5]  # Every 5th starting at index 4
        
        # Normalize and compute score
        std_score = np.mean(std_features[std_features > 0]) if np.any(std_features > 0) else 0
        range_score = np.mean(range_features[range_features > 0]) if np.any(range_features > 0) else 0
        
        # Simple heuristic: normalize to [0, 1]
        combined_score = (std_score + range_score) / 200  # Rough normalization
        prob = float(np.clip(combined_score, 0.0, 1.0))
        
        # Low confidence for heuristic prediction
        confidence = 0.4
        
        # Get signals based on high variance features
        top_signals = self._get_heuristic_signals(features)
        
        return prob, confidence, top_signals
    
    def _get_top_signals(self, X: np.ndarray, prob: float, k: int = 5) -> List[str]:
        """Get top contributing signals.
        
        Fast path: uses pre-computed LightGBM gain importances (O(1)).
        Slow path (fallback): per-sample SHAP values.
        
        Args:
            X: Feature array (1, n_features)
            prob: Predicted probability
            k: Number of top signals to return
            
        Returns:
            List of human-readable signal names
        """
        # ---- Fast path: pre-computed global importances ----
        if self._global_top_signals is not None:
            return list(self._global_top_signals)  # return a copy
        
        # ---- Slow path: per-sample SHAP (only if global importances unavailable) ----
        if self.explainer is None or not HAS_SHAP:
            return self._get_heuristic_signals(X.flatten())
        
        try:
            shap_values = self.explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                values = shap_values
            
            values = values.flatten()
            top_indices = np.argsort(np.abs(values))[-k:][::-1]
            
            signals = []
            for idx in top_indices:
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                    sensor_name = "_".join(feature_name.split("_")[:2])
                    signal = SIGNAL_MAPPINGS.get(sensor_name, feature_name)
                    if signal not in signals:
                        signals.append(signal)
            
            return signals[:k] if signals else ["unknown_signal"]
            
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return self._get_heuristic_signals(X.flatten())
    
    def _get_heuristic_signals(self, features: np.ndarray, k: int = 5) -> List[str]:
        """Get top signals based on feature importance heuristics.
        
        Uses variance and range features to identify anomalous sensors.
        """
        signals = []
        
        # Check std features (every 5th starting at index 1)
        for i in range(self.num_sensors):
            std_idx = i * 5 + 1
            range_idx = i * 5 + 4
            
            if std_idx < len(features) and range_idx < len(features):
                std_val = features[std_idx]
                range_val = features[range_idx]
                
                # High variance indicates anomaly
                if std_val > 50 or range_val > 100:  # Thresholds based on data
                    sensor_name = f"sensor_{i:02d}"
                    signal = SIGNAL_MAPPINGS.get(sensor_name, f"anomaly_{sensor_name}")
                    if signal not in signals:
                        signals.append(signal)
        
        # If no anomalies detected, return generic signals
        if not signals:
            signals = ["normal_operation"]
        
        return signals[:k]


class JointSensorImageModel:
    """Joint sensor+image model using LightGBM on 772-dim features.

    Input: 260-dim sensor features (5 stats × 52 sensors) concatenated with
    512-dim CLIP ViT-B/32 image embeddings.

    The model is trained by ``scripts/train_joint_multimodal.py`` which joins
    sensor data and images on ``serial_number`` (NORMAL→Normal, RECOVERING→Corroded).

    When no joint artifact is available, all calls return ``None`` so the
    orchestrator seamlessly falls back to the sensor-only baseline.
    """

    _CACHE_MAX = 2048

    def __init__(self, model_path: Optional[Path] = None):
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.model_path = model_path or (settings.artifacts_dir / "joint_sensor_image.pkl")
        self._global_top_signals: Optional[List[str]] = None
        self._cache: Dict[str, Tuple[float, float, List[str]]] = {}
        self._load_model()

    @property
    def available(self) -> bool:
        return self.model is not None

    def _load_model(self) -> None:
        if not Path(self.model_path).exists():
            logger.info(
                "Joint sensor+image model not found at %s – "
                "will use sensor-only baseline. "
                "Run: python scripts/train_joint_multimodal.py",
                self.model_path,
            )
            return
        try:
            with open(self.model_path, "rb") as f:
                saved = pickle.load(f)
            self.model = saved.get("model")
            self.feature_names = saved.get("feature_names", [])
            logger.info(
                "Loaded joint sensor+image model from %s (%d features)",
                self.model_path,
                len(self.feature_names),
            )
            self._precompute_importances()
        except Exception as e:
            logger.warning("Failed to load joint sensor+image model: %s", e)

    def _precompute_importances(self) -> None:
        if self.model is None:
            return
        try:
            gains = self.model.feature_importance(importance_type="gain")
            k = settings.top_k_signals
            top_idx = np.argsort(np.abs(gains))[-k:][::-1]
            signals: List[str] = []
            for idx in top_idx:
                if idx < len(self.feature_names):
                    fname = self.feature_names[idx]
                    # sensor features → human-readable, clip features → keep as-is
                    if fname.startswith("sensor_"):
                        sensor_name = "_".join(fname.split("_")[:2])
                        sig = SIGNAL_MAPPINGS.get(sensor_name, fname)
                    elif fname.startswith("clip_"):
                        sig = "image_feature"
                    else:
                        sig = fname
                    if sig not in signals:
                        signals.append(sig)
            self._global_top_signals = signals[:k] if signals else ["unknown_signal"]
            logger.info("Joint model top signals: %s", self._global_top_signals)
        except Exception as e:
            logger.warning("Joint model precompute_importances failed: %s", e)

    def predict(
        self,
        sensor_features: np.ndarray,
        image_embedding: np.ndarray,
    ) -> Optional[Tuple[float, float, List[str]]]:
        """Predict failure probability from concatenated sensor+image features.

        Parameters
        ----------
        sensor_features : np.ndarray
            260-dim sensor feature vector (from SensorBaselineModel.extract_features).
        image_embedding : np.ndarray
            512-dim CLIP ViT-B/32 embedding (from ImageEncoder / CLIPImageEncoder).

        Returns
        -------
        ``(failure_probability, confidence, top_signals)`` or ``None`` if
        the model is unavailable.
        """
        if self.model is None:
            return None

        # Cache key from both inputs
        cache_key = hashlib.md5(
            sensor_features.tobytes() + image_embedding.tobytes()
        ).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Concatenate → 772-dim
        joint = np.concatenate([
            sensor_features.flatten(),
            image_embedding.flatten(),
        ]).reshape(1, -1)

        prob = float(self.model.predict(joint)[0])
        prob = float(np.clip(prob, 0.0, 1.0))
        confidence = float(abs(2 * prob - 1) * 0.85 + 0.15)  # slightly higher base

        top_signals = list(self._global_top_signals) if self._global_top_signals else ["unknown"]

        result = (prob, confidence, top_signals)
        self._put_cache(cache_key, result)
        return result

    def _put_cache(self, key: str, value: Tuple[float, float, List[str]]) -> None:
        if len(self._cache) >= self._CACHE_MAX:
            for k in list(self._cache.keys())[: self._CACHE_MAX // 4]:
                del self._cache[k]
        self._cache[key] = value