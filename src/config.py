"""Application configuration settings."""
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    artifacts_dir: Path = base_dir / "artifacts"
    multimodal_data_dir: Path = data_dir / "multimodal_model"
    
    # Model settings
    model_version: str = "v1.0.0"
    sensor_model_path: Optional[str] = None
    image_model_path: Optional[str] = None
    
    # API settings
    api_version: str = "v1"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model hyperparameters
    sensor_window_size: int = 10  # Expected window size for aggregation
    num_sensors: int = 52  # sensor_00 to sensor_51
    top_k_signals: int = 5  # Number of top signals to return
    
    # Feature names for explainability mapping
    sensor_feature_names: list = [f"sensor_{i:02d}" for i in range(52)]
    
    # Human-readable signal mappings
    signal_mappings: dict = {
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
    }
    
    # Inference settings
    max_batch_size: int = 100
    inference_timeout_s: float = 30.0
    
    # Caching
    enable_embedding_cache: bool = True
    cache_max_size: int = 1000

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()