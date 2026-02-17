"""Train sensor-only baseline model for pump fault risk prediction.

Uses LightGBM with statistical aggregation features and Optuna
hyperparameter optimization for improved accuracy.

Label mapping:
- NORMAL -> 0 (low failure risk)
- RECOVERING -> 1 (high failure risk, actively recovering from fault)

Features computed per sensor:
- mean, std, min, max, range (5 features per sensor)
- 52 sensors -> 260 features total
"""
import logging
import pickle
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MULTIMODAL_DATA = DATA_DIR / "multimodal_model" / "sensor_data.csv"


def load_data() -> pd.DataFrame:
    """Load sensor data from CSV.
    
    Returns:
        DataFrame with sensor readings and machine_status labels
    """
    logger.info(f"Loading data from {MULTIMODAL_DATA}")
    df = pd.read_csv(MULTIMODAL_DATA)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Label distribution:\n{df['machine_status'].value_counts()}")
    return df


def extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract aggregated features from sensor data.
    
    For training, each row is treated as a single reading.
    In production, multiple readings form a window.
    
    Args:
        df: DataFrame with sensor columns
        
    Returns:
        X: feature array (n_samples, n_features)
        y: label array (n_samples,)
        feature_names: list of feature names
    """
    logger.info("Extracting features...")
    
    # Get sensor columns
    sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
    logger.info(f"Found {len(sensor_cols)} sensor columns")
    
    # For single-row samples, we compute stats per sample
    # In a real scenario with window data, we'd aggregate
    X_list = []
    feature_names = []
    
    for i, row in df.iterrows():
        features = []
        
        for col in sensor_cols:
            val = row[col]
            # For single reading, use value as mean, 0 for std, same for min/max/range
            if pd.isna(val):
                mean_val = std_val = min_val = max_val = range_val = 0.0
            else:
                mean_val = float(val)
                std_val = 0.0  # Single value has no variance
                min_val = float(val)
                max_val = float(val)
                range_val = 0.0
            
            features.extend([mean_val, std_val, min_val, max_val, range_val])
            
            if i == 0:  # Build feature names on first iteration
                feature_names.extend([
                    f"{col}_mean", f"{col}_std",
                    f"{col}_min", f"{col}_max", f"{col}_range"
                ])
        
        X_list.append(features)
    
    X = np.array(X_list, dtype=np.float32)
    
    # Handle NaN/inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Labels: NORMAL=0, RECOVERING=1
    label_map = {"NORMAL": 0, "RECOVERING": 1}
    y = df['machine_status'].map(label_map).values
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Label distribution: {np.bincount(y)}")
    
    return X, y, feature_names


def _optuna_objective(trial, X_train, y_train):
    """Optuna objective for LightGBM hyperparameter tuning."""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'verbose': -1,
        'seed': 42,
    }

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        train_ds = lgb.Dataset(X_tr, label=y_tr)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

        model = lgb.train(
            params,
            train_ds,
            num_boost_round=500,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        y_prob = model.predict(X_val)
        auc = roc_auc_score(y_val, y_prob)
        auc_scores.append(auc)

    return np.mean(auc_scores)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_optuna_trials: int = 50,
) -> Tuple[lgb.Booster, dict]:
    """Train LightGBM classifier with Optuna hyperparameter optimization.
    
    Args:
        X: feature array
        y: label array
        feature_names: list of feature names
        n_optuna_trials: number of Optuna trials (default 50)
        
    Returns:
        Trained model and metrics dict
    """
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM is required for training. Install with: pip install lightgbm")
    
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # --- Optuna hyperparameter tuning ---
    if HAS_OPTUNA:
        logger.info(f"Running Optuna hyperparameter optimization ({n_optuna_trials} trials)...")
        study = optuna.create_study(direction='maximize', study_name='lgb_baseline')
        study.optimize(
            lambda trial: _optuna_objective(trial, X_train, y_train),
            n_trials=n_optuna_trials,
            show_progress_bar=True,
        )
        best_params = study.best_params
        best_params.update({
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'seed': 42,
        })
        logger.info(f"Best Optuna AUC (CV): {study.best_value:.4f}")
        logger.info(f"Best params: {best_params}")
    else:
        logger.info("Optuna not available — using default parameters")
        best_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
        }
    
    # --- Train final model with best params ---
    logger.info("Training final LightGBM model with best parameters...")
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        best_params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Evaluate on test set
    logger.info("Evaluating model...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    logger.info("=" * 50)
    logger.info("Model Performance:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info("=" * 50)
    
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred, 
                                              target_names=['NORMAL', 'RECOVERING']))
    
    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    top_features = sorted(zip(feature_names, importance), 
                          key=lambda x: x[1], reverse=True)[:20]
    
    logger.info("\nTop 20 Feature Importances:")
    for feat, imp in top_features:
        logger.info(f"  {feat}: {imp:.2f}")
    
    return model, metrics


def save_model(model: lgb.Booster, feature_names: List[str], metrics: dict) -> None:
    """Save trained model to artifacts directory.
    
    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        metrics: Training metrics
    """
    # Create artifacts directory
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = ARTIFACTS_DIR / "sensor_baseline.pkl"
    
    save_dict = {
        'model': model,
        'feature_names': feature_names,
        'metrics': metrics,
        'model_type': 'lightgbm',
        'version': 'v1.0.0'
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_dict, f)
    
    logger.info(f"Model saved to {output_path}")
    
    # Also save feature names separately for reference
    names_path = ARTIFACTS_DIR / "feature_names.txt"
    with open(names_path, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    logger.info(f"Feature names saved to {names_path}")


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Sensor Baseline Model Training")
    logger.info("=" * 60)
    
    # Load data
    df = load_data()
    
    # Extract features
    X, y, feature_names = extract_features(df)
    
    # Train model
    model, metrics = train_model(X, y, feature_names)
    
    # Save model
    save_model(model, feature_names, metrics)
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
