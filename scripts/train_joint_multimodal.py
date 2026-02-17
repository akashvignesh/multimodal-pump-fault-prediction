"""Joint multimodal training: sensor features + CLIP image embeddings.

Instead of training sensor and image models **separately**, this script:

1. **Joins** ``sensor_data.csv`` and ``image_mapping.csv`` on ``serial_number``
   so every sample has both 52 sensor readings **and** a pump image.

2. **Extracts features** for each modality:
   - Sensor: 260-dim (5 statistics × 52 sensors, same as baseline)
   - Image:  512-dim CLIP ViT-B/32 embeddings (frozen encoder)

3. **Trains a joint classifier** (two options):
   a. **LightGBM** on concatenated [260 + 512 = 772]-dim features  → fast, strong
   b. **Neural fusion head** that mirrors the TransformerCrossModalFusion
      architecture used at inference time → end-to-end trainable

4. **Trains the TransformerCrossModalFusion** weights so the inference-time
   cross-modal attention is no longer randomly initialised.

Label mapping (confirmed from data):
    NORMAL     + Normal (image_type)  → 0  (healthy)
    RECOVERING + Corroded             → 1  (fault / at risk)

Usage:
    python scripts/train_joint_multimodal.py [--epochs 20] [--batch-size 16]
"""
import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    f1_score,
)

try:
    from PIL import Image as PILImage

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import CLIPModel, CLIPProcessor

    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "multimodal_model"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


# ===================================================================
# 1.  DATA LOADING & JOINING
# ===================================================================

def load_joined_data() -> pd.DataFrame:
    """Merge sensor readings + image mapping on ``serial_number``.

    Returns a DataFrame with columns:
        serial_number, sensor_00 … sensor_51, machine_status,
        image_location, image_type
    """
    sensor_df = pd.read_csv(DATA_DIR / "sensor_data.csv")
    image_df = pd.read_csv(DATA_DIR / "image_mapping.csv")

    merged = sensor_df.merge(
        image_df[["serial_number", "image_location", "image_type"]],
        on="serial_number",
        how="inner",
    )
    logger.info(
        f"Joined data: {len(merged)} rows  "
        f"(sensor={len(sensor_df)}, images={len(image_df)})"
    )
    logger.info(f"Label distribution:\n{merged['machine_status'].value_counts().to_string()}")

    # Verify consistency
    mapping_ok = (
        (merged.loc[merged["machine_status"] == "NORMAL", "image_type"] == "Normal").all()
        and (merged.loc[merged["machine_status"] == "RECOVERING", "image_type"] == "Corroded").all()
    )
    if not mapping_ok:
        logger.warning("machine_status ↔ image_type mapping has mismatches!")
    else:
        logger.info("Confirmed: NORMAL→Normal, RECOVERING→Corroded  ✓")

    return merged


# ===================================================================
# 2.  FEATURE EXTRACTION
# ===================================================================

SENSOR_COLS: Optional[List[str]] = None  # populated at runtime


def extract_sensor_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """260-dim sensor features (5 stats × 52 sensors).

    Mirrors ``train_baseline.py`` so the same feature space is used.
    """
    global SENSOR_COLS
    SENSOR_COLS = sorted([c for c in df.columns if c.startswith("sensor_")])
    feature_names: List[str] = []

    rows: List[List[float]] = []
    for _, row in df.iterrows():
        feats: List[float] = []
        for col in SENSOR_COLS:
            val = row[col]
            if pd.isna(val):
                feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                v = float(val)
                feats.extend([v, 0.0, v, v, 0.0])  # mean, std, min, max, range
        rows.append(feats)

    if not feature_names:
        for col in SENSOR_COLS:
            feature_names.extend(
                [f"{col}_mean", f"{col}_std", f"{col}_min", f"{col}_max", f"{col}_range"]
            )

    X = np.nan_to_num(np.array(rows, dtype=np.float32))
    logger.info(f"Sensor features shape: {X.shape}")
    return X, feature_names


def extract_clip_embeddings(
    df: pd.DataFrame,
    clip_model,
    clip_processor,
    batch_size: int = 32,
) -> np.ndarray:
    """512-dim CLIP ViT-B/32 image embeddings (frozen encoder)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = clip_model.to(device).eval()

    all_embeds: List[np.ndarray] = []
    paths = [str(DATA_DIR / row["image_location"]) for _, row in df.iterrows()]

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        pil_images = []
        for p in batch_paths:
            try:
                pil_images.append(PILImage.open(p).convert("RGB"))
            except Exception:
                pil_images.append(PILImage.new("RGB", (224, 224)))

        inputs = clip_processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            raw = clip_model.get_image_features(pixel_values=pixel_values)
            # get_image_features may return a tensor or a BaseModelOutput
            if hasattr(raw, "pooler_output"):
                raw = raw.pooler_output
            embeds = raw / raw.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds.cpu().numpy())

    X_img = np.concatenate(all_embeds, axis=0).astype(np.float32)
    logger.info(f"CLIP image embeddings shape: {X_img.shape}")
    return X_img


def build_labels(df: pd.DataFrame) -> np.ndarray:
    """Binary labels: NORMAL=0, RECOVERING=1."""
    label_map = {"NORMAL": 0, "RECOVERING": 1}
    return df["machine_status"].map(label_map).values.astype(np.int64)


# ===================================================================
# 3-a.  LIGHTGBM ON JOINT FEATURES (concatenated)
# ===================================================================

def _optuna_joint_objective(trial, X_train, y_train):
    """Optuna objective for joint LightGBM hyperparameter tuning."""
    from sklearn.model_selection import StratifiedKFold

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


def train_joint_lgb(
    X_sensor: np.ndarray,
    X_image: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_optuna_trials: int = 50,
) -> Dict:
    """Train LightGBM on [sensor || image] = 772-dim feature vector with Optuna tuning."""
    if not HAS_LGB:
        logger.warning("LightGBM not installed – skipping joint LGB training")
        return {}

    X = np.hstack([X_sensor, X_image])
    img_feat_names = [f"clip_{i:03d}" for i in range(X_image.shape[1])]
    all_names = feature_names + img_feat_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # --- Optuna hyperparameter tuning ---
    if HAS_OPTUNA:
        logger.info(f"Running Optuna hyperparameter optimization ({n_optuna_trials} trials)...")
        study = optuna.create_study(direction='maximize', study_name='lgb_joint')
        study.optimize(
            lambda trial: _optuna_joint_objective(trial, X_train, y_train),
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
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
        }

    # --- Train final model with best params ---
    logger.info("Training final joint LightGBM with best parameters...")
    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=all_names)
    valid_ds = lgb.Dataset(X_test, label=y_test, reference=train_ds)

    model = lgb.train(
        best_params,
        train_ds,
        num_boost_round=500,
        valid_sets=[valid_ds],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    logger.info("=" * 55)
    logger.info("Joint LightGBM  (sensor + CLIP image)")
    logger.info(f"  Accuracy : {acc:.4f}")
    logger.info(f"  ROC-AUC  : {auc:.4f}")
    logger.info(f"  F1       : {f1:.4f}")
    logger.info("=" * 55)
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["NORMAL", "RECOVERING"]))

    # Feature importance – top 20
    importance = model.feature_importance(importance_type="gain")
    top = sorted(zip(all_names, importance), key=lambda x: x[1], reverse=True)[:20]
    logger.info("Top-20 features:")
    for name, imp in top:
        logger.info(f"  {name:>25s}  {imp:.1f}")

    return {
        "model": model,
        "feature_names": all_names,
        "metrics": {"accuracy": acc, "roc_auc": auc, "f1": f1},
        "model_type": "lightgbm_joint",
        "version": "v1.0.0",
    }


# ===================================================================
# 3-b.  NEURAL FUSION HEAD  (train TransformerCrossModalFusion)
# ===================================================================

class JointDataset(Dataset):
    """Returns (sensor_embedding, image_embedding, label)."""

    def __init__(self, X_sensor: np.ndarray, X_image: np.ndarray, y: np.ndarray):
        self.sensor = torch.tensor(X_sensor, dtype=torch.float32)
        self.image = torch.tensor(X_image, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sensor[idx], self.image[idx], self.labels[idx]


def train_transformer_fusion(
    X_sensor: np.ndarray,
    X_image: np.ndarray,
    y: np.ndarray,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 3e-4,
) -> Dict:
    """Train the **TransformerCrossModalFusion** so cross-modal attention
    is no longer randomly initialised.

    Uses the same architecture defined in ``src/models/transformer_fusion.py``
    (d_model=256, 2 layers, 4 heads) but trains the risk_head end-to-end
    with BCE loss on fault labels.
    """
    if not HAS_TORCH:
        logger.warning("PyTorch not installed – skipping transformer fusion training")
        return {}

    # Ensure src is importable
    import sys
    src_root = str(BASE_DIR)
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    from src.models.transformer_fusion import TransformerFusion

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Build the model from the inference class -----
    fusion = TransformerFusion()  # initialises _TransformerCrossModalFusion
    if not fusion.available:
        logger.error("TransformerFusion reports unavailable (no PyTorch?)")
        return {}

    model = fusion._model.to(device)  # the nn.Module
    model.train()

    # ----- Data split -----
    X_s_train, X_s_test, X_i_train, X_i_test, y_train, y_test = train_test_split(
        X_sensor, X_image, y, test_size=0.2, random_state=42, stratify=y,
    )

    train_ds = JointDataset(X_s_train, X_i_train, y_train)
    test_ds = JointDataset(X_s_test, X_i_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # ----- Optimiser & scheduler -----
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    logger.info(f"Training TransformerCrossModalFusion for {epochs} epochs on {device} …")

    best_auc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for sensor_batch, image_batch, label_batch in train_loader:
            sensor_batch = sensor_batch.to(device)
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            modality_embs = {
                "sensor": sensor_batch,
                "image": image_batch,
            }
            out = model(modality_embs)  # dict with "failure_prob", "confidence"
            risk = out["failure_prob"].reshape(-1)

            loss = criterion(risk, label_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # ----- Evaluate every 5 epochs -----
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for sb, ib, lb in test_loader:
                    sb, ib, lb = sb.to(device), ib.to(device), lb.to(device)
                    out = model({"sensor": sb, "image": ib})
                    p = out["failure_prob"].reshape(-1).cpu().numpy()
                    all_probs.extend(p.tolist())
                    all_labels.extend(lb.cpu().numpy().tolist())

            probs = np.array(all_probs)
            labels = np.array(all_labels)
            preds = (probs > 0.5).astype(int)
            acc = accuracy_score(labels, preds)
            auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
            f1 = f1_score(labels, preds)
            avg_loss = running_loss / max(n_batches, 1)

            logger.info(
                f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
                f"acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}"
            )

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ----- Final eval with best weights -----
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for sb, ib, lb in test_loader:
            sb, ib = sb.to(device), ib.to(device)
            out = model({"sensor": sb, "image": ib})
            p = out["failure_prob"].reshape(-1).cpu().numpy()
            all_probs.extend(p.tolist())
            all_labels.extend(lb.numpy().tolist())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs > 0.5).astype(int)
    final_acc = accuracy_score(labels, preds)
    final_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    final_f1 = f1_score(labels, preds)

    logger.info("=" * 55)
    logger.info("TransformerCrossModalFusion  (sensor + image, trained)")
    logger.info(f"  Accuracy : {final_acc:.4f}")
    logger.info(f"  ROC-AUC  : {final_auc:.4f}")
    logger.info(f"  F1       : {final_f1:.4f}")
    logger.info("=" * 55)
    logger.info(
        "\n"
        + classification_report(
            labels, preds, target_names=["NORMAL", "RECOVERING"], zero_division=0,
        )
    )

    return {
        "state_dict": best_state or model.state_dict(),
        "metrics": {"accuracy": final_acc, "roc_auc": final_auc, "f1": final_f1},
        "model_config": {"d_model": 256, "nhead": 4, "num_layers": 2},
    }


# ===================================================================
# 4.  SAVE ARTIFACTS
# ===================================================================

def save_artifacts(
    lgb_result: Dict,
    transformer_result: Dict,
    clip_embeds: Optional[np.ndarray] = None,
) -> None:
    """Persist trained models to ``artifacts/``."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Joint LightGBM
    if lgb_result:
        path = ARTIFACTS_DIR / "joint_sensor_image.pkl"
        with open(path, "wb") as f:
            pickle.dump(lgb_result, f)
        logger.info(f"Joint LightGBM saved → {path}")

    # Trained transformer fusion weights
    if transformer_result and "state_dict" in transformer_result:
        path = ARTIFACTS_DIR / "transformer_fusion_trained.pt"
        torch.save(
            {
                "state_dict": transformer_result["state_dict"],
                "model_config": transformer_result["model_config"],
                "metrics": transformer_result["metrics"],
            },
            path,
        )
        logger.info(f"Transformer fusion weights saved → {path}")

    # Pre-computed CLIP embeddings (useful for fast retraining)
    if clip_embeds is not None:
        path = ARTIFACTS_DIR / "clip_image_embeddings.npy"
        np.save(path, clip_embeds)
        logger.info(f"CLIP embeddings cache saved → {path}")

    # Updated multimodal config
    config = {
        "modalities": ["sensor", "image"],
        "image_model": "clip_vit_b32",
        "image_encoder": "openai/clip-vit-base-patch32",
        "image_embedding_dim": 512,
        "sensor_feature_dim": 260,
        "joint_feature_dim": 772,
        "fusion_type": "hybrid_transformer_gated",
        "fusion_trained": True,
        "version": "v2.0.0",
    }
    path = ARTIFACTS_DIR / "multimodal_config.pkl"
    with open(path, "wb") as f:
        pickle.dump(config, f)
    logger.info(f"Config saved → {path}")


# ===================================================================
# 5.  MAIN
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Joint multimodal training")
    parser.add_argument("--epochs", type=int, default=30, help="Transformer training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Joint Multimodal Training  (sensor + CLIP image)")
    logger.info("=" * 60)

    # ---------- Load & join ----------
    df = load_joined_data()
    y = build_labels(df)

    # ---------- Sensor features ----------
    logger.info("-" * 60)
    logger.info("Extracting sensor features …")
    X_sensor, sensor_feat_names = extract_sensor_features(df)

    # ---------- CLIP image embeddings ----------
    logger.info("-" * 60)
    logger.info("Extracting CLIP image embeddings …")

    if not HAS_CLIP or not HAS_PIL:
        raise RuntimeError("transformers + Pillow required  (pip install transformers Pillow)")

    # Check for cached embeddings
    cache_path = ARTIFACTS_DIR / "clip_image_embeddings.npy"
    if cache_path.exists():
        X_image = np.load(cache_path)
        logger.info(f"Loaded cached CLIP embeddings from {cache_path}  shape={X_image.shape}")
        if len(X_image) != len(df):
            logger.warning("Cache size mismatch – re-extracting")
            cache_path.unlink()
            X_image = None
    else:
        X_image = None

    if X_image is None:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        X_image = extract_clip_embeddings(df, clip_model, clip_processor)
        del clip_model, clip_processor  # free VRAM / RAM
        if HAS_TORCH:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---------- Train joint LightGBM ----------
    logger.info("-" * 60)
    logger.info("Phase 1: Joint LightGBM  (sensor + image concatenated)")
    lgb_result = train_joint_lgb(X_sensor, X_image, y, sensor_feat_names)

    # ---------- Train TransformerCrossModalFusion ----------
    logger.info("-" * 60)
    logger.info("Phase 2: TransformerCrossModalFusion  (cross-modal attention)")
    transformer_result = train_transformer_fusion(
        X_sensor, X_image, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # ---------- Save ----------
    save_artifacts(lgb_result, transformer_result, clip_embeds=X_image)

    logger.info("=" * 60)
    logger.info("Joint Multimodal Training Complete!")
    if lgb_result:
        m = lgb_result["metrics"]
        logger.info(f"  LightGBM   → Acc={m['accuracy']:.4f}  AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}")
    if transformer_result:
        m = transformer_result["metrics"]
        logger.info(f"  Transformer→ Acc={m['accuracy']:.4f}  AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
