#!/usr/bin/env python
"""Unified training entry-point for all models.

Usage:
    python scripts/train.py                     # train all models
    python scripts/train.py --model baseline    # sensor-only LightGBM
    python scripts/train.py --model multimodal  # joint sensor+image model
    python scripts/train.py --model all         # both (default)

Outputs are written to artifacts/.
"""
import argparse
import importlib
import logging
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_script(name: str, script_path: Path, extra_args: list[str] | None = None) -> bool:
    """Run a training sub-script and return True on success."""
    logger.info(f"{'=' * 60}")
    logger.info(f"  Training: {name}")
    logger.info(f"  Script:   {script_path.relative_to(BASE_DIR)}")
    logger.info(f"{'=' * 60}")

    cmd = [sys.executable, str(script_path)] + (extra_args or [])
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        logger.info(f"  {name} completed in {elapsed:.1f}s")
        return True
    else:
        logger.error(f"  {name} FAILED (exit code {result.returncode})")
        return False


def train_baseline(extra_args: list[str] | None = None) -> bool:
    """Train the sensor-only LightGBM baseline model."""
    script = BASE_DIR / "scripts" / "train_baseline.py"
    if not script.exists():
        logger.error(f"Baseline training script not found: {script}")
        return False
    return run_script("Sensor Baseline (LightGBM)", script, extra_args)


def train_multimodal(extra_args: list[str] | None = None) -> bool:
    """Train the joint multimodal (sensor + image) model."""
    script = BASE_DIR / "scripts" / "train_joint_multimodal.py"
    if not script.exists():
        logger.error(f"Multimodal training script not found: {script}")
        return False
    return run_script("Joint Multimodal (LightGBM + Transformer)", script, extra_args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train pump fault risk prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py                         Train all models
  python scripts/train.py --model baseline        Train sensor-only model
  python scripts/train.py --model multimodal      Train multimodal model
  python scripts/train.py --model multimodal -- --epochs 30
        """,
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "multimodal", "all"],
        default="all",
        help="Which model to train (default: all)",
    )

    args, extra = parser.parse_known_args()

    logger.info("Pump Fault Risk — Model Training")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Model selection: {args.model}")

    results: dict[str, bool] = {}

    if args.model in ("baseline", "all"):
        results["baseline"] = train_baseline(extra if args.model == "baseline" else None)

    if args.model in ("multimodal", "all"):
        results["multimodal"] = train_multimodal(extra if args.model == "multimodal" else None)

    # Summary
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<20s} {status}")
    print("=" * 60)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
