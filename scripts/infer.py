#!/usr/bin/env python
"""Command-line inference for pump fault risk prediction.

Usage:
    # From JSON file
    python scripts/infer.py --input sample.json

    # From stdin
    echo '{"sensor_window": [...]}' | python scripts/infer.py --stdin

    # Using built-in sample data
    python scripts/infer.py --sample normal
    python scripts/infer.py --sample at-risk

    # Multimodal with image
    python scripts/infer.py --input sample.json --image path/to/pump.jpg

    # Batch mode
    python scripts/infer.py --input batch.json --batch

Outputs JSON to stdout. Logs go to stderr.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def load_sample(name: str) -> dict:
    """Load a built-in sample from artifacts/sample_data.json."""
    sample_path = BASE_DIR / "artifacts" / "sample_data.json"
    if not sample_path.exists():
        logger.error(f"Sample data not found: {sample_path}")
        sys.exit(1)

    raw = sample_path.read_text()
    # Handle NaN values in JSON
    raw = raw.replace(": NaN", ": null").replace(":NaN", ":null")
    samples = json.loads(raw)

    key_map = {
        "normal": "normal",
        "at-risk": "recovering",
        "at_risk": "recovering",
        "recovering": "recovering",
    }
    key = key_map.get(name.lower())
    if not key or key not in samples:
        logger.error(f"Unknown sample '{name}'. Available: normal, at-risk")
        sys.exit(1)

    sample = samples[key]
    # Wrap in sensor_window format if it's a list of dicts
    if isinstance(sample, list):
        return {"sensor_window": sample}
    return sample


def predict_sensor(sensor_window: list[dict], asset_id: str = "cli") -> dict:
    """Run sensor-only prediction offline (no API server needed)."""
    from src.models.risk_model import SensorBaselineModel

    model = SensorBaselineModel()
    prob, confidence, top_signals = model.predict(sensor_window)

    # Generate risk label
    if prob < 0.3:
        risk_label = "NORMAL"
    elif prob < 0.7:
        risk_label = "AT_RISK"
    else:
        risk_label = "CRITICAL"

    return {
        "asset_id": asset_id,
        "model_version": "sensor-baseline",
        "failure_probability": round(prob, 6),
        "confidence": round(confidence, 4),
        "risk_label": risk_label,
        "top_signals": top_signals,
    }


def predict_multimodal(
    sensor_window: list[dict] | None,
    image_path: str | None,
    asset_id: str = "cli",
) -> dict:
    """Run multimodal prediction offline.
    
    Uses the sensor baseline for the sensor branch and CLIP for images.
    For full fusion, use the API endpoint instead.
    """
    from src.models.risk_model import SensorBaselineModel

    results: dict = {"asset_id": asset_id, "model_version": "multimodal-offline"}

    # Sensor branch
    if sensor_window:
        model = SensorBaselineModel()
        prob, confidence, top_signals = model.predict(sensor_window)
        results["sensor_failure_probability"] = round(prob, 6)
        results["sensor_confidence"] = round(confidence, 4)
        results["top_signals"] = top_signals

    # Image branch
    if image_path:
        p = Path(image_path)
        if not p.exists():
            logger.error(f"Image not found: {image_path}")
            sys.exit(1)
        try:
            from src.models.clip_encoder import CLIPImageEncoder
            encoder = CLIPImageEncoder()
            image_bytes = p.read_bytes()
            image_result = encoder.classify_image(image_bytes)
            results["image_fault_probability"] = round(image_result.get("fault_probability", 0), 6)
            results["image_classification"] = image_result.get("classification", "unknown")
            logger.info(f"Image classified: {results['image_classification']}")
        except Exception as e:
            logger.warning(f"Image classification failed: {e}")
            results["image_error"] = str(e)

    # Combined probability (simple average if both present)
    if "sensor_failure_probability" in results and "image_fault_probability" in results:
        results["failure_probability"] = round(
            0.6 * results["sensor_failure_probability"] + 0.4 * results["image_fault_probability"], 6
        )
    elif "sensor_failure_probability" in results:
        results["failure_probability"] = results["sensor_failure_probability"]
    elif "image_fault_probability" in results:
        results["failure_probability"] = results["image_fault_probability"]

    return results


def predict_batch(items: list[dict]) -> list[dict]:
    """Run batch sensor-only predictions."""
    from src.models.risk_model import SensorBaselineModel

    model = SensorBaselineModel()
    results = []
    for i, item in enumerate(items):
        sw = item.get("sensor_window", [])
        aid = item.get("asset_id", f"batch_{i}")
        try:
            prob, confidence, top_signals = model.predict(sw)
            results.append({
                "asset_id": aid,
                "failure_probability": round(prob, 6),
                "confidence": round(confidence, 4),
                "risk_label": "NORMAL" if prob < 0.3 else ("AT_RISK" if prob < 0.7 else "CRITICAL"),
            })
        except Exception as e:
            results.append({
                "asset_id": aid,
                "error": str(e),
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pump fault risk prediction — CLI inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, help="Path to JSON input file")
    parser.add_argument("--stdin", action="store_true", help="Read JSON from stdin")
    parser.add_argument(
        "--sample",
        choices=["normal", "at-risk"],
        help="Use built-in sample data",
    )
    parser.add_argument("--image", type=str, help="Path to pump image (enables multimodal)")
    parser.add_argument("--batch", action="store_true", help="Batch mode (input is array)")
    parser.add_argument("--asset-id", type=str, default="cli", help="Asset identifier")
    parser.add_argument(
        "--pretty", action="store_true", default=True,
        help="Pretty-print JSON output (default: true)",
    )

    args = parser.parse_args()

    # Load input
    if args.sample:
        data = load_sample(args.sample)
        logger.info(f"Using built-in sample: {args.sample}")
    elif args.stdin:
        raw = sys.stdin.read()
        data = json.loads(raw)
    elif args.input:
        p = Path(args.input)
        if not p.exists():
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        raw = p.read_text()
        raw = raw.replace(": NaN", ": null").replace(":NaN", ":null")
        data = json.loads(raw)
    else:
        parser.print_help()
        sys.exit(1)

    # Determine mode and run prediction
    if args.batch:
        items = data if isinstance(data, list) else data.get("items", [data])
        logger.info(f"Batch prediction: {len(items)} items")
        output = predict_batch(items)
    elif args.image:
        sensor_window = data.get("sensor_window", data if isinstance(data, list) else [])
        logger.info("Multimodal prediction (sensor + image)")
        output = predict_multimodal(sensor_window, args.image, args.asset_id)
    else:
        sensor_window = data.get("sensor_window", data if isinstance(data, list) else [])
        logger.info("Sensor-only prediction")
        output = predict_sensor(sensor_window, args.asset_id)

    # Output
    indent = 2 if args.pretty else None
    print(json.dumps(output, indent=indent, default=str))


if __name__ == "__main__":
    main()
