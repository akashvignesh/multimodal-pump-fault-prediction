#!/usr/bin/env python
"""Benchmark per-request inference latency.

Usage:
    python scripts/benchmark_latency.py                   # via API (default)
    python scripts/benchmark_latency.py --offline          # direct model call
    python scripts/benchmark_latency.py --requests 500     # 500 requests
    python scripts/benchmark_latency.py --output results.json

Reports: p50, p95, p99, mean, std, min, max latency in ms.
"""
import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Default sensor payload for benchmarking
SENSOR_PAYLOAD = {
    "asset_id": "bench_pump",
    "timestamp": "2026-01-01T00:00:00Z",
    "sensor_window": [
        {
            f"sensor_{i:02d}": round(2.0 + i * 0.5, 2)
            for i in range(52)
        }
    ],
}


def benchmark_api(base_url: str, n_requests: int, warmup: int = 5) -> list[float]:
    """Benchmark latency via HTTP API calls.

    Returns list of latencies in milliseconds.
    """
    import httpx

    client = httpx.Client(base_url=base_url, timeout=30)

    # Health check
    try:
        r = client.get("/health")
        r.raise_for_status()
    except Exception as e:
        logger.error(f"API not reachable at {base_url}: {e}")
        sys.exit(1)

    # Warmup
    logger.info(f"Warming up ({warmup} requests)...")
    for _ in range(warmup):
        client.post("/predict/baseline", data={
            "sensor_json": json.dumps(SENSOR_PAYLOAD["sensor_window"]),
            "asset_id": "warmup",
        })

    # Benchmark
    logger.info(f"Benchmarking {n_requests} requests...")
    latencies: list[float] = []
    for i in range(n_requests):
        t0 = time.perf_counter()
        r = client.post("/predict/baseline", data={
            "sensor_json": json.dumps(SENSOR_PAYLOAD["sensor_window"]),
            "asset_id": f"bench_{i}",
        })
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(elapsed)

        if r.status_code != 200:
            logger.warning(f"Request {i} failed: {r.status_code}")

        if (i + 1) % 100 == 0:
            logger.info(f"  {i + 1}/{n_requests} done")

    client.close()
    return latencies


def benchmark_offline(n_requests: int, warmup: int = 5) -> list[float]:
    """Benchmark latency via direct model calls (no network overhead)."""
    from src.models.risk_model import SensorBaselineModel

    model = SensorBaselineModel()
    sensor_window = SENSOR_PAYLOAD["sensor_window"]

    # Warmup
    logger.info(f"Warming up ({warmup} calls)...")
    for _ in range(warmup):
        model.predict(sensor_window)

    # Benchmark
    logger.info(f"Benchmarking {n_requests} direct model calls...")
    latencies: list[float] = []
    for i in range(n_requests):
        t0 = time.perf_counter()
        model.predict(sensor_window)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(elapsed)

        if (i + 1) % 100 == 0:
            logger.info(f"  {i + 1}/{n_requests} done")

    return latencies


def compute_stats(latencies: list[float]) -> dict[str, float]:
    """Compute latency statistics from a list of values in ms."""
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    return {
        "n_requests": n,
        "mean_ms": round(statistics.mean(latencies), 3),
        "std_ms": round(statistics.stdev(latencies), 3) if n > 1 else 0.0,
        "min_ms": round(min(latencies), 3),
        "p50_ms": round(latencies_sorted[int(n * 0.50)], 3),
        "p95_ms": round(latencies_sorted[int(n * 0.95)], 3),
        "p99_ms": round(latencies_sorted[min(int(n * 0.99), n - 1)], 3),
        "max_ms": round(max(latencies), 3),
        "throughput_rps": round(n / (sum(latencies) / 1000), 1) if sum(latencies) > 0 else 0,
    }


def print_results(stats: dict, mode: str) -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"  LATENCY BENCHMARK  ({mode})")
    print(f"{'=' * 60}")
    print(f"  Requests  : {stats['n_requests']}")
    print(f"  Mean      : {stats['mean_ms']:.3f} ms")
    print(f"  Std Dev   : {stats['std_ms']:.3f} ms")
    print(f"  Min       : {stats['min_ms']:.3f} ms")
    print(f"  P50       : {stats['p50_ms']:.3f} ms")
    print(f"  P95       : {stats['p95_ms']:.3f} ms")
    print(f"  P99       : {stats['p99_ms']:.3f} ms")
    print(f"  Max       : {stats['max_ms']:.3f} ms")
    print(f"  Throughput: {stats['throughput_rps']:.1f} req/s (sequential)")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark pump fault prediction inference latency",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run direct model calls instead of API requests",
    )
    parser.add_argument(
        "--requests", "-n",
        type=int,
        default=200,
        help="Number of benchmark requests (default: 200)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup requests (default: 10)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    if args.offline:
        latencies = benchmark_offline(args.requests, args.warmup)
        mode = "offline / direct model"
    else:
        latencies = benchmark_api(args.url, args.requests, args.warmup)
        mode = f"API ({args.url})"

    stats = compute_stats(latencies)
    stats["mode"] = mode
    print_results(stats, mode)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(stats, indent=2))
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
