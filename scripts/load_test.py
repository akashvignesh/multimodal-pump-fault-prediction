"""Load test for pump fault risk prediction API.

Run:
    python scripts/load_test.py                  # baseline (before)
    python scripts/load_test.py --after          # after optimisation

Three traffic levels: Light (5 users), Medium (25), Heavy (75).
Reports throughput, p50, p95, p99 latency, CPU%, and RAM.
"""
import json
import sys
import time
import statistics
import concurrent.futures
import threading
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import psutil

# ── Payloads (matching current PredictionRequest schema) ──────────────

SENSOR_PAYLOAD = {
    "asset_id": "pump_017",
    "timestamp": "2026-02-12T10:30:00Z",
    "sensor_window": [
        {
            "sensor_00": 2.44, "sensor_01": 46.31, "sensor_02": 52.34,
            "sensor_03": 44.66, "sensor_04": 628.59, "sensor_05": 79.70,
            "sensor_06": 13.12, "sensor_07": 16.17, "sensor_08": 15.74,
            "sensor_09": 15.08, "sensor_10": 39.30
        },
        {
            "sensor_00": 2.50, "sensor_01": 46.80, "sensor_02": 52.90,
            "sensor_03": 44.70, "sensor_04": 629.10, "sensor_05": 80.00,
            "sensor_06": 13.30, "sensor_07": 16.25, "sensor_08": 15.80,
            "sensor_09": 15.10, "sensor_10": 39.50
        },
        {
            "sensor_00": 2.55, "sensor_01": 47.10, "sensor_02": 53.10,
            "sensor_03": 44.90, "sensor_04": 630.00, "sensor_05": 80.40,
            "sensor_06": 13.50, "sensor_07": 16.40, "sensor_08": 15.90,
            "sensor_09": 15.15, "sensor_10": 39.80
        },
    ],
}

# ── Level definitions ─────────────────────────────────────────────────

LEVELS = [
    {"name": "Light",  "users": 5,  "duration_s": 20},
    {"name": "Medium", "users": 25, "duration_s": 20},
    {"name": "Heavy",  "users": 75, "duration_s": 20},
]


def _percentile(sorted_vals: list, pct: float) -> float:
    idx = min(int(len(sorted_vals) * pct), len(sorted_vals) - 1)
    return sorted_vals[idx]


def _find_server_pid() -> int | None:
    """Find the uvicorn/python process listening on port 8000."""
    for conn in psutil.net_connections(kind="tcp"):
        if conn.laddr.port == 8000 and conn.status == "LISTEN":
            return conn.pid
    return None


def run_level(
    base_url: str,
    num_users: int,
    duration_s: int,
    label: str,
) -> dict:
    """Run a single load-test level and return metrics."""
    print(f"\n{'=' * 64}")
    print(f"  {label}: {num_users} concurrent users x {duration_s}s")
    print(f"{'=' * 64}")

    latencies: list = []
    errors = [0]
    lat_lock = threading.Lock()
    start = time.time()

    # Monitor the *server* process
    server_pid = _find_server_pid()
    proc = psutil.Process(server_pid) if server_pid else psutil.Process()
    cpu_samples: list = []
    mem_samples: list = []

    pool = httpx.Client(
        timeout=60,
        limits=httpx.Limits(
            max_connections=num_users + 10,
            max_keepalive_connections=num_users,
        ),
    )

    def _worker():
        while time.time() - start < duration_s:
            try:
                t0 = time.perf_counter()
                r = pool.post(f"{base_url}/predict", json=SENSOR_PAYLOAD)
                dt = (time.perf_counter() - t0) * 1000
                if r.status_code == 200:
                    with lat_lock:
                        latencies.append(dt)
                else:
                    errors[0] += 1
            except Exception:
                errors[0] += 1

    def _monitor():
        while time.time() - start < duration_s:
            try:
                cpu_samples.append(proc.cpu_percent(interval=1))
                mem_samples.append(proc.memory_info().rss / 1024 / 1024)
            except Exception:
                time.sleep(1)

    mon = threading.Thread(target=_monitor, daemon=True)
    mon.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as ex:
        futs = [ex.submit(_worker) for _ in range(num_users)]
        concurrent.futures.wait(futs)

    pool.close()
    elapsed = time.time() - start

    if not latencies:
        print(f"  No successful requests (errors={errors[0]})")
        return {}

    latencies.sort()
    throughput = len(latencies) / elapsed
    p50 = _percentile(latencies, 0.50)
    p95 = _percentile(latencies, 0.95)
    p99 = _percentile(latencies, 0.99)
    avg_cpu = statistics.mean(cpu_samples) if cpu_samples else 0
    peak_cpu = max(cpu_samples) if cpu_samples else 0
    avg_mem = statistics.mean(mem_samples) if mem_samples else 0

    result = {
        "level": label,
        "concurrent_users": num_users,
        "duration_s": duration_s,
        "total_requests": len(latencies),
        "errors": errors[0],
        "throughput_rps": round(throughput, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "avg_cpu_pct": round(avg_cpu, 1),
        "peak_cpu_pct": round(peak_cpu, 1),
        "avg_mem_mb": round(avg_mem, 1),
    }

    print(f"  Requests : {result['total_requests']}  (errors {result['errors']})")
    print(f"  Throughput: {result['throughput_rps']:.1f} req/s")
    print(f"  Latency  : p50={p50:.1f} ms  p95={p95:.1f} ms  p99={p99:.1f} ms")
    print(f"  CPU      : avg={avg_cpu:.0f}%  peak={peak_cpu:.0f}%")
    print(f"  RAM      : {avg_mem:.0f} MB")
    return result


def run_load_tests(base_url: str = "http://localhost:8000") -> list:
    # Warm-up: 3 requests so JIT / caches are primed
    print("Warming up ...")
    client = httpx.Client(timeout=30)
    for _ in range(3):
        client.post(f"{base_url}/predict", json=SENSOR_PAYLOAD)
    client.close()
    print("Warm-up done.\n")

    results = []
    for lvl in LEVELS:
        r = run_level(base_url, lvl["users"], lvl["duration_s"], lvl["name"])
        if r:
            results.append(r)

    # ── Summary table ──
    print(f"\n{'=' * 90}")
    print("LOAD TEST SUMMARY")
    print(f"{'=' * 90}")
    hdr = (
        f"{'Level':<8} | {'Users':>5} | {'Reqs':>6} | {'Err':>4} | "
        f"{'RPS':>8} | {'p50 ms':>8} | {'p95 ms':>8} | {'p99 ms':>8} | "
        f"{'CPU%':>5} | {'RAM MB':>7}"
    )
    print(hdr)
    print("-" * 90)
    for r in results:
        print(
            f"{r['level']:<8} | {r['concurrent_users']:>5} | "
            f"{r['total_requests']:>6} | {r['errors']:>4} | "
            f"{r['throughput_rps']:>7.1f}/s | {r['p50_ms']:>8.1f} | "
            f"{r['p95_ms']:>8.1f} | {r['p99_ms']:>8.1f} | "
            f"{r['avg_cpu_pct']:>4.0f}% | {r['avg_mem_mb']:>7.0f}"
        )

    # ── Persist ──
    out_dir = Path(__file__).parent.parent / "artifacts"
    out_dir.mkdir(exist_ok=True)
    tag = "after" if "--after" in sys.argv else "before"
    save_path = out_dir / f"load_test_results_{tag}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    canonical = out_dir / "load_test_results.json"
    with open(canonical, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {save_path}")
    return results


if __name__ == "__main__":
    run_load_tests()
