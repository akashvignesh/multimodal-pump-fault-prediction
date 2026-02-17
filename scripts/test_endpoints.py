"""Quick smoke test for all prediction endpoints."""
import json
import httpx

BASE = "http://localhost:8000"

def test_health():
    r = httpx.get(f"{BASE}/health")
    print(f"[health] {r.status_code} {r.json()}")
    assert r.status_code == 200

def test_predict():
    payload = {
        "asset_id": "pump_017",
        "timestamp": "2026-02-12T10:30:00Z",
        "sensor_window": [{"sensor_00": 2.44, "sensor_01": 46.31}],
        "text": ["vibration noticed"],
        "image_refs": [],
        "video_refs": [],
        "audio_refs": [],
        "attachment": [],
    }
    r = httpx.post(f"{BASE}/predict", json=payload)
    print(f"[predict] {r.status_code} {r.json()}")
    assert r.status_code == 200

def test_predict_no_sensor():
    """Test multimodal predict with no sensor data (text only)."""
    payload = {
        "asset_id": "pump_text_only",
        "timestamp": "2026-02-12T10:30:00Z",
        "sensor_window": [],
        "text": ["severe vibration detected, maintenance overdue"],
        "image_refs": [],
        "video_refs": [],
        "audio_refs": [],
        "attachment": [],
    }
    r = httpx.post(f"{BASE}/predict", json=payload)
    print(f"[predict_no_sensor] {r.status_code} {r.json()}")
    assert r.status_code == 200

def test_baseline():
    sensor_json = json.dumps([{"sensor_00": 2.44, "sensor_01": 46.31, "sensor_02": 52.34}])
    r = httpx.post(
        f"{BASE}/predict/baseline",
        data={"sensor_json": sensor_json, "asset_id": "pump_bl_test"},
    )
    print(f"[baseline] {r.status_code} {r.json()}")
    assert r.status_code == 200

def test_multimodal_text_only():
    r = httpx.post(
        f"{BASE}/predict/multimodal",
        data={
            "asset_id": "mm_text_only",
            "text": "Severe vibration and seal pressure dropping",
        },
    )
    print(f"[multimodal_text] {r.status_code} {r.json()}")
    assert r.status_code == 200

if __name__ == "__main__":
    test_health()
    test_predict()
    test_predict_no_sensor()
    test_baseline()
    test_multimodal_text_only()
    print("\n=== ALL TESTS PASSED ===")
