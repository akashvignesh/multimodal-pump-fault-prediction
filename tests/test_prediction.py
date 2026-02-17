from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_predict_single():
    response = client.post("/predict", json={
        "asset_id": "PUMP-001",
        "timestamp": "2024-01-01T00:00:00Z",
        "sensor_window": [{"sensor_00": 1.0, "sensor_01": 2.0}],
    })
    assert response.status_code == 200
    data = response.json()
    assert "failure_probability" in data
    assert "fault_confidence" in data
    assert "top_signals" in data
    assert 0.0 <= data["failure_probability"] <= 1.0


def test_predict_batch():
    response = client.post("/predict/batch", json={
        "items": [
            {
                "asset_id": "PUMP-001",
                "timestamp": "2024-01-01T00:00:00Z",
                "sensor_window": [{"sensor_00": 1.0, "sensor_01": 2.0}],
            },
            {
                "asset_id": "PUMP-002",
                "timestamp": "2024-01-01T00:00:00Z",
                "sensor_window": [{"sensor_00": 3.0, "sensor_01": 4.0}],
            },
        ]
    })
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_predict_missing_required_field():
    # Missing asset_id
    response = client.post("/predict", json={
        "timestamp": "2024-01-01T00:00:00Z",
        "sensor_window": [{"sensor_00": 1.0}],
    })
    assert response.status_code == 422