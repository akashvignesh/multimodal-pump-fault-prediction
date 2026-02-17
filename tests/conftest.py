from pytest import fixture

@fixture
def sample_data():
    return {
        "sensor_data": {
            "temperature": 75,
            "pressure": 30,
            "vibration": 0.5
        },
        "additional_info": {
            "pump_id": "pump_123",
            "timestamp": "2023-10-01T12:00:00Z"
        }
    }