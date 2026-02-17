from src.services.preprocessing import preprocess_sensor_window, validate_sensor_data
import pytest


def test_preprocess_sensor_window_valid_input():
    input_data = [
        {"sensor_00": 1.0, "sensor_01": 2.0, "sensor_02": None, "sensor_03": 4.0}
    ]
    output = preprocess_sensor_window(input_data)
    assert output == [{"sensor_00": 1.0, "sensor_01": 2.0, "sensor_02": 0.0, "sensor_03": 4.0}]


def test_preprocess_sensor_window_missing_all():
    input_data = [
        {"sensor_00": None, "sensor_01": None}
    ]
    output = preprocess_sensor_window(input_data)
    assert output == [{"sensor_00": 0.0, "sensor_01": 0.0}]


def test_preprocess_sensor_window_string_values():
    input_data = [
        {"sensor_00": "1.5", "sensor_01": "invalid"}
    ]
    output = preprocess_sensor_window(input_data)
    assert output == [{"sensor_00": 1.5, "sensor_01": 0.0}]


def test_preprocess_sensor_window_empty():
    assert preprocess_sensor_window([]) == []


def test_validate_sensor_data_valid():
    assert validate_sensor_data([{"sensor_00": 1.0}]) is True


def test_validate_sensor_data_empty():
    assert validate_sensor_data([]) is False