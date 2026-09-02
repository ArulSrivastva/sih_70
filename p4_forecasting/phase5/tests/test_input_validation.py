"""Input validation tests: accepted vs rejected histories."""

import numpy as np
import pytest

from phase5.inference.input_validation import (CANONICAL_FIELDS, BOUNDS,
                                               InputError, parse_history,
                                               validate_timestamps)


def _np_history(mutate=None):
    h = np.array([
        [23.30, 68.50, 55.6, 994.0, 28.67, 8.5095, 1.2749],
        [23.40, 68.00, 64.8, 993.0, 28.26, 4.4456, 7.0864],
        [23.40, 67.10, 64.8, 993.0, 28.43, 7.9810, -3.3996],
        [23.60, 66.40, 64.8, 992.0, 28.69, 2.0987, -4.1294],
        [23.50, 65.70, 74.1, 991.0, 28.85, 2.4551, 4.3582],
    ], dtype=np.float32)
    if mutate:
        mutate(h)
    return h


def test_valid_numpy_accepted(example_history_np):
    out = parse_history(example_history_np)
    assert out.shape == (5, 7)
    assert out.dtype == np.float32


def test_valid_dict_accepted(example_history_dict):
    out = parse_history(example_history_dict)
    assert out.shape == (5, 7)


def test_bare_list_of_dicts_accepted(example_history_dict):
    out = parse_history(example_history_dict["history"])
    assert out.shape == (5, 7)


def test_wrong_timestep_count_rejected():
    steps = {"timestamps": ["2024-08-25T00:00:00Z", "2024-08-25T06:00:00Z",
                            "2024-08-25T12:00:00Z", "2024-08-25T18:00:00Z"],
             "history": [dict.fromkeys(CANONICAL_FIELDS, 20.0)] * 4}
    _expect_code(steps, "insufficient_history")


def test_wrong_shape_rejected():
    h = _np_history()
    with pytest.raises(InputError) as ei:
        parse_history(h[:4])          # 4 rows
    assert ei.value.code == "wrong_shape"
    with pytest.raises(InputError) as ei:
        parse_history(np.zeros((5, 16), dtype=np.float32))
    assert ei.value.code == "wrong_shape"


def test_wrong_feature_count_rejected():
    h = np.zeros((5, 8), dtype=np.float32)
    with pytest.raises(InputError) as ei:
        parse_history(h)
    assert ei.value.code == "wrong_shape"


def _expect_code(history, code):
    with pytest.raises(InputError) as ei:
        parse_history(history)
    assert ei.value.code == code, f"expected {code}, got {ei.value.code}"


def test_nan_rejected():
    h = _np_history(lambda a: a.__setitem__((2, 1), np.nan))
    _expect_code(h, "nan_values")


def test_infinity_rejected():
    h = _np_history(lambda a: a.__setitem__((4, 4), np.inf))
    _expect_code(h, "infinite_values")


def test_invalid_latitude_rejected():
    h = _np_history(lambda a: a.__setitem__((0, 0), 95.0))
    _expect_code(h, "out_of_range")
    h = _np_history(lambda a: a.__setitem__((0, 0), -91.0))
    _expect_code(h, "out_of_range")


def test_invalid_longitude_rejected():
    h = _np_history(lambda a: a.__setitem__((1, 1), -10.0))
    _expect_code(h, "out_of_range")
    h = _np_history(lambda a: a.__setitem__((2, 1), 500.0))
    _expect_code(h, "out_of_range")


def test_impossible_wind_rejected():
    h = _np_history(lambda a: a.__setitem__((3, 2), -2.0))
    _expect_code(h, "out_of_range")


def test_impossible_pressure_rejected():
    h = _np_history(lambda a: a.__setitem__((0, 3), 700.0))
    _expect_code(h, "out_of_range")


def test_insufficient_history_rejected():
    steps = {"timestamps": ["2024-08-25T00:00:00Z"] * 4,
             "history": [dict.fromkeys(CANONICAL_FIELDS, 20.0)] * 4}
    _expect_code(steps, "insufficient_history")


def test_malformed_timestamp_rejected():
    bad = {"timestamps": ["nonsense", "2024-08-25T06:00:00Z",
                          "2024-08-25T12:00:00Z", "2024-08-25T18:00:00Z",
                          "2024-08-26T00:00:00Z"],
           "history": [dict.fromkeys(CANONICAL_FIELDS, 20.0)] * 5}
    _expect_code(bad, "malformed_timestamp")


def test_non_monotonic_timestamps_rejected():
    bad = {"timestamps": ["2024-08-25T06:00:00Z", "2024-08-25T00:00:00Z",
                          "2024-08-25T12:00:00Z", "2024-08-25T18:00:00Z",
                          "2024-08-26T00:00:00Z"],
           "history": [dict.fromkeys(CANONICAL_FIELDS, 20.0)] * 5}
    _expect_code(bad, "non_monotonic_timestamps")


def test_not_6hourly_rejected():
    bad = {"timestamps": ["2024-08-25T00:00:00Z", "2024-08-25T08:00:00Z",
                          "2024-08-25T12:00:00Z", "2024-08-25T18:00:00Z",
                          "2024-08-26T00:00:00Z"],
           "history": [dict.fromkeys(CANONICAL_FIELDS, 20.0)] * 5}
    _expect_code(bad, "timestamps_not_6hourly")


def test_valid_timestamps_accepted():
    times = ["2024-08-25T00:00:00Z", "2024-08-25T06:00:00Z",
             "2024-08-25T12:00:00Z", "2024-08-25T18:00:00Z",
             "2024-08-26T00:00:00Z"]
    parsed = validate_timestamps(times)
    assert len(parsed) == 5


def test_bounds_documented():
    # recorded physical bounds used by validation (used for the report)
    assert BOUNDS["lat"] == (-90.0, 90.0)
    assert BOUNDS["lon"] == (0.0, 360.0)
    assert BOUNDS["wind_speed"][0] >= 0.0