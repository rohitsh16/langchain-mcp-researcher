"""
Unit tests for temperature scaling and isotonic calibrators.
"""
import pytest
from ai_correctness_mcp_server.uncertainty.calibration import TemperatureScaling, IsotonicCalibrator


def test_temperature_scaling_overconfident():
    # Overconfident model where 0.9 confidence only has 50% accuracy
    scores = [0.9] * 10 + [0.1] * 10
    labels = [1] * 5 + [0] * 5 + [0] * 10
    calibrator = TemperatureScaling()
    calibrator.fit(scores, labels)

    # Temperature should soften overconfidence (T > 1.0)
    assert calibrator.temperature > 1.0
    calibrated_09 = calibrator.calibrate(0.9)
    assert calibrated_09 < 0.9


def test_isotonic_calibrator_monotonicity():
    scores = [0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9]
    labels = [0, 0, 1, 0, 1, 1, 1]
    iso = IsotonicCalibrator()
    iso.fit(scores, labels)

    cal_scores = [iso.calibrate(s) for s in [0.15, 0.3, 0.6, 0.85]]
    # Output must be non-decreasing
    for i in range(len(cal_scores) - 1):
        assert cal_scores[i] <= cal_scores[i + 1] + 1e-6
