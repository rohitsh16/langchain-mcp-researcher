"""
Unit tests for calibration and selective risk metrics.
"""
import pytest
from research.metrics.calibration import (
    compute_expected_calibration_error,
    compute_adaptive_calibration_error,
    compute_brier_score,
)
from research.metrics.risk import (
    compute_empirical_coverage,
    compute_selective_risk,
    wilson_score_interval,
)


def test_perfect_calibration_ece():
    # Model that predicts 0.8 on 10 samples, exactly 8 are correct
    probs = [0.8] * 10
    labels = [1] * 8 + [0] * 2
    ece = compute_expected_calibration_error(probs, labels, num_bins=5)
    assert abs(ece) < 1e-4


def test_brier_score():
    # Perfect predictions have Brier score 0
    assert compute_brier_score([1.0, 0.0], [1, 0]) == 0.0
    # Worst predictions have Brier score 1
    assert compute_brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_wilson_interval():
    ci_low, ci_high = wilson_score_interval(successes=90, total=100, confidence=0.95)
    assert 0.80 < ci_low < 0.90
    assert 0.90 < ci_high < 0.96
