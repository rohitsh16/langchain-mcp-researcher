"""
Unit tests for selective prediction, risk-coverage trade-off, and AURC.
"""
import pytest
from ai_correctness_mcp_server.selective.abstention import SelectivePredictor
from ai_correctness_mcp_server.selective.risk_coverage import (
    compute_risk_coverage_curve,
    compute_accuracy_at_coverage,
)


def test_selective_predictor_threshold():
    predictor = SelectivePredictor(threshold=0.8)
    assert predictor.decide(0.85).accepted is True
    assert predictor.decide(0.85).action == "ACCEPT"
    assert predictor.decide(0.75).accepted is False
    assert predictor.decide(0.75).action == "ABSTAIN"


def test_calibrate_threshold_for_coverage():
    confs = [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]
    # For target coverage 50% (top 5 of 10), threshold should be around 0.70
    predictor = SelectivePredictor.calibrate_threshold_for_coverage(confs, target_coverage=0.5)
    decisions = predictor.decide_batch(confs)
    accepted_ratio = sum(1 for d in decisions if d.accepted) / len(confs)
    assert abs(accepted_ratio - 0.5) <= 0.1


def test_risk_coverage_and_aurc():
    # Model with good ranking: top confidence has 0 loss, lowest confidence has 1 loss
    confs = [0.9, 0.8, 0.7, 0.6, 0.5]
    losses = [0.0, 0.0, 0.0, 1.0, 1.0]

    res = compute_risk_coverage_curve(confs, losses)
    assert len(res["coverages"]) == 5
    assert len(res["risks"]) == 5
    assert res["risks"][0] == 0.0  # At highest confidence, risk is 0
    assert res["aurc"] >= 0.0
    assert res["excess_aurc"] == 0.0  # Matches oracle ordering perfectly


def test_accuracy_at_coverage():
    confs = [0.9, 0.8, 0.7, 0.6, 0.5]
    correct = [1, 1, 1, 0, 0]
    accs = compute_accuracy_at_coverage(confs, correct, target_coverages=[0.4, 0.6, 1.0])
    assert accs[0.4] == 1.0
    assert accs[0.6] == 1.0
    assert accs[1.0] == 0.6
