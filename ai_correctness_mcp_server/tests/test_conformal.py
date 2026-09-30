"""
Unit tests and statistical coverage verification for Conformal Prediction algorithms.
"""
import pytest
import random
from ai_correctness_mcp_server.conformal.split import (
    compute_conformal_quantile,
    SplitConformalPredictor,
)
from ai_correctness_mcp_server.conformal.risk_control import (
    ConformalRiskController,
    hoeffding_bentkus_p_value,
)
from ai_correctness_mcp_server.conformal.adaptive import MondrianConformalPredictor
from ai_correctness_mcp_server.conformal.shift import (
    compute_weighted_conformal_quantile,
    DistributionShiftDetector,
    kolmogorov_smirnov_2sample,
)


def test_conformal_quantile_formula():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # n = 10, alpha = 0.2. ceil((10 + 1) * 0.8) / 10 = ceil(8.8)/10 = 9/10 -> 9th score (0.9)
    q = compute_conformal_quantile(scores, alpha=0.2)
    assert q == 0.9


def test_finite_sample_coverage_simulation():
    """
    Monte Carlo test verifying that empirical expected coverage >= 1 - alpha under exchangeability.
    """
    rng = random.Random(42)
    n_cal = 150
    n_test = 300
    alpha = 0.10  # 90% nominal coverage
    n_trials = 20

    coverages = []
    for _ in range(n_trials):
        cal_scores = [abs(rng.gauss(0, 1)) for _ in range(n_cal)]
        test_scores = [abs(rng.gauss(0, 1)) for _ in range(n_test)]
        predictor = SplitConformalPredictor(alpha=alpha).fit(cal_scores)
        covered = sum(1 for s in test_scores if predictor.predict_inclusion(s))
        coverages.append(covered / n_test)

    mean_coverage = sum(coverages) / len(coverages)
    # Expected coverage under exchangeability is >= 1 - alpha = 0.90
    assert mean_coverage >= (1.0 - alpha - 0.02)


def test_conformal_risk_controller():
    # Candidate thresholds
    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5]
    # Synthetic monotonic losses
    # At lambda = 0.1, high loss; at lambda = 0.5, low loss
    sample_losses = [
        [0.40] * 50,
        [0.25] * 50,
        [0.10] * 50,
        [0.03] * 50,
        [0.01] * 50,
    ]

    controller = ConformalRiskController(target_risk=0.05, max_loss=1.0)
    controller.fit(lambdas, lambda lam: sample_losses[lambdas.index(lam)])
    assert controller.get_parameter() >= 0.4


def test_mondrian_conformal():
    scores = [0.1] * 20 + [0.8] * 20
    groups = ["easy"] * 20 + ["hard"] * 20
    mondrian = MondrianConformalPredictor(alpha=0.1).fit(scores, groups)

    easy_q = mondrian.get_group_quantile("easy")
    hard_q = mondrian.get_group_quantile("hard")
    assert easy_q < hard_q


def test_ks_distribution_shift():
    rng = random.Random(137)
    sample_a = [rng.gauss(0, 1) for _ in range(100)]
    sample_b = [rng.gauss(0, 1) for _ in range(100)]
    sample_shifted = [rng.gauss(2, 1) for _ in range(100)]

    detector = DistributionShiftDetector(significance_level=0.05)

    # Identical distributions -> no shift detected
    res_null = detector.test_shift(sample_a, sample_b)
    assert res_null["shifted"] is False

    # Shifted distributions -> shift detected (p < 0.05)
    res_shift = detector.test_shift(sample_a, sample_shifted)
    assert res_shift["shifted"] is True
    assert res_shift["p_value"] < 0.01
