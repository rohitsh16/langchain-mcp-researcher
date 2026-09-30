"""
Unit tests for sequential verification, Wald's SPRT, and confidence sequences.
"""
import pytest
from ai_correctness_mcp_server.sequential.sprt import WaldSPRT
from ai_correctness_mcp_server.sequential.confidence_sequences import EmpiricalBernsteinSequence
from ai_correctness_mcp_server.sequential.evalues import EValueMartingale


def test_wald_sprt_early_accept_h0():
    # Stream of all correct claims (errors = False) should quickly accept H0 (low error)
    sprt = WaldSPRT(p0=0.05, p1=0.20)
    for _ in range(30):
        res = sprt.step(is_error=False)
        if res["status"] != "CONTINUE":
            break
    assert res["status"] == "ACCEPT_H0"
    assert res["step_count"] <= 30


def test_wald_sprt_early_accept_h1():
    # Stream of high errors should quickly trigger H1 (reject system)
    sprt = WaldSPRT(p0=0.05, p1=0.20)
    for _ in range(30):
        res = sprt.step(is_error=True)
        if res["status"] != "CONTINUE":
            break
    assert res["status"] == "ACCEPT_H1"


def test_empirical_bernstein_sequence():
    seq = EmpiricalBernsteinSequence(alpha=0.05)
    bounds = None
    for _ in range(50):
        bounds = seq.step(0.8)

    assert bounds is not None
    assert bounds["t"] == 50
    assert 0.7 <= bounds["mean"] <= 0.9
    assert bounds["lower"] <= bounds["mean"] <= bounds["upper"]


def test_evalue_martingale():
    martingale = EValueMartingale(alpha=0.05, h0_rate=0.05)
    # Consecutive errors should drive martingale value above 1/alpha = 20
    for _ in range(25):
        martingale.step(is_error=True)
        if martingale.rejected:
            break
    assert martingale.rejected is True
    assert martingale.martingale_value >= martingale.threshold
