"""
Unit tests for JSON-RPC 2.0 MCP server handlers and tool execution.
"""
import pytest
from ai_correctness_mcp_server.server import handle_tool_call, TOOLS


def test_tools_list_completeness():
    tool_names = {t["name"] for t in TOOLS}
    expected_tools = {
        "extract_claims",
        "verify_claim",
        "estimate_semantic_uncertainty",
        "calibrate_scores",
        "selective_predict",
        "conformal_calibrate",
        "conformal_risk_control",
        "test_distribution_shift",
        "sequential_verify",
        "evaluate_verifier",
        "generate_certificate",
        "create_experiment",
        "get_experiment",
    }
    assert expected_tools.issubset(tool_names)


def test_extract_claims_tool():
    res = handle_tool_call("extract_claims", {"text": "Python was created by Guido van Rossum in 1991."})
    assert res["count"] >= 1
    assert "Python was created by Guido van Rossum in 1991" in [c["text"] for c in res["claims"]]


def test_verify_claim_tool():
    res = handle_tool_call(
        "verify_claim",
        {
            "claim": "Python was created by Guido van Rossum in 1991",
            "context": "Guido van Rossum released Python in 1991.",
            "verifier_type": "evidence",
        },
    )
    assert res["is_verified"] is True
    assert res["confidence"] >= 0.5


def test_conformal_calibrate_tool():
    res = handle_tool_call(
        "conformal_calibrate",
        {
            "nonconformity_scores": [0.05, 0.10, 0.15, 0.20, 0.25],
            "alpha": 0.2,
        },
    )
    assert "quantile" in res
    assert res["quantile"] > 0.0


def test_selective_predict_tool():
    res = handle_tool_call("selective_predict", {"confidence": 0.95, "threshold": 0.80})
    assert res["accepted"] is True
    assert res["action"] == "ACCEPT"


def test_generate_certificate_tool():
    res = handle_tool_call(
        "generate_certificate",
        {
            "target_query": "Explain Ohm's law.",
            "target_output": "V = I * R.",
            "claims_total": 1,
            "claims_verified": 1,
            "certified_risk": 0.05,
            "confidence_level": 0.95,
            "calibration_size": 50,
            "assumptions": ["IID electrical law queries"],
        },
    )
    assert res["decision"] == "CERTIFIED"
    assert res["certified_risk"] == 0.05
