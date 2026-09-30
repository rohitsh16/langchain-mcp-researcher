"""
Unit tests for Certificate generation and strict validation rules.
"""
import pytest
from ai_correctness_mcp_server.certificates.builder import CertificateBuilder
from ai_correctness_mcp_server.certificates.validator import (
    CertificateValidator,
    CertificateValidationError,
)
from ai_correctness_mcp_server.certificates.schema import OutputCorrectnessCertificate


def test_valid_certificate_build():
    builder = CertificateBuilder(
        target_query="What is 2+2?",
        target_output="2+2 is 4.",
    )
    builder.with_claims_summary(total=1, verified=1)
    builder.with_verifiers(["exact_verifier"])
    builder.with_dimensions(["FACTUAL_ACCURACY"])
    builder.with_conformal_guarantee(
        certified_risk=0.05,
        confidence_level=0.95,
        calibration_size=100,
        assumptions=["Data exchangeability"],
    )
    cert = builder.build()

    assert cert.decision == "CERTIFIED"
    assert cert.certified_risk == 0.05
    assert cert.confidence_level == 0.95
    assert cert.claims_total == 1
    assert cert.claims_verified == 1
    assert cert.verifier_fingerprint != ""

    is_valid, errors = CertificateValidator.validate(cert)
    assert is_valid is True
    assert len(errors) == 0


def test_certificate_validation_failure_missing_assumptions():
    cert = OutputCorrectnessCertificate(
        certificate_id="cert-bad-1",
        target_query="Query",
        target_output="Output",
        decision="CERTIFIED",
        guarantee_type="CONFORMAL_RISK_CONTROL",
        estimated_correctness=1.0,
        certified_risk=0.05,
        confidence_level=0.95,
        calibration_dataset_size=100,
        assumptions=[],  # Missing assumptions!
        claims_total=1,
        claims_verified=1,
        verifier_fingerprint="abc12345",
    )

    is_valid, errors = CertificateValidator.validate(cert)
    assert is_valid is False
    assert any("assumptions" in e for e in errors)

    with pytest.raises(CertificateValidationError):
        CertificateValidator.enforce(cert)


def test_certificate_validation_failure_small_calibration_sample():
    cert = OutputCorrectnessCertificate(
        certificate_id="cert-bad-2",
        target_query="Query",
        target_output="Output",
        decision="CERTIFIED",
        guarantee_type="CONFORMAL_RISK_CONTROL",
        estimated_correctness=1.0,
        certified_risk=0.05,
        confidence_level=0.95,
        calibration_dataset_size=5,  # Too small for distribution-free guarantee!
        assumptions=["Exchangeability"],
        claims_total=1,
        claims_verified=1,
        verifier_fingerprint="abc12345",
    )

    is_valid, errors = CertificateValidator.validate(cert)
    assert is_valid is False
    assert any("calibration_dataset_size" in e for e in errors)
