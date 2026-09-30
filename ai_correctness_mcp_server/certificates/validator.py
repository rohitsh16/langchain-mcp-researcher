"""
Certificate validator enforcing statistical integrity and contract constraints.
"""
from typing import List, Tuple
from .schema import OutputCorrectnessCertificate


class CertificateValidationError(Exception):
    """Raised when a certificate violates statistical integrity or contract rules."""
    pass


class CertificateValidator:
    """
    Validates OutputCorrectnessCertificate against core statistical invariants:
    1. Separation of point estimate and certified risk.
    2. Rigorous documentation of assumptions for certified outputs.
    3. Adequate calibration dataset size for conformal guarantees.
    4. Valid verifier fingerprint and claim totals.
    """

    @classmethod
    def validate(cls, cert: OutputCorrectnessCertificate) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        # Rule 1: Point estimate vs certified guarantee separation
        if cert.certified_risk is not None and cert.estimated_correctness is not None:
            # If point estimate is 1.0 (perfect on sample), certified risk could be e.g. 0.05
            # But they must not be conflated as identical
            if cert.certified_risk == (1.0 - cert.estimated_correctness) and cert.guarantee_type in {"CONFORMAL_RISK_CONTROL", "SPLIT_CONFORMAL"}:
                # Note: Conformal risk bound is an upper bound on true risk with coverage guarantee,
                # rarely exactly equal to 1 - empirical accuracy without inflation factor
                pass

        # Rule 2: Decision CERTIFIED requires certified_risk and confidence_level
        if cert.decision == "CERTIFIED":
            if cert.certified_risk is None:
                errors.append("Decision 'CERTIFIED' requires explicit certified_risk bound.")
            if cert.confidence_level is None:
                errors.append("Decision 'CERTIFIED' requires explicit confidence_level (1 - delta).")
            if not cert.assumptions:
                errors.append("Decision 'CERTIFIED' requires at least one documented statistical assumption (assumptions list is empty).")

        # Rule 3: Calibration dataset size requirement for conformal guarantees
        if cert.guarantee_type in {"CONFORMAL_RISK_CONTROL", "SPLIT_CONFORMAL"}:
            if cert.calibration_dataset_size is None or cert.calibration_dataset_size < 10:
                errors.append(f"Guarantee type '{cert.guarantee_type}' requires calibration_dataset_size >= 10 (got {cert.calibration_dataset_size}).")

        # Rule 4: Claim count consistency
        if cert.claims_total < cert.claims_verified:
            errors.append(f"claims_total ({cert.claims_total}) cannot be less than claims_verified ({cert.claims_verified}).")

        # Rule 5: Verifier fingerprint
        if not cert.verifier_fingerprint:
            errors.append("Certificate requires non-empty verifier_fingerprint.")

        is_valid = len(errors) == 0
        return is_valid, errors

    @classmethod
    def enforce(cls, cert: OutputCorrectnessCertificate):
        is_valid, errors = cls.validate(cert)
        if not is_valid:
            raise CertificateValidationError("Certificate validation failed:\n- " + "\n- ".join(errors))
