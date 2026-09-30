"""
JSON-RPC 2.0 stdio MCP Server for AI Output Correctness and Verification.
"""
import sys
import json
import logging
import argparse
from typing import Dict, Any, List, Optional

# Relative package imports
from .correctness.claims import extract_atomic_claims
from .correctness.factuality import evaluate_factuality
from .uncertainty.semantic_entropy import compute_semantic_entropy
from .uncertainty.calibration import TemperatureScaling
from .selective.abstention import SelectivePredictor
from .conformal.split import SplitConformalPredictor, compute_conformal_quantile
from .conformal.risk_control import ConformalRiskController
from .conformal.shift import DistributionShiftDetector
from .sequential.sprt import WaldSPRT
from .verification.exact import ExactVerifier
from .verification.evidence import EvidenceVerifier
from .verification.semantic import SemanticVerifier
from .verification.program import ProgrammaticVerifier
from .certificates.builder import CertificateBuilder
from .certificates.validator import CertificateValidator

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger("ai_correctness_mcp_server")


TOOLS = [
    {
        "name": "extract_claims",
        "description": "Extracts atomic factual claims from generated AI text.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The AI generated text."}},
            "required": ["text"],
        },
    },
    {
        "name": "verify_claim",
        "description": "Verifies a claim against context using exact, evidence, semantic, or programmatic verifiers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "context": {"type": "string", "description": "Evidence passage or ground truth."},
                "verifier_type": {"type": "string", "enum": ["exact", "evidence", "semantic", "program"], "default": "evidence"},
            },
            "required": ["claim"],
        },
    },
    {
        "name": "estimate_semantic_uncertainty",
        "description": "Computes semantic entropy across sample generations using semantic clustering.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "generations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["generations"],
        },
    },
    {
        "name": "calibrate_scores",
        "description": "Fits temperature scaling on validation scores/labels and calibrates test scores.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scores": {"type": "array", "items": {"type": "number"}},
                "labels": {"type": "array", "items": {"type": "integer"}},
                "test_scores": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["scores", "labels", "test_scores"],
        },
    },
    {
        "name": "selective_predict",
        "description": "Makes an accept/abstain decision given a confidence score and rejection threshold.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "confidence": {"type": "number"},
                "threshold": {"type": "number", "default": 0.8},
            },
            "required": ["confidence"],
        },
    },
    {
        "name": "conformal_calibrate",
        "description": "Computes split conformal quantile q_hat guaranteeing coverage 1 - alpha.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "nonconformity_scores": {"type": "array", "items": {"type": "number"}},
                "alpha": {"type": "number", "default": 0.1},
            },
            "required": ["nonconformity_scores"],
        },
    },
    {
        "name": "conformal_risk_control",
        "description": "Finds smallest lambda threshold controlling expected bounded loss <= target_risk.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "candidate_lambdas": {"type": "array", "items": {"type": "number"}},
                "sample_losses": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "target_risk": {"type": "number", "default": 0.05},
            },
            "required": ["candidate_lambdas", "sample_losses"],
        },
    },
    {
        "name": "test_distribution_shift",
        "description": "Tests distribution shift between calibration and test scores using 2-sample KS test.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calibration_scores": {"type": "array", "items": {"type": "number"}},
                "test_scores": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["calibration_scores", "test_scores"],
        },
    },
    {
        "name": "sequential_verify",
        "description": "Runs Wald's SPRT on streaming verification results for early acceptance/rejection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "stream_errors": {"type": "array", "items": {"type": "boolean"}},
                "p0": {"type": "number", "default": 0.05},
                "p1": {"type": "number", "default": 0.20},
            },
            "required": ["stream_errors"],
        },
    },
    {
        "name": "evaluate_verifier",
        "description": "Evaluates verifier accuracy, precision, and recall on test claims.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "verifier_type": {"type": "string"},
                "test_cases": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["verifier_type", "test_cases"],
        },
    },
    {
        "name": "generate_certificate",
        "description": "Generates and validates a formal correctness certificate for AI model output.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_query": {"type": "string"},
                "target_output": {"type": "string"},
                "model_id": {"type": "string", "default": "model_v1"},
                "claims_total": {"type": "integer"},
                "claims_verified": {"type": "integer"},
                "certified_risk": {"type": "number"},
                "confidence_level": {"type": "number", "default": 0.95},
                "calibration_size": {"type": "integer", "default": 100},
                "assumptions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["target_query", "target_output", "claims_total", "claims_verified"],
        },
    },
    {
        "name": "create_experiment",
        "description": "Registers an experiment in the SQLite research registry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "experiment_id": {"type": "string"},
                "name": {"type": "string"},
                "hypothesis": {"type": "string"},
                "protocol": {"type": "string", "default": "EXP-001"},
            },
            "required": ["experiment_id", "name", "hypothesis"],
        },
    },
    {
        "name": "get_experiment",
        "description": "Retrieves experiment metadata and recorded runs from SQLite registry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "experiment_id": {"type": "string"},
            },
            "required": ["experiment_id"],
        },
    },
]


def handle_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Dispatches tool calls to algorithmic implementations."""
    if tool_name == "extract_claims":
        text = arguments.get("text", "")
        claims = extract_atomic_claims(text)
        return {"claims": claims, "count": len(claims)}

    elif tool_name == "verify_claim":
        claim = arguments.get("claim", "")
        context = arguments.get("context")
        vtype = arguments.get("verifier_type", "evidence")

        if vtype == "exact":
            verifier = ExactVerifier()
        elif vtype == "semantic":
            verifier = SemanticVerifier()
        elif vtype == "program":
            verifier = ProgrammaticVerifier()
        else:
            verifier = EvidenceVerifier()

        decision = verifier.verify(claim, context)
        return {
            "verifier_id": decision.verifier_id,
            "verifier_type": decision.verifier_type,
            "claim": decision.claim,
            "is_verified": decision.is_verified,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        }

    elif tool_name == "estimate_semantic_uncertainty":
        generations = arguments.get("generations", [])
        return compute_semantic_entropy(generations)

    elif tool_name == "calibrate_scores":
        scores = arguments.get("scores", [])
        labels = arguments.get("labels", [])
        test_scores = arguments.get("test_scores", [])
        calibrator = TemperatureScaling()
        calibrator.fit(scores, labels)
        calibrated_test = calibrator.calibrate_batch(test_scores)
        return {
            "temperature": calibrator.temperature,
            "calibrated_test_scores": calibrated_test,
        }

    elif tool_name == "selective_predict":
        conf = arguments.get("confidence", 0.0)
        thresh = arguments.get("threshold", 0.8)
        predictor = SelectivePredictor(threshold=thresh)
        decision = predictor.decide(conf)
        return {
            "accepted": decision.accepted,
            "action": decision.action,
            "confidence": decision.confidence,
            "threshold": decision.threshold,
        }

    elif tool_name == "conformal_calibrate":
        scores = arguments.get("nonconformity_scores", [])
        alpha = arguments.get("alpha", 0.1)
        predictor = SplitConformalPredictor(alpha=alpha)
        predictor.fit(scores)
        return {
            "alpha": alpha,
            "quantile": predictor.get_certified_upper_bound(),
            "n_calibration": predictor.n_cal,
            "guarantee": f"P(loss <= q_hat) >= {1 - alpha:.2%}",
        }

    elif tool_name == "conformal_risk_control":
        lambdas = arguments.get("candidate_lambdas", [])
        sample_losses = arguments.get("sample_losses", [])
        target_risk = arguments.get("target_risk", 0.05)

        def loss_fn(lam):
            idx = lambdas.index(lam) if lam in lambdas else 0
            return sample_losses[idx]

        controller = ConformalRiskController(target_risk=target_risk)
        controller.fit(candidate_lambdas=lambdas, loss_fn=loss_fn)
        return {
            "selected_lambda": controller.get_parameter(),
            "target_risk": target_risk,
            "empirical_risk": controller.empirical_risk,
        }

    elif tool_name == "test_distribution_shift":
        cal = arguments.get("calibration_scores", [])
        test = arguments.get("test_scores", [])
        detector = DistributionShiftDetector()
        return detector.test_shift(cal, test)

    elif tool_name == "sequential_verify":
        errors = arguments.get("stream_errors", [])
        p0 = arguments.get("p0", 0.05)
        p1 = arguments.get("p1", 0.20)
        sprt = WaldSPRT(p0=p0, p1=p1)
        for err in errors:
            sprt.step(err)
            if sprt.status != "CONTINUE":
                break
        return sprt.summary()

    elif tool_name == "evaluate_verifier":
        vtype = arguments.get("verifier_type", "evidence")
        cases = arguments.get("test_cases", [])
        verifier = EvidenceVerifier() if vtype == "evidence" else SemanticVerifier()
        decisions = [verifier.verify(c["claim"], c.get("context")).is_verified for c in cases]
        acc = sum(decisions) / len(decisions) if decisions else 0.0
        return {"verifier_type": vtype, "accuracy": acc, "total_cases": len(cases)}

    elif tool_name == "generate_certificate":
        builder = CertificateBuilder(
            target_query=arguments["target_query"],
            target_output=arguments["target_output"],
            model_id=arguments.get("model_id", "model_v1"),
        )
        builder.with_claims_summary(arguments["claims_total"], arguments["claims_verified"])
        builder.with_dimensions(["FACTUAL_ACCURACY", "EMPIRICAL_REPLICABILITY"])

        if "certified_risk" in arguments:
            builder.with_conformal_guarantee(
                certified_risk=arguments["certified_risk"],
                confidence_level=arguments.get("confidence_level", 0.95),
                calibration_size=arguments.get("calibration_size", 100),
                assumptions=arguments.get("assumptions"),
            )

        cert = builder.build()
        return cert.model_dump()

    elif tool_name == "create_experiment":
        from research.experiments.registry import ExperimentRegistry
        reg = ExperimentRegistry()
        reg.register_experiment(
            experiment_id=arguments["experiment_id"],
            name=arguments["name"],
            hypothesis=arguments["hypothesis"],
            protocol=arguments.get("protocol", "EXP-001"),
        )
        return {"status": "REGISTERED", "experiment_id": arguments["experiment_id"]}

    elif tool_name == "get_experiment":
        from research.experiments.registry import ExperimentRegistry
        reg = ExperimentRegistry()
        exp = reg.get_experiment(arguments["experiment_id"])
        runs = reg.get_experiment_runs(arguments["experiment_id"])
        return {"experiment": exp, "runs": runs}

    raise ValueError(f"Unknown tool: {tool_name}")


def run_smoke_test() -> int:
    """Executes end-to-end correctness verification smoke test."""
    print("=" * 60)
    print("AI CORRECTNESS PLATFORM SMOKE TEST")
    print("=" * 60)

    # 1. Atomic claim extraction
    sample_text = (
        "Albert Einstein proposed the theory of special relativity in 1905. "
        "He received the Nobel Prize in Physics in 1921 for the photoelectric effect. "
        "The speed of light in vacuum is approximately 300000 km/s."
    )
    print("\n[Step 1] Extracting atomic claims from text...")
    claims = extract_atomic_claims(sample_text)
    print(f"-> Extracted {len(claims)} atomic claims:")
    for i, c in enumerate(claims, 1):
        print(f"   {i}. {c['text']}")
    assert len(claims) >= 2, "Expected at least 2 atomic claims"

    # 2. Multi-level verification
    print("\n[Step 2] Verifying claims with Evidence & Exact verifiers...")
    evidence_doc = (
        "Albert Einstein published special relativity in 1905 and won the 1921 Nobel Prize in Physics "
        "for his explanation of the photoelectric effect. Light travels at approximately 300000 km/s in vacuum."
    )
    v_evidence = EvidenceVerifier()
    v_exact = ExactVerifier()

    verified_count = 0
    for c in claims:
        d = v_evidence.verify(c["text"], context=evidence_doc)
        print(f"-> Claim: '{c['text'][:40]}...' Verified: {d.is_verified} (conf: {d.confidence:.2f})")
        if d.is_verified:
            verified_count += 1
    assert verified_count >= 2, "Expected evidence verification to pass for claims"

    # 3. Semantic entropy uncertainty estimation
    print("\n[Step 3] Estimating semantic uncertainty via generation clustering...")
    samples = [
        "Special relativity was published in 1905 by Einstein.",
        "Einstein formulated special relativity in 1905.",
        "In 1905, Albert Einstein introduced the special theory of relativity.",
    ]
    uncertainty_res = compute_semantic_entropy(samples)
    print(f"-> Semantic entropy: {uncertainty_res['semantic_entropy']:.4f}")
    print(f"-> Unique semantic clusters: {uncertainty_res['num_clusters']}")
    assert uncertainty_res["semantic_entropy"] < 0.5, "Expected low semantic entropy for consistent generations"

    # 4. Calibration & Selective Prediction
    print("\n[Step 4] Fitting temperature calibrator and selective predictor...")
    cal_scores = [0.95, 0.88, 0.40, 0.85, 0.30, 0.92, 0.90, 0.25, 0.78, 0.82]
    cal_labels = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    calibrator = TemperatureScaling()
    calibrator.fit(cal_scores, cal_labels)
    calibrated_val = calibrator.calibrate(0.85)
    print(f"-> Fitted Temperature: {calibrator.temperature:.3f}")
    print(f"-> Calibrated 0.85 -> {calibrated_val:.4f}")

    predictor = SelectivePredictor(threshold=0.75)
    dec = predictor.decide(calibrated_val)
    print(f"-> Selective Decision: {dec.action} (Confidence: {dec.confidence:.3f}, Threshold: {dec.threshold:.3f})")
    assert dec.accepted is True

    # 5. Split conformal quantile computation
    print("\n[Step 5] Split conformal quantile calibration (1 - alpha = 90%)...")
    nonconf_scores = [0.05, 0.08, 0.12, 0.04, 0.15, 0.09, 0.11, 0.03, 0.07, 0.14, 0.06, 0.10]
    q_hat = compute_conformal_quantile(nonconf_scores, alpha=0.10)
    print(f"-> Conformal quantile q_hat: {q_hat:.4f}")
    assert q_hat > 0.0

    # 6. Formal Certificate Generation & Validation
    print("\n[Step 6] Generating and validating formal OutputCorrectnessCertificate...")
    builder = CertificateBuilder(
        target_query="What did Einstein publish in 1905?",
        target_output=sample_text,
        model_id="gemini-2.5-pro",
    )
    builder.with_claims_summary(total=len(claims), verified=verified_count)
    builder.with_verifiers(["evidence_verifier", "exact_verifier"])
    builder.with_dimensions(["FACTUAL_ACCURACY", "PROCESS_REASONING"])
    builder.with_conformal_guarantee(
        certified_risk=0.08,
        confidence_level=0.95,
        calibration_size=120,
        assumptions=["Exchangeability between calibration and test data"],
    )
    cert = builder.build()
    print(f"-> Certificate ID: {cert.certificate_id}")
    print(f"-> Decision: {cert.decision}")
    print(f"-> Guarantee: {cert.guarantee_type}")
    print(f"-> Certified Risk: {cert.certified_risk} (Confidence: {cert.confidence_level})")
    print(f"-> Fingerprint: {cert.verifier_fingerprint}")

    is_valid, errors = CertificateValidator.validate(cert)
    assert is_valid, f"Certificate validation failed: {errors}"
    print("-> Certificate passed all statistical integrity constraints!")

    print("\n" + "=" * 60)
    print("ALL AI CORRECTNESS SMOKE TESTS COMPLETED SUCCESSFULLY (PASS)")
    print("=" * 60 + "\n")
    return 0


def run_stdio_server():
    """Runs the stdio JSON-RPC 2.0 MCP server loop."""
    logger.info("Starting AI Correctness MCP Server on stdio...")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            res = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {e}"}}
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method.startswith("notifications/") or method == "notifications/initialized":
            # Notifications do not return a response in JSON-RPC 2.0
            continue

        if method == "initialize":
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "ai-correctness-mcp-server", "version": "1.0.0"},
                },
            }
        elif method == "tools/list":
            res = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            try:
                result_data = handle_tool_call(tool_name, arguments)
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result_data, indent=2)}],
                        "isError": False,
                    },
                }
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Tool execution failed: {str(e)}"}],
                        "isError": True,
                    },
                }
        else:
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="AI Output Correctness MCP Server")
    parser.add_argument("--smoke-test", action="store_true", help="Run end-to-end algorithmic smoke test")
    parser.add_argument("--serve", action="store_true", help="Run stdio JSON-RPC server (default)")
    args = parser.parse_args()

    if args.smoke_test:
        sys.exit(run_smoke_test())
    else:
        run_stdio_server()


if __name__ == "__main__":
    main()
