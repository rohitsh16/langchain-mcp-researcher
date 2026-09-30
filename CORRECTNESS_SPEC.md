# Correctness Specification & Ontology

## 1. Correctness Dimensions

All experiments, verifiers, and claims must declare the primary dimension of correctness they measure:

1. **`FACTUAL_CORRECTNESS`**: Alignment of declared facts against consensus world knowledge or reference ground truth.
2. **`GROUNDING`**: Degree to which claims in the output are directly supported by citation spans in the provided context.
3. **`SEMANTIC_EQUIVALENCE`**: Semantic invariance across generation samples or answer paraphrases under bidirectional entailment.
4. **`LOGICAL_VALIDITY`**: Internal logical consistency, non-contradiction, and sound deductive steps.
5. **`MATHEMATICAL_VALIDITY`**: Correctness of mathematical steps, symbolic manipulations, and numerical outputs.
6. **`INSTRUCTION_FOLLOWING`**: Adherence to user-specified constraints, output schemas, and task instructions.
7. **`CODE_CORRECTNESS`**: Syntactic and semantic validity of generated source code, passing unit tests.
8. **`TEMPORAL_CONSISTENCY`**: Accuracy with respect to stated time periods, sequence of events, and causality.
9. **`SPATIAL_CONSISTENCY`**: Spatial relationships, 3D/2D geometry, and physical plausibility.
10. **`AUDIO_VISUAL_CONSISTENCY`**: Cross-modal alignment between audio, image, video, and textual descriptions.
11. **`PROCESS_CORRECTNESS`**: Step-level correctness in multi-step reasoning trajectories (e.g. Chain-of-Thought).
12. **`ROBUSTNESS`**: Stability of correctness against semantic paraphrases, noise, and adversarial prompting.
13. **`CALIBRATION`**: Agreement between estimated probability of correctness and empirical frequency of correctness.
14. **`CERTIFICATION`**: Mathematical or statistical risk control certifying residual error below $\alpha$.

## 2. Certificate Specification

A **Correctness Certificate** represents the verified output of the statistical certification pipeline.

### JSON Schema
```json
{
  "certificate_id": "string",
  "output_id": "string",
  "correctness_type": "string",
  "decision": "CERTIFY | ABSTAIN | REJECT | INSUFFICIENT_EVIDENCE",
  "estimated_correctness": 0.985,
  "certified_risk": 0.025,
  "confidence_level": 0.99,
  "assumptions": [
    "calibration_exchangeability",
    "bounded_loss_unit_interval"
  ],
  "verifier_results": ["vr-101", "vr-102"],
  "evidence_ids": ["ev-201", "ev-202"],
  "method": "split_conformal_risk_control",
  "valid_under": ["i.i.d.", "covariate_shift_bounded"],
  "verification_calls": 3,
  "verification_tokens": 420
}
```

### Invariants:
1. `estimated_correctness` $\in [0, 1]$ is an empirical point estimate.
2. `certified_risk` $\in [0, 1]$ is a statistical upper bound under confidence $1 - \delta$ ($\delta = 1 - \text{confidence\_level}$).
3. When `decision == "ABSTAIN"` or `"REJECT"`, `certified_risk` applies conditionally to accepted samples (selective risk control).
