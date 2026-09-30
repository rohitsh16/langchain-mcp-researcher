# AI Output Correctness Research Charter

## 1. Mission & Research North Star

> **Can we mathematically bound, estimate, and eventually certify the residual probability that a generated AI output is incorrect, under explicit assumptions?**

This research program establishes rigorous statistical, algorithmic, and provenance infrastructure for verifying, calibrating, and certifying AI model generations.

## 2. Core Tenets

1. **Separation of Estimation and Guarantee**:
   An empirical prediction score or LLM self-evaluation is never silently promoted to a formal or statistical guarantee. Guarantees must explicitly state their mathematical assumptions (e.g., exchangeability, bounded loss, independent verification).
2. **Provenance is Mandatory**:
   Every claim, theorem, and empirical finding must trace back to concrete evidence: an immutable paper span, a theorem formulation, or a registered experiment ID.
3. **No Uncalibrated Decisions**:
   Probabilistic confidence values must be empirically calibrated. When an AI system asserts 90% confidence, the observed empirical correctness rate must demonstrably approach 90%.
4. **Adversarial Resilience**:
   Any verifier, calibration method, or certificate generation pipeline must be subjected to correlated hallucination, judge exploitation, and distribution shift attacks before claiming validity.
5. **Compute-Aware Adaptive Verification**:
   Verification is a resource-constrained decision process. The system aims to minimize verification compute (calls, tokens, latency) while maintaining target certified risk.

## 3. Scope and Research Progression

- **Level 1 — Text Factuality & Grounding**: Atomic claim extraction, evidence alignment, and semantic entropy.
- **Level 2 — Calibration & Selective Prediction**: Reliability diagrams, ECE minimization, and risk-coverage optimization.
- **Level 3 — Conformal Prediction & Risk Control**: Finite-sample risk bounds $E[L] \le \alpha$ under exchangeability.
- **Level 4 — Sequential & Adaptive Verification**: Wald's SPRT, time-uniform confidence sequences, and early-stopping rules.
- **Level 5 — Distribution Shift & Robustness**: Domain, style, and adversarial distribution shift testing.
- **Level 6 — Process & Formal Reasoning**: Step-by-step reasoning verification and formal proof integration.
- **Level 7 — Multimodal Consistency**: Cross-modal contradiction detection and modality-independent risk certification.
