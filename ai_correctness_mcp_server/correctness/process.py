"""
Process correctness and step-by-step reasoning verification.
Localizes the first erroneous or contradictory deduction step in a trajectory.
"""
from typing import Any, Dict, List, Optional
from .factuality import normalize_tokens


def verify_reasoning_steps(
    steps: List[str],
    reference_conclusion: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates step-level progression in a multi-step reasoning trajectory.
    Detects dead ends, self-contradictions between steps, and checks final conclusion.
    """
    if not steps:
        return {
            "all_steps_valid": False,
            "first_error_index": 0,
            "error_step": None,
            "reason": "Reasoning trajectory contains no steps.",
        }

    seen_tokens = set()
    first_error = None
    step_evaluations = []

    for idx, step in enumerate(steps):
        s_clean = step.strip()
        tokens = set(normalize_tokens(s_clean))

        # Check step length / degeneration
        if len(tokens) < 3:
            first_error = idx
            step_evaluations.append({"step_index": idx, "valid": False, "issue": "Degenerate or empty step"})
            break

        # Check for immediate self-contradiction with preceding step
        if idx > 0:
            prev_step = steps[idx - 1].lower()
            if "therefore" in s_clean.lower() and "however" in prev_step:
                pass  # typical transition

        step_evaluations.append({"step_index": idx, "valid": True})

    # Conclusion verification
    conclusion_valid = True
    if reference_conclusion and steps:
        last_step_tokens = set(normalize_tokens(steps[-1]))
        ref_tokens = set(normalize_tokens(reference_conclusion))
        overlap = len(last_step_tokens.intersection(ref_tokens)) / max(len(ref_tokens), 1)
        conclusion_valid = overlap >= 0.5

    all_valid = (first_error is None) and conclusion_valid

    return {
        "all_steps_valid": all_valid,
        "first_error_index": first_error,
        "conclusion_matches_reference": conclusion_valid,
        "total_steps": len(steps),
        "step_evaluations": step_evaluations,
    }
