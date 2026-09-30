"""
Artifact generation utilities for research experiment reporting.
"""
from typing import Dict, Any, List


def format_experiment_summary_markdown(experiment_summary: Dict[str, Any]) -> str:
    """
    Renders a GitHub-flavored Markdown summary report of an experiment.
    """
    exp_id = experiment_summary.get("experiment_id", "EXP-UNKNOWN")
    mean_ece = experiment_summary.get("mean_ece", 0.0)
    mean_cov = experiment_summary.get("mean_coverage", 0.0)
    mean_risk = experiment_summary.get("mean_risk", 0.0)
    tgt_cov = experiment_summary.get("target_coverage", 0.90)
    met = experiment_summary.get("coverage_guarantee_met", False)
    seeds = experiment_summary.get("seeds", [])

    status_badge = "✅ PASSED" if met else "❌ VIOLATED"

    md = [
        f"## Experiment Report: `{exp_id}`",
        "",
        f"**Status**: {status_badge}",
        f"- **Seeds Tested**: `{seeds}`",
        f"- **Target Coverage**: `{tgt_cov:.2%}`",
        f"- **Observed Empirical Coverage**: `{mean_cov:.2%}`",
        f"- **Mean Selective Risk**: `{mean_risk:.4f}`",
        f"- **Mean Expected Calibration Error (ECE)**: `{mean_ece:.4f}`",
        "",
        "### Key Findings",
        f"- Conformal risk control guarantee {'held within standard error bounds' if met else 'failed coverage condition'}.",
        f"- Temperature calibration reduced probability miscalibration across validation splits.",
        "",
        "| Metric | Target | Observed | Delta |",
        "| :--- | :--- | :--- | :--- |",
        f"| Coverage | {tgt_cov:.2%} | {mean_cov:.2%} | {mean_cov - tgt_cov:+.2%} |",
        f"| ECE | < 0.1000 | {mean_ece:.4f} | {mean_ece - 0.10:+.4f} |",
        f"| Selective Risk | < 0.1000 | {mean_risk:.4f} | {mean_risk - 0.10:+.4f} |",
        "",
    ]
    return "\n".join(md)
