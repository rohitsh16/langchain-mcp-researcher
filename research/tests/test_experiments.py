"""
Unit tests for SQLite experiment registry and persistence.
"""
import pytest
import tempfile
import os
from research.experiments.registry import ExperimentRegistry


def test_sqlite_experiment_registry_lifecycle():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name

    try:
        reg = ExperimentRegistry(db_path=temp_db)
        reg.register_experiment(
            experiment_id="EXP-TEST-001",
            name="Testing conformal calibration",
            hypothesis="Conformal risk control guarantees 90% coverage.",
            protocol="EXP-001",
            metadata={"alpha": 0.10},
        )

        exp = reg.get_experiment("EXP-TEST-001")
        assert exp is not None
        assert exp.name == "Testing conformal calibration"

        # Record a run
        reg.record_run(
            experiment_id="EXP-TEST-001",
            run_id="run-42",
            parameters={"seed": 42, "q_hat": 0.12},
            metrics={"coverage": 0.91, "ece": 0.04},
        )

        runs = reg.get_runs_for_experiment("EXP-TEST-001")
        assert len(runs) == 1
        assert runs[0].run_id == "run-42"
        assert runs[0].config["seed"] == 42

        # Check recorded metrics
        metrics = reg.get_metrics_for_run("run-42")
        assert len(metrics) == 2
        metric_dict = {m["metric_name"]: m["metric_value"] for m in metrics}
        assert metric_dict["coverage"] == 0.91
        assert metric_dict["ece"] == 0.04
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)
