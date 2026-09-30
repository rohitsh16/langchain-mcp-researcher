"""
SQLite-backed Experiment Registry for reproducible research runs, metrics, and certificates.
"""
from datetime import datetime, timezone
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ExperimentRecord(BaseModel):
    experiment_id: str
    name: str
    hypothesis: str
    code_commit: str
    created_at: str


class ExperimentRunRecord(BaseModel):
    run_id: str
    experiment_id: str
    seed: int
    model_id: str
    config: Dict[str, Any]
    status: str
    created_at: str


class ExperimentRegistry:
    """Manages immutable experiment lineage, metadata, metrics, and generated certificates."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(os.getcwd(), "data", "experiments.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    hypothesis TEXT NOT NULL,
                    code_commit TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    metric_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    ci_lower REAL,
                    ci_upper REAL,
                    metadata_json TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS certificates (
                    certificate_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    output_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    estimated_correctness REAL,
                    certified_risk REAL,
                    confidence_level REAL,
                    method TEXT NOT NULL,
                    certificate_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                """
            )

    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        hypothesis: str,
        code_commit: str = "HEAD",
    ) -> ExperimentRecord:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments (experiment_id, name, hypothesis, code_commit, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (experiment_id, name, hypothesis, code_commit, now),
            )
        return ExperimentRecord(
            experiment_id=experiment_id,
            name=name,
            hypothesis=hypothesis,
            code_commit=code_commit,
            created_at=now,
        )

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            if not row:
                return None
            return ExperimentRecord(
                experiment_id=row["experiment_id"],
                name=row["name"],
                hypothesis=row["hypothesis"],
                code_commit=row["code_commit"],
                created_at=row["created_at"],
            )

    def register_experiment(
        self,
        experiment_id: str,
        name: str,
        hypothesis: str,
        protocol: str = "EXP-001",
        code_commit: str = "HEAD",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExperimentRecord:
        return self.create_experiment(
            experiment_id=experiment_id,
            name=name,
            hypothesis=hypothesis,
            code_commit=code_commit,
        )

    def record_run(
        self,
        run_id: str,
        experiment_id: str,
        seed: int = 42,
        model_id: str = "default_model",
        config: Optional[Dict[str, Any]] = None,
        status: str = "COMPLETED",
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ExperimentRunRecord:
        now = datetime.now(timezone.utc).isoformat()
        run_config = config or parameters or {}
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (run_id, experiment_id, seed, model_id, config_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, experiment_id, seed, model_id, json.dumps(run_config), status, now),
            )
        if metrics:
            for m_name, m_val in metrics.items():
                self.record_metric(run_id=run_id, name=m_name, value=float(m_val))

        return ExperimentRunRecord(
            run_id=run_id,
            experiment_id=experiment_id,
            seed=seed,
            model_id=model_id,
            config=run_config,
            status=status,
            created_at=now,
        )

    def record_metric(
        self,
        run_id: str,
        name: str,
        value: float,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        import uuid

        m_id = f"m-{uuid.uuid4().hex[:10]}"
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO metrics (metric_id, run_id, metric_name, metric_value, ci_lower, ci_upper, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (m_id, run_id, name, float(value), ci_lower, ci_upper, json.dumps(metadata or {})),
            )

    def get_metrics_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM metrics WHERE run_id = ?", (run_id,)).fetchall()
            return [
                {
                    "metric_name": r["metric_name"],
                    "metric_value": r["metric_value"],
                    "ci_lower": r["ci_lower"],
                    "ci_upper": r["ci_upper"],
                    "metadata": json.loads(r["metadata_json"] or "{}"),
                }
                for r in rows
            ]

    def get_runs_for_experiment(self, experiment_id: str) -> List[ExperimentRunRecord]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM runs WHERE experiment_id = ?", (experiment_id,)).fetchall()
            return [
                ExperimentRunRecord(
                    run_id=r["run_id"],
                    experiment_id=r["experiment_id"],
                    seed=r["seed"],
                    model_id=r["model_id"],
                    config=json.loads(r["config_json"] or "{}"),
                    status=r["status"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    get_experiment_runs = get_runs_for_experiment

    def record_certificate(self, certificate: Any, run_id: Optional[str] = None):
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO certificates (
                    certificate_id, run_id, output_id, decision,
                    estimated_correctness, certified_risk, confidence_level, method, certificate_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    certificate.certificate_id,
                    run_id,
                    certificate.output_id,
                    certificate.decision.value if hasattr(certificate.decision, "value") else str(certificate.decision),
                    certificate.estimated_correctness,
                    certificate.certified_risk,
                    certificate.confidence_level,
                    certificate.method,
                    certificate.model_dump_json() if hasattr(certificate, "model_dump_json") else json.dumps(certificate),
                ),
            )
