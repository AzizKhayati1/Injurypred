"""Thin MLflow wrapper.

Tracking must never break training: if MLflow is not installed or the server
is unreachable, calls become no-ops and a warning is logged.

* ``MLFLOW_TRACKING_URI`` set (Docker: ``http://mlflow:5000``) → remote server
* otherwise → local SQLite store ``mlflow.db`` (+ ``mlruns/`` artefacts)
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, int | float) and v == v:  # drop NaN
            out[key] = float(v)
    return out


class ExperimentTracker:
    def __init__(self, experiment: str, enabled: bool = True):
        self.enabled = enabled and os.environ.get("MLFLOW_DISABLED", "0") != "1"
        self.mlflow = None
        if not self.enabled:
            return
        try:
            import mlflow

            uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            self.mlflow = mlflow
            logger.info("MLflow tracking → %s (experiment '%s')", uri, experiment)
        except Exception as exc:
            logger.warning("MLflow unavailable, tracking disabled: %s", exc)
            self.enabled = False

    @contextlib.contextmanager
    def run(self, name: str, nested: bool = False, tags: dict | None = None):
        if not self.enabled:
            yield None
            return
        try:
            ctx = self.mlflow.start_run(run_name=name, nested=nested, tags=tags or {})
        except Exception as exc:
            logger.warning("Could not start MLflow run: %s", exc)
            yield None
            return
        with ctx as active:
            yield active

    def _safe(self, fn, *args, **kwargs):
        if not self.enabled:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.warning("MLflow call %s failed: %s", getattr(fn, "__name__", fn), exc)
            return None

    def log_params(self, params: dict) -> None:
        flat = {k: str(v)[:250] for k, v in _flatten_any(params).items()}
        self._safe(lambda: self.mlflow.log_params(flat))

    def log_metrics(self, metrics: dict, prefix: str = "") -> None:
        flat = _flatten(metrics, prefix)
        self._safe(lambda: self.mlflow.log_metrics(flat))

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if Path(path).exists():
            self._safe(lambda: self.mlflow.log_artifact(str(path), artifact_path))

    def set_tags(self, tags: dict) -> None:
        self._safe(lambda: self.mlflow.set_tags(tags))

    def log_model(self, bundle_path: str | Path, registered_name: str | None = None) -> None:
        """Log the bundle as an MLflow pyfunc model and register it (model versioning)."""
        if not self.enabled:
            return
        mlflow = self.mlflow

        class _PlayerLoadRiskPyfunc(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                import joblib

                self.model = joblib.load(context.artifacts["bundle"])["model"]

            def predict(self, context, model_input, params=None):
                return self.model.predict_proba(model_input)

        def _log():
            kwargs = dict(python_model=_PlayerLoadRiskPyfunc(), artifacts={"bundle": str(bundle_path)},
                          code_paths=[str(PROJECT_ROOT / "src")], registered_model_name=registered_name)
            try:
                return mlflow.pyfunc.log_model(name="model", **kwargs)
            except TypeError:
                return mlflow.pyfunc.log_model(artifact_path="model", **kwargs)

        self._safe(_log)

    @property
    def run_id(self) -> str | None:
        if not self.enabled:
            return None
        run = self.mlflow.active_run()
        return run.info.run_id if run else None


def _flatten_any(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_any(v, key + "."))
        else:
            out[key] = v
    return out
