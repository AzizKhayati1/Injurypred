"""Local model registry: versioned artefacts on disk (mirrored to MLflow when available).

``models/``
  ├── model.joblib            fitted RiskModel + explainer background (the served model)
  ├── metadata.json           version, training data fingerprint, config hash, git commit,
  │                           metrics, risk-engine thresholds, calibration table
  ├── feature_schema.json     ordered features with dtype/description/training ranges
  ├── reference_sample.parquet  training-period feature/prediction sample for drift monitoring
  └── registry.json           history of every trained version
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
SCHEMA_FILE = "feature_schema.json"
REFERENCE_FILE = "reference_sample.parquet"
REGISTRY_FILE = "registry.json"


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip() or None
    except Exception:
        return None


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def save_bundle(model_dir: str | Path, model, background: pd.DataFrame, metadata: dict,
                feature_schema: dict, reference: pd.DataFrame) -> Path:
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "background": background}, d / MODEL_FILE, compress=3)
    (d / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str))
    (d / SCHEMA_FILE).write_text(json.dumps(feature_schema, indent=2, default=str))
    reference.to_parquet(d / REFERENCE_FILE)
    reg_path = d / REGISTRY_FILE
    history = json.loads(reg_path.read_text()) if reg_path.exists() else []
    history.append({k: metadata.get(k) for k in ("model_version", "build_id", "trained_at", "model_type",
                                                 "data_fingerprint", "config_hash", "git_commit",
                                                 "mlflow_run_id")}
                   | {"test_metrics": {k: metadata.get("metrics", {}).get("test", {}).get(k)
                                       for k in ("roc_auc", "pr_auc", "brier")}})
    reg_path.write_text(json.dumps(history[-50:], indent=2, default=str))
    logger.info("Saved model bundle %s to %s", metadata.get("build_id"), d)
    return d


def load_bundle(model_dir: str | Path) -> dict:
    d = Path(model_dir)
    if not (d / MODEL_FILE).exists():
        raise FileNotFoundError(f"no trained model in {d} – run `make train` first")
    payload = joblib.load(d / MODEL_FILE)
    payload["metadata"] = json.loads((d / METADATA_FILE).read_text())
    payload["feature_schema"] = json.loads((d / SCHEMA_FILE).read_text())
    ref = d / REFERENCE_FILE
    payload["reference"] = pd.read_parquet(ref) if ref.exists() else None
    return payload


def feature_schema(features: list[str], train: pd.DataFrame) -> dict:
    """Ordered feature list with training-period ranges (used for input validation & drift)."""
    from src.features.catalog import FEATURE_SPECS

    desc = train[features].describe(percentiles=[0.01, 0.5, 0.99]).T
    out = []
    for f in features:
        spec = FEATURE_SPECS.get(f)
        out.append({
            "name": f, "dtype": "float64", "group": spec.group if spec else None,
            "unit": spec.unit if spec else None, "description": spec.description if spec else None,
            "train_p01": float(desc.loc[f, "1%"]) if pd.notna(desc.loc[f, "1%"]) else None,
            "train_median": float(desc.loc[f, "50%"]) if pd.notna(desc.loc[f, "50%"]) else None,
            "train_p99": float(desc.loc[f, "99%"]) if pd.notna(desc.loc[f, "99%"]) else None,
            "train_missing_rate": float(train[f].isna().mean()),
        })
    return {"n_features": len(features), "features": out}
