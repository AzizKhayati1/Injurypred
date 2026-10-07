"""Integration: the full training pipeline on a small synthetic dataset."""

import json
from pathlib import Path

import numpy as np

from src.models.registry import load_bundle


def test_training_produces_versioned_bundle(trained, small_cfg):
    d = Path(small_cfg["paths"]["model_dir"])
    for f in ("model.joblib", "metadata.json", "feature_schema.json", "reference_sample.parquet", "registry.json"):
        assert (d / f).exists(), f
    meta = json.loads((d / "metadata.json").read_text())
    for k in ("model_version", "build_id", "data_fingerprint", "config_hash", "metrics", "risk_engine", "features"):
        assert k in meta
    assert meta["data_origin"] == "synthetic"
    assert meta["model_type"] in meta["model_comparison"]


def test_model_beats_chance_and_is_calibrated(trained):
    test = trained["metrics"]["test"]
    assert test["roc_auc"] > 0.55
    assert abs(test["mean_predicted"] - test["base_rate"]) < 0.05
    for k in ("pr_auc", "brier", "ece", "precision", "recall", "f1", "top_k", "category_thresholds"):
        assert k in test


def test_bundle_predicts_probabilities(trained, small_cfg, service):
    b = load_bundle(small_cfg["paths"]["model_dir"])
    X = service.features[service.features["eligible"]].head(200)
    p = b["model"].predict_proba(X)
    assert p.shape == (200,) and np.all((p >= 0) & (p <= 1))


def test_reproducible_training(small_cfg, small_ds, trained):
    from training.train import run_training

    again = run_training(small_cfg, ds=small_ds, evaluate_after=False, track=False)
    assert again["model_type"] == trained["model_type"]
    assert again["metrics"]["test"]["roc_auc"] == trained["metrics"]["test"]["roc_auc"]
