import numpy as np

from src.monitoring.drift import psi, run_monitoring


def test_psi_identical_is_zero(rng):
    x = rng.normal(size=5000)
    assert psi(x, x) < 1e-3


def test_psi_detects_shift(rng):
    x = rng.normal(size=5000)
    assert psi(x, x + 1.0) > 0.25
    assert psi(x, rng.normal(size=5000)) < 0.1


def test_psi_handles_nan_and_constant():
    assert np.isnan(psi(np.array([np.nan]), np.array([1.0])))
    assert psi(np.zeros(100), np.zeros(100)) < 1e-3


def test_monitoring_report(service, tmp_path):
    rep = run_monitoring(service=service, out_dir=tmp_path)
    assert rep["summary"]["overall_status"] in ("ok", "warning", "alert")
    assert {"feature_drift", "prediction_drift", "data_quality"} <= set(rep)
    assert (tmp_path / "drift_report.html").exists()
