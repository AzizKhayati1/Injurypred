"""Shared fixtures: a small synthetic dataset and a quickly-trained model (session scoped)."""

from __future__ import annotations

import copy
import os
import warnings

import numpy as np
import pandas as pd
import pytest

os.environ["MLFLOW_DISABLED"] = "1"
os.environ.pop("DATABASE_URL", None)
warnings.filterwarnings("ignore")

from src.config import load_config  # noqa: E402
from src.data.repository import Dataset, load_from_csv  # noqa: E402
from src.data.synthetic import generate, save  # noqa: E402


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory):
    cfg = copy.deepcopy(load_config())
    root = tmp_path_factory.mktemp("plr")
    cfg["paths"] = {"synthetic_dir": str(root / "synthetic"), "processed_dir": str(root / "processed"),
                    "model_dir": str(root / "models"), "reports_dir": str(root / "reports")}
    cfg["synthetic"].update(n_teams=2, athletes_per_team=25, seed=7)
    models = cfg["training"]["models"]
    for name in models:
        models[name]["enabled"] = name in ("logistic_regression", "lightgbm")
    models["lightgbm"].update(n_estimators=300, early_stopping_rounds=50, min_child_samples=50)
    cfg["split"]["forward_cv"]["enabled"] = False
    return cfg


@pytest.fixture(scope="session")
def small_ds(small_cfg) -> Dataset:
    save(generate(small_cfg), small_cfg["paths"]["synthetic_dir"])
    return load_from_csv(small_cfg["paths"]["synthetic_dir"])


@pytest.fixture(scope="session")
def trained(small_cfg, small_ds):
    from training.train import run_training

    return run_training(small_cfg, ds=small_ds, evaluate_after=False, track=False)


@pytest.fixture(scope="session")
def service(small_cfg, small_ds, trained):
    from src.service import RiskService

    return RiskService(small_cfg, dataset=small_ds)


@pytest.fixture(scope="session")
def available_athlete_date(service):
    """An (athlete_id, date) pair where the athlete is available and in-season."""
    f = service.features
    ok = f[f["eligible"]]
    athlete, date = ok.index[len(ok) // 2]
    return athlete, str(date.date())


def make_dataset(daily_loads, start="2024-07-01", absences=None, athlete_id="A1", position="CM",
                 fixtures=None) -> Dataset:
    """Hand-crafted single-athlete dataset: one training session per day with ``player_load = load``.

    ``None``/0 in ``daily_loads`` means rest day (no session).
    """
    dates = pd.date_range(start, periods=len(daily_loads), freq="D")
    rows = []
    for i, (d, load) in enumerate(zip(dates, daily_loads)):
        if not load:
            continue
        rows.append(dict(session_id=f"S{i:05d}", athlete_id=athlete_id, team_id="T1", date=d, season="2024-25",
                         session_type="training", match_or_training="training", duration_minutes=60.0,
                         minutes_played=0.0, total_distance=load * 10.0, high_speed_running_distance=load * 0.5,
                         sprint_distance=load * 0.1, sprint_count=max(1, int(load / 100)), accelerations=10,
                         decelerations=10, max_speed=30.0, player_load=float(load), perceived_exertion=5.0,
                         training_load=300.0, position=position))
    sessions = pd.DataFrame(rows)
    athletes = pd.DataFrame([{"athlete_id": athlete_id, "team_id": "T1", "position": position,
                              "date_of_birth": pd.Timestamp("1998-01-01"), "prior_injury_count": 0}])
    ab = pd.DataFrame(absences or [], columns=["absence_id", "athlete_id", "onset_date", "return_date", "days_lost",
                                               "absence_type"])
    for c in ("onset_date", "return_date"):
        ab[c] = pd.to_datetime(ab[c])
    fx = fixtures if fixtures is not None else pd.DataFrame(columns=["team_id", "date"])
    fx["date"] = pd.to_datetime(fx["date"])
    return Dataset(athletes, sessions, ab, fx)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
