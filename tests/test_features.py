"""Feature pipeline tests on hand-crafted data."""

import pandas as pd
import pytest

from src.config import load_config
from src.features.build import build_features
from src.features.catalog import FEATURE_NAMES
from tests.conftest import make_dataset

CFG = load_config()


def test_rest_and_consecutive_days():
    loads = [300] * 40 + [0, 0] + [300] * 5
    f = build_features(make_dataset(loads), CFG).xs("A1", level="athlete_id")
    last = f.iloc[-1]
    assert last["consecutive_training_days"] == 5
    assert last["rest_days_7d"] == 2
    assert last["days_since_last_session"] == 0
    assert f.iloc[41]["days_since_last_session"] == 2


def test_rolling_loads_and_acwr_values():
    loads = [100] * 28 + [200] * 7
    f = build_features(make_dataset(loads), CFG).xs("A1", level="athlete_id")
    last = f.iloc[-1]
    assert last["player_load_7d"] == 1400
    assert last["player_load_28d"] == 21 * 100 + 7 * 200
    assert last["acwr_total_load"] == pytest.approx(1400 / ((2100 + 1400) / 4))
    assert last["acwr_total_load_uncoupled"] == pytest.approx(1400 / 700)
    assert last["weekly_load_change"] == pytest.approx(1.0)


def test_athlete_baseline_deviation():
    # 12 stable weeks at 100/day, then one week at 150/day → +50 % vs personal baseline
    loads = [100] * 98 + [150] * 7
    f = build_features(make_dataset(loads), CFG).xs("A1", level="athlete_id")
    last = f.iloc[-1]
    assert last["player_load_baseline_weekly"] == pytest.approx(700)
    assert last["player_load_vs_baseline_pct"] == pytest.approx(50.0)


def test_baseline_excludes_unavailable_days():
    loads = [100] * 70 + [0] * 14 + [100] * 21
    absence = [{"absence_id": "X1", "athlete_id": "A1", "onset_date": "2024-09-09", "return_date": "2024-09-23",
                "days_lost": 14, "absence_type": "injury"}]
    f = build_features(make_dataset(loads, absences=absence), CFG).xs("A1", level="athlete_id")
    last = f.iloc[-1]
    assert last["player_load_baseline_weekly"] == pytest.approx(700)  # absence days don't drag the baseline down
    assert last["previous_injury_count"] == 1
    assert last["previous_90d_injury_burden"] == 14
    assert last["days_since_return"] == 20  # returned 2024-09-23, last day 2024-10-13


def test_unavailable_days_not_eligible():
    loads = [100] * 70 + [0] * 5 + [100] * 10
    absence = [{"absence_id": "X1", "athlete_id": "A1", "onset_date": "2024-09-09", "return_date": "2024-09-14",
                "days_lost": 5, "absence_type": "illness"}]
    f = build_features(make_dataset(loads, absences=absence), CFG).xs("A1", level="athlete_id")
    assert not f.loc["2024-09-10", "available"]
    assert not f.loc["2024-09-10", "eligible"]
    assert f.loc["2024-09-14", "available"]


def test_burn_in_period_not_eligible():
    f = build_features(make_dataset([100] * 60), CFG).xs("A1", level="athlete_id")
    burn = CFG["features"]["burn_in_days"]
    assert not f.iloc[: burn]["eligible"].any()
    assert f.iloc[burn:]["eligible"].all()


def test_schema_matches_catalogue_and_has_no_latent_traits(small_ds, small_cfg):
    f = build_features(small_ds, small_cfg)
    assert all(c in f.columns for c in FEATURE_NAMES)
    latent = {"workload_tolerance", "injury_susceptibility", "recovery_capacity", "selection_propensity", "fitness",
              "vmax"}
    assert not latent & set(f.columns)
    assert f[FEATURE_NAMES].dtypes.eq("float64").all()


def test_fixture_calendar_feature():
    fx = pd.DataFrame({"team_id": ["T1", "T1"], "date": ["2024-08-03", "2024-08-07"]})
    f = build_features(make_dataset([100] * 40, fixtures=fx), CFG).xs("A1", level="athlete_id")
    assert f.loc["2024-08-01", "scheduled_matches_next_7d"] == 2
    assert f.loc["2024-08-03", "scheduled_matches_next_7d"] == 1  # same-day fixture not in (t, t+7]
