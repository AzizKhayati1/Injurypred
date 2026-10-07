import pandas as pd
import pytest

from src.simulation.whatif import Scenario, apply_scenario

SESSION = {"duration_minutes": 80.0, "minutes_played": 0.0, "total_distance": 6000.0,
           "high_speed_running_distance": 400.0, "sprint_distance": 100.0, "sprint_count": 6, "accelerations": 40,
           "decelerations": 45, "player_load": 650.0, "max_speed": 30.0, "perceived_exertion": 6.0}


def test_rest_day_removes_session():
    assert apply_scenario(SESSION, Scenario("rest", rest_day=True)) is None


def test_sprint_increase_keeps_band_consistency():
    s = apply_scenario(SESSION, Scenario("s", sprint_load_pct=20))
    assert s["sprint_distance"] == pytest.approx(120)
    assert s["high_speed_running_distance"] == pytest.approx(420)
    assert s["total_distance"] == pytest.approx(6020)
    assert s["sprint_distance"] <= s["high_speed_running_distance"] <= s["total_distance"]


def test_total_load_scales_volume():
    s = apply_scenario(SESSION, Scenario("t", total_load_pct=10))
    assert s["player_load"] == pytest.approx(715)
    assert s["duration_minutes"] == SESSION["duration_minutes"]


def test_reduced_session():
    s = apply_scenario(SESSION, Scenario("r", duration_pct=-30))
    assert s["duration_minutes"] == pytest.approx(56)
    assert s["total_distance"] == pytest.approx(4200)


def test_scenario_from_dict_names_itself():
    assert Scenario.from_dict({"sprint_load_pct": 20}).name == "+20% sprint"
    assert Scenario.from_dict({"rest_day": True}).name == "Additional recovery day"


def test_simulation_end_to_end(service, available_athlete_date):
    aid, date = available_athlete_date
    res = service.simulate(aid, date, [{"name": "rest", "rest_day": True}, {"name": "big", "total_load_pct": 100,
                                                                             "sprint_load_pct": 100}])
    assert res["simulated_day"] == str((pd.Timestamp(date) + pd.Timedelta(days=1)).date())
    rest, big = res["scenarios"]
    base_load = res["baseline"]["key_features"]["player_load_7d"]
    assert rest["feature_changes"]["player_load_7d"]["scenario"] < base_load
    assert big["feature_changes"]["player_load_7d"]["scenario"] > base_load
    for s in res["scenarios"]:
        assert 0 <= s["risk_probability"] <= 1
        assert s["delta_probability"] == pytest.approx(s["risk_probability"] - res["baseline"]["risk_probability"],
                                                       abs=1e-3)
    assert "not causal" in res["disclaimer"]


def test_simulation_ignores_future_data(service, small_ds, small_cfg, available_athlete_date):
    """Changing data after the as-of date must not change the simulated risk."""
    from src.data.repository import Dataset
    from src.simulation.whatif import run_simulation

    aid, date = available_athlete_date
    sc = [Scenario("x", sprint_load_pct=20)]
    r1 = run_simulation(small_ds, small_cfg, service.model, service.engine, aid, date, sc)
    s = small_ds.sessions.copy()
    s.loc[s["date"] > pd.Timestamp(date), "player_load"] *= 5
    ds2 = Dataset(small_ds.athletes, s, small_ds.absences, small_ds.fixtures)
    r2 = run_simulation(ds2, small_cfg, service.model, service.engine, aid, date, sc)
    assert r1.baseline["risk_probability"] == r2.baseline["risk_probability"]
    assert r1.scenarios[0]["risk_probability"] == r2.scenarios[0]["risk_probability"]
