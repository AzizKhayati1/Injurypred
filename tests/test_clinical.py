"""Player profiles, clinical injury records and prevention recommendations."""

import pandas as pd
import pytest

from src.clinical.injuries import CATALOGUE, enrich_absences, player_injury_summary, severity
from src.clinical.prevention import REFERENCES, recommend
from src.data.profiles import generate_profiles
from src.explainability.narrative import FORBIDDEN_PHRASES


def test_profiles_are_deterministic_and_consistent(small_ds):
    p1, c1 = generate_profiles(small_ds.athletes)
    p2, _ = generate_profiles(small_ds.athletes)
    pd.testing.assert_frame_equal(p1, p2)
    assert p1["name"].is_unique
    merged = p1.merge(small_ds.athletes[["athlete_id", "team_id"]], on="athlete_id")
    assert not merged.duplicated(["team_id", "shirt_number"]).any()
    assert set(c1["team_id"]) == set(small_ds.athletes["team_id"])
    assert (p1["data_origin"] == "synthetic").all()


@pytest.mark.parametrize("days,label", [(1, "Slight"), (3, "Slight"), (4, "Minimal"), (8, "Mild"), (28, "Mild"),
                                        (29, "Moderate"), (91, "Severe")])
def test_severity_bands(days, label):
    assert severity(days) == label


def test_enrichment_matches_recorded_category(small_ds):
    d = small_ds.injury_details.merge(small_ds.absences[["absence_id", "category"]], on="absence_id")
    assert len(d) == len(small_ds.absences)
    for r in d.itertuples():
        assert CATALOGUE[r.diagnosis_code].category == r.category
        assert (r.side in ("Left", "Right")) == CATALOGUE[r.diagnosis_code].sided


def test_recurrence_definition():
    ab = pd.DataFrame([
        {"absence_id": "X1", "athlete_id": "A1", "onset_date": "2025-01-01", "return_date": "2025-01-15", "days_lost": 14,
         "absence_type": "injury", "category": "muscle"},
        {"absence_id": "X2", "athlete_id": "A1", "onset_date": "2025-02-10", "return_date": "2025-02-20", "days_lost": 10,
         "absence_type": "injury", "category": "muscle"}])
    for c in ("onset_date", "return_date"):
        ab[c] = pd.to_datetime(ab[c])
    players = pd.DataFrame([{"athlete_id": "A1", "preferred_foot": "Right"}])
    # force same diagnosis & side by retrying seeds until the generator assigns them (deterministic search)
    for seed in range(200):
        d = enrich_absences(ab, players, seed=seed)
        if d["diagnosis_code"].nunique() == 1 and d["side"].nunique() == 1:
            break
    assert d.loc[1, "recurrence"] == "Early recurrence"   # 26 days after return
    assert pd.isna(d.loc[0, "recurrence"])


def test_injury_summary_hides_future(small_ds):
    ab = small_ds.absences
    cut = ab["onset_date"].sort_values().iloc[len(ab) // 2]
    aid = ab.loc[ab["onset_date"] == cut, "athlete_id"].iloc[0]
    s = player_injury_summary(small_ds.injury_details, ab, aid, cut - pd.Timedelta(days=1))
    assert all(pd.Timestamp(r["onset_date"]) < cut for r in s["records"])


ROW = {"sprint_distance_vs_baseline_pct": 80.0, "acwr_sprint_load": 1.8, "weekly_load_change": 0.5,
       "ewma_acwr_total_load": 1.6, "rest_days_7d": 0, "consecutive_training_days": 9, "rpe_trend": 1.5,
       "days_since_return": 10, "days_since_near_max_speed": 30}


def test_prevention_rules_trigger_from_workload():
    recs = recommend(ROW, {"records": []}, "W", "VERY_HIGH")
    ids = {r["id"] for r in recs}
    assert {"hsr_spike", "load_spike", "recovery", "wellness", "post_return", "max_speed", "nordic", "warmup"} <= ids
    assert recs[0]["priority"] == "High"
    calm = {"sprint_distance_vs_baseline_pct": 0, "acwr_sprint_load": 1.0, "weekly_load_change": 0.0,
            "ewma_acwr_total_load": 1.0, "rest_days_7d": 2, "consecutive_training_days": 2, "rpe_trend": 0,
            "days_since_return": 365, "days_since_near_max_speed": 2}
    assert {r["id"] for r in recommend(calm, {"records": []}, "CB", "LOW")} == {"warmup"}


def test_prevention_history_rules_and_rtp():
    hist = {"records": [{"absence_type": "injury", "diagnosis_code": "ANK", "onset_date": "2025-03-01", "status": "Resolved"}]}
    ids = {r["id"] for r in recommend(None, hist, "CB", None, as_of="2025-06-01")}
    assert "balance" in ids
    current = {"records": [], "current": {"diagnosis": "Concussion", "diagnosis_code": "CON", "onset_date": "2025-06-01"}}
    out = recommend(None, current, "CB", None)
    assert [r["id"] for r in out] == ["rtp"]
    assert any(ref["key"] == "patricios_2023" for ref in out[0]["references"])


def test_prevention_language_and_references():
    for r in recommend(ROW, {"records": []}, "W", "HIGH"):
        text = " ".join([r["title"], r["why"], r["evidence"], *r["actions"]]).lower()
        assert not any(p in text for p in FORBIDDEN_PHRASES)
        assert r["references"] and all(ref["citation"] == REFERENCES[ref["key"]] for ref in r["references"])


def test_service_player_views(service, available_athlete_date):
    aid, _ = available_athlete_date
    card = service.player_card(aid)
    assert card["name"] and card["club"]["name"] and card["status"] in ("Available", "Day-to-day", "Out")
    prof = service.player_profile(aid)
    assert prof["stats"]["career"]["sessions"] > 0
    rep = service.injury_report()
    assert rep["n_current"] == len(rep["current"])
    assert all(x["days_to_return"] >= 0 for x in rep["current"])
