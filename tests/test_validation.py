import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.schemas import SessionRecord
from src.data.validation import DataValidationError, validate_dataset
from tests.conftest import make_dataset


@pytest.fixture
def ds():
    return make_dataset([300] * 30)


def _expect(ds, check, mutate):
    s = ds.sessions.copy()
    mutate(s)
    with pytest.raises(DataValidationError) as exc:
        validate_dataset(ds.athletes, s, ds.absences)
    assert check in {i.check for i in exc.value.report.errors}


def test_clean_data_passes(ds):
    assert validate_dataset(ds.athletes, ds.sessions, ds.absences).ok


def test_negative_max_speed_fails(ds):
    _expect(ds, "impossible_speed", lambda s: s.__setitem__("max_speed", s["max_speed"].where(s.index != 3, -1.0)))


def test_impossible_speed_fails(ds):
    _expect(ds, "impossible_speed", lambda s: s.__setitem__("max_speed", s["max_speed"].where(s.index != 3, 61.0)))


def test_negative_distance_fails(ds):
    _expect(ds, "negative_values", lambda s: s.__setitem__("total_distance", -s["total_distance"]))


def test_invalid_duration_fails(ds):
    _expect(ds, "invalid_duration", lambda s: s.__setitem__("duration_minutes", 0.0))


def test_missing_values_fail(ds):
    _expect(ds, "missing_values", lambda s: s.__setitem__("player_load", s["player_load"].where(s.index != 1)))


def test_duplicate_sessions_fail(ds):
    s = pd.concat([ds.sessions, ds.sessions.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError) as exc:
        validate_dataset(ds.athletes, s, ds.absences)
    checks = {i.check for i in exc.value.report.errors}
    assert {"duplicate_session_id", "duplicate_session"} <= checks


def test_invalid_timestamp_fails(ds):
    s = ds.sessions.copy().astype({"date": object})
    s.loc[2, "date"] = "not-a-date"
    with pytest.raises(DataValidationError) as exc:
        validate_dataset(ds.athletes, s, ds.absences)
    assert "invalid_timestamp" in {i.check for i in exc.value.report.errors}


def test_unknown_athlete_fails(ds):
    _expect(ds, "unknown_athlete_id", lambda s: s.__setitem__("athlete_id", "GHOST"))


def test_sprint_band_consistency(ds):
    _expect(ds, "sprint_exceeds_hsr", lambda s: s.__setitem__("sprint_distance", s["high_speed_running_distance"] + 5))


def test_chronological_order_warning(ds):
    s = ds.sessions.copy()
    s["session_id"] = s["session_id"].iloc[::-1].values
    rep = validate_dataset(ds.athletes, s, ds.absences)
    assert "chronological_order" in {i.check for i in rep.warnings}


def test_overlapping_absences_fail(ds):
    ab = pd.DataFrame([
        {"absence_id": "X1", "athlete_id": "A1", "onset_date": "2024-07-05", "return_date": "2024-07-15",
         "days_lost": 10, "absence_type": "injury"},
        {"absence_id": "X2", "athlete_id": "A1", "onset_date": "2024-07-10", "return_date": "2024-07-12",
         "days_lost": 2, "absence_type": "illness"}])
    with pytest.raises(DataValidationError) as exc:
        validate_dataset(ds.athletes, ds.sessions, ab)
    assert "overlapping_absences" in {i.check for i in exc.value.report.errors}


def test_pydantic_record_rejects_negative_speed():
    base = dict(session_id="S1", athlete_id="A1", date="2025-01-01", session_type="training",
                match_or_training="training", duration_minutes=60, minutes_played=0, total_distance=5000,
                high_speed_running_distance=300, sprint_distance=50, sprint_count=3, accelerations=10,
                decelerations=10, max_speed=30, player_load=500, perceived_exertion=5)
    SessionRecord(**base)
    with pytest.raises(ValidationError):
        SessionRecord(**{**base, "max_speed": -1})
    with pytest.raises(ValidationError):
        SessionRecord(**{**base, "sprint_distance": 400})


def test_generated_dataset_is_valid(small_ds):
    assert validate_dataset(small_ds.athletes, small_ds.sessions, small_ds.absences).ok
