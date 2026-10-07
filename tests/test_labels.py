
from src.config import load_config
from src.features.build import DailyPanel, build_features
from src.features.labels import attach_labels, build_labels
from tests.conftest import make_dataset

CFG = load_config()


def _table(loads, absences):
    ds = make_dataset(loads, absences=absences)
    p = DailyPanel(ds)
    return attach_labels(build_features(ds, CFG, panel=p), build_labels(ds, p)).xs("A1", level="athlete_id")


def test_label_windows():
    absences = [{"absence_id": "X1", "athlete_id": "A1", "onset_date": "2024-08-20", "return_date": "2024-08-25",
                 "days_lost": 5, "absence_type": "injury"},
                {"absence_id": "X2", "athlete_id": "A1", "onset_date": "2024-09-10", "return_date": "2024-09-12",
                 "days_lost": 2, "absence_type": "illness"}]
    t = _table([100] * 120, absences)
    assert t.loc["2024-08-13", "injury_event_next_7d"] == 1      # onset at t+7
    assert t.loc["2024-08-12", "injury_event_next_7d"] == 0      # onset at t+8
    assert t.loc["2024-08-12", "injury_event_next_14d"] == 1
    assert t.loc["2024-08-19", "availability_days_lost"] == 5
    assert t.loc["2024-09-05", "availability_risk_7d"] == 1      # illness counts for availability …
    assert t.loc["2024-09-05", "injury_event_next_7d"] == 0      # … but not for the injury target
    assert t.loc["2024-08-19", "event_next_1d"] == 1


def test_observability_respects_season_end():
    t = _table([100] * 400, [])
    assert t.loc["2025-05-24", "observable_7d"]
    assert not t.loc["2025-05-25", "observable_7d"]  # window would cross the season end (31 May)
