"""Leakage prevention: features at t must not change when the future changes."""

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_NAMES, build_modeling_table
from src.features.build import build_features
from src.features.labels import TARGETS
from src.models.splitting import temporal_split


@pytest.mark.parametrize("cut", ["2025-02-14", "2025-10-01", "2026-03-20"])
def test_features_invariant_to_truncating_the_future(small_ds, small_cfg, cut):
    """Recomputing features from data known at the end of `cut` gives identical values at `cut`."""
    full = build_features(small_ds, small_cfg)
    trunc = build_features(small_ds.truncate(cut), small_cfg, end=cut)
    t = pd.Timestamp(cut)
    a = full.xs(t, level="date")[FEATURE_NAMES]
    b = trunc.xs(t, level="date")[FEATURE_NAMES]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)


def test_features_invariant_to_perturbing_future_sessions(small_ds, small_cfg):
    cut = pd.Timestamp("2025-11-15")
    future = small_ds.sessions["date"] > cut
    perturbed = small_ds.sessions.copy()
    perturbed.loc[future, ["player_load", "sprint_distance", "total_distance"]] *= 10
    from src.data.repository import Dataset

    ds2 = Dataset(small_ds.athletes, perturbed, small_ds.absences, small_ds.fixtures)
    a = build_features(small_ds, small_cfg).loc[pd.IndexSlice[:, :cut], FEATURE_NAMES]
    b = build_features(ds2, small_cfg).loc[pd.IndexSlice[:, :cut], FEATURE_NAMES]
    pd.testing.assert_frame_equal(a, b)


def test_labels_are_not_features():
    assert not set(TARGETS) & set(FEATURE_NAMES)
    assert not any(n.startswith(("observable", "label")) for n in FEATURE_NAMES)


def test_labels_depend_on_future_only(small_ds, small_cfg):
    table = build_modeling_table(small_ds, small_cfg)
    ab = small_ds.absences
    row = ab.iloc[len(ab) // 2]
    aid, onset = row["athlete_id"], row["onset_date"]
    before = table.loc[(aid, onset - pd.Timedelta(days=3))]
    assert before["availability_risk_7d"] == 1
    on_day = table.loc[(aid, onset)]
    assert not on_day["eligible"]  # the onset day itself is never a prediction row


def test_split_purging_prevents_label_overlap(small_ds, small_cfg):
    table = build_modeling_table(small_ds, small_cfg)
    sp = temporal_split(table, small_cfg["split"], horizon=7)
    for name in ("train", "validation", "test"):
        dates = sp[name].index.get_level_values("date")
        end = pd.Timestamp(small_cfg["split"][name]["end"])
        assert (dates + pd.Timedelta(days=7) <= end).all()
    tr_end = sp["train"].index.get_level_values("date").max() + pd.Timedelta(days=7)
    assert tr_end < sp["validation"].index.get_level_values("date").min()
    assert sp["test"].index.get_level_values("date").min() > sp["validation"].index.get_level_values("date").max()


def test_rows_only_where_label_window_observable(small_ds, small_cfg):
    table = build_modeling_table(small_ds, small_cfg)
    end = small_ds.data_end
    late = table[table.index.get_level_values("date") > end - pd.Timedelta(days=7)]
    assert not late["observable_7d"].any()
    assert np.all(table.loc[table["observable_7d"], "availability_risk_7d"].notna())
