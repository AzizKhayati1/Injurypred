import pandas as pd
import pytest

from src.features import build_modeling_table
from src.models.splitting import athlete_group, athlete_holdout, forward_folds, temporal_split


@pytest.fixture(scope="module")
def table(small_ds, small_cfg):
    return build_modeling_table(small_ds, small_cfg)


def test_splits_are_chronological_and_disjoint(table, small_cfg):
    sp = temporal_split(table, small_cfg["split"])
    idx = [set(sp[n].index) for n in ("train", "validation", "test", "production")]
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            assert not idx[i] & idx[j]
    assert sp["train"].index.get_level_values("date").max() < sp["validation"].index.get_level_values("date").min()


def test_forward_folds_train_strictly_before_test(table):
    for _month, tr, te in forward_folds(table, ["2025-10", "2026-02"], horizon=7):
        last_label = tr.index.get_level_values("date").max() + pd.Timedelta(days=7)
        assert last_label < te.index.get_level_values("date").min()


def test_athlete_holdout_has_no_overlap(table, small_cfg):
    sp = temporal_split(table, small_cfg["split"])
    tr, va, te = athlete_holdout(sp, 0.3)
    seen = set(tr.index.get_level_values("athlete_id"))
    unseen = set(te.index.get_level_values("athlete_id"))
    assert seen and unseen and not seen & unseen


def test_athlete_group_is_deterministic():
    assert [athlete_group(f"A{i}", 0.2) for i in range(50)] == [athlete_group(f"A{i}", 0.2) for i in range(50)]
