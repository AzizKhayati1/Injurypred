"""Feature engineering: workload maths, athlete baselines, temporal features and labels."""

from __future__ import annotations

import pandas as pd

from src.data.repository import Dataset
from src.features.build import DailyPanel, build_features
from src.features.catalog import FEATURE_NAMES
from src.features.labels import TARGETS, attach_labels, build_labels

__all__ = ["FEATURE_NAMES", "TARGETS", "build_modeling_table"]


def build_modeling_table(ds: Dataset, cfg: dict) -> pd.DataFrame:
    """Features + labels for every athlete-day (eligibility/observability flags included)."""
    panel = DailyPanel(ds)
    feats = build_features(ds, cfg, panel=panel)
    labels = build_labels(ds, panel=panel)
    return attach_labels(feats, labels)
