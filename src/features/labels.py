"""Target construction.

For a prediction made at the **end of day t** (all data dated ≤ t known):

* ``availability_risk_7d``   – 1 if a new absence (injury *or* illness) starts in (t, t+7]
* ``injury_event_next_7d``   – 1 if a new injury starts in (t, t+7]
* ``injury_event_next_14d``  – 1 if a new injury starts in (t, t+14]
* ``availability_days_lost`` – number of days in (t, t+7] the athlete is unavailable
* ``event_next_1d``          – 1 if a new absence starts on t+1 (discrete-time hazard target)

A label for horizon h is only *observable* when ``t + h`` is not after the
end of the season containing ``t`` nor after the last day of data. Rows whose
label window is not fully observable are excluded from training/evaluation
(but are still scored in production).
"""

from __future__ import annotations

import pandas as pd

from src.data.repository import Dataset
from src.features import workload as W
from src.features.build import DailyPanel

TARGETS = ["availability_risk_7d", "injury_event_next_7d", "injury_event_next_14d", "availability_days_lost",
           "event_next_1d"]


def build_labels(ds: Dataset, panel: DailyPanel | None = None, observed_end=None) -> pd.DataFrame:
    p = panel or DailyPanel(ds)
    observed_end = pd.Timestamp(observed_end) if observed_end is not None else p.end
    frames = {
        "availability_risk_7d": W.forward_window_sum(p.onset_any.astype(float), 7) > 0,
        "injury_event_next_7d": W.forward_window_sum(p.onset_injury.astype(float), 7) > 0,
        "injury_event_next_14d": W.forward_window_sum(p.onset_injury.astype(float), 14) > 0,
        "availability_days_lost": W.forward_window_sum(p.unavailable.astype(float), 7),
        "event_next_1d": p.onset_any.shift(-1, fill_value=False),
    }
    cols = {}
    for name, fr in frames.items():
        fr.index.name, fr.columns.name = "date", "athlete_id"
        cols[name] = fr.stack(future_stack=True)
    out = pd.DataFrame(cols).swaplevel().sort_index()
    for c in ["availability_risk_7d", "injury_event_next_7d", "injury_event_next_14d", "event_next_1d"]:
        out[c] = out[c].astype(int)
    out["observed_end"] = observed_end
    return out


def attach_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join labels to features and compute per-horizon observability flags."""
    df = features.join(labels, how="left")
    dates = df.index.get_level_values("date")
    window_end = pd.concat([df["season_end"], df["observed_end"]], axis=1).min(axis=1)
    for h in (1, 7, 14):
        df[f"observable_{h}d"] = (dates + pd.Timedelta(days=h)) <= window_end
    return df
