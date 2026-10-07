"""Temporal validation splits.

Rows are athlete-days indexed by prediction date ``t`` whose label covers
``(t, t+h]``. Random K-fold would let the model train on day ``t+1`` and be
tested on day ``t`` of the same athlete — the rolling features of neighbouring
days overlap almost completely, so the scores would be wildly optimistic.

Rules implemented here
----------------------
* **Chronological split** – train (season 1) → validation (early season 2) →
  test (late season 2). The production period (current season) is never used
  for fitting.
* **Purging** – a row belongs to a labelled split only if its *label window*
  ``t + h`` also ends inside that split. Labels of training rows therefore
  never peek into the validation/test periods.
* **Forward (rolling-origin) CV** – for each monthly fold the model is trained
  on rows whose label window ends *before* the fold starts.
* **Athlete hold-out** – optional additional check: athletes in the hold-out
  group are never seen during training (tests generalisation to new squads).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def _dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(df.index.get_level_values("date"))


def labelled_rows(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    return df[df["eligible"] & df[f"observable_{horizon}d"]]


def temporal_split(df: pd.DataFrame, split_cfg: dict, horizon: int = 7) -> dict[str, pd.DataFrame]:
    """Split a modelling table into train / validation / test / production frames."""
    purge = pd.Timedelta(days=max(split_cfg.get("purge_days", horizon), horizon))
    out = {}
    dates = _dates(df)
    for name in ("train", "validation", "test"):
        s, e = pd.Timestamp(split_cfg[name]["start"]), pd.Timestamp(split_cfg[name]["end"])
        mask = (dates >= s) & (dates + purge <= e) & df["eligible"].values & df[f"observable_{horizon}d"].values
        out[name] = df[mask]
    s, e = pd.Timestamp(split_cfg["production"]["start"]), pd.Timestamp(split_cfg["production"]["end"])
    out["production"] = df[(dates >= s) & (dates <= e) & df["eligible"].values]
    _assert_no_temporal_overlap(out, horizon)
    return out


def _assert_no_temporal_overlap(splits: dict[str, pd.DataFrame], horizon: int) -> None:
    """Every label window of an earlier split must end before the next split starts."""
    order = [n for n in ("train", "validation", "test") if len(splits.get(n, [])) > 0]
    for earlier, later in zip(order, order[1:]):
        last_label_day = _dates(splits[earlier]).max() + pd.Timedelta(days=horizon)
        first_later = _dates(splits[later]).min()
        if last_label_day >= first_later:
            raise AssertionError(f"temporal leakage: {earlier} labels reach {last_label_day.date()} "
                                 f"but {later} starts {first_later.date()}")


def forward_folds(df: pd.DataFrame, months: list[str], horizon: int = 7):
    """Yield (month, train_frame, test_frame) for rolling-origin evaluation."""
    data = labelled_rows(df, horizon)
    dates = _dates(data)
    for month in months:
        start = pd.Timestamp(f"{month}-01")
        end = start + pd.offsets.MonthEnd(0)
        train = data[dates + pd.Timedelta(days=horizon) < start]
        test = data[(dates >= start) & (dates <= end)]
        if len(test) and len(train):
            yield month, train, test


def athlete_group(athlete_id: str, fraction: float, salt: str = "holdout") -> bool:
    """Deterministic hash-based assignment of an athlete to the hold-out group."""
    h = int(hashlib.md5(f"{salt}:{athlete_id}".encode()).hexdigest(), 16)
    return (h % 10_000) / 10_000 < fraction


def athlete_holdout(splits: dict[str, pd.DataFrame], fraction: float) -> tuple[pd.DataFrame, pd.DataFrame,
                                                                                 pd.DataFrame]:
    """(train on seen athletes, validation on seen athletes, test on unseen athletes)."""
    def is_held(frame):
        ids = frame.index.get_level_values("athlete_id")
        return pd.Series([athlete_group(a, fraction) for a in ids], index=frame.index).values
    tr, va, te = splits["train"], splits["validation"], splits["test"]
    return tr[~is_held(tr)], va[~is_held(va)], te[is_held(te)]
