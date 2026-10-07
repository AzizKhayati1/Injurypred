"""Table-level data validation.

Two layers:

1. **Record-level** – every session row is parsed with the Pydantic
   :class:`~src.data.schemas.SessionRecord` (types, ranges, physical
   plausibility, internal consistency).
2. **Table-level** – vectorised checks that need the whole table:
   missing values, duplicates, timestamp validity, chronological ordering,
   athlete-ID consistency across tables, overlapping absences, sessions logged
   while an athlete was recorded as absent.

``ERROR`` issues block training/serving; ``WARNING`` issues are reported.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date

import pandas as pd
from pydantic import TypeAdapter, ValidationError

from src.data.schemas import MAX_PLAUSIBLE_SPEED_KMH, MAX_SESSION_MINUTES, SessionRecord

logger = logging.getLogger(__name__)

REQUIRED_SESSION_COLUMNS = [
    "session_id", "athlete_id", "date", "session_type", "match_or_training", "duration_minutes",
    "minutes_played", "total_distance", "high_speed_running_distance", "sprint_distance", "sprint_count",
    "accelerations", "decelerations", "max_speed", "player_load", "perceived_exertion",
]
REQUIRED_ABSENCE_COLUMNS = ["absence_id", "athlete_id", "onset_date", "return_date", "days_lost", "absence_type"]
REQUIRED_ATHLETE_COLUMNS = ["athlete_id", "team_id", "position", "date_of_birth", "prior_injury_count"]
NON_NEGATIVE_COLUMNS = ["total_distance", "high_speed_running_distance", "sprint_distance", "sprint_count",
                        "accelerations", "decelerations", "player_load", "minutes_played"]


class DataValidationError(ValueError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(report.summary())


@dataclass
class Issue:
    check: str
    severity: str  # "ERROR" | "WARNING"
    table: str
    n_rows: int
    message: str
    examples: list = field(default_factory=list)


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    n_rows: dict[str, int] = field(default_factory=dict)

    def add(self, check: str, severity: str, table: str, mask_or_n, message: str, examples=None) -> None:
        n = int(mask_or_n.sum()) if hasattr(mask_or_n, "sum") else int(mask_or_n)
        if n > 0:
            self.issues.append(Issue(check, severity, table, n, message, list(examples or [])[:5]))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [f"Validation: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        for i in self.issues:
            lines.append(f"  [{i.severity}] {i.table}.{i.check}: {i.n_rows} row(s) – {i.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "n_rows": self.n_rows, "issues": [asdict(i) for i in self.issues]}


def _examples(df: pd.DataFrame, mask, cols) -> list:
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    return df.loc[mask, cols].head(5).astype(str).to_dict("records")


def validate_sessions(sessions: pd.DataFrame, report: ValidationReport | None = None,
                      today: date | None = None, record_level: bool = True) -> ValidationReport:
    report = report or ValidationReport()
    t = "sessions"
    report.n_rows[t] = len(sessions)
    missing_cols = [c for c in REQUIRED_SESSION_COLUMNS if c not in sessions.columns]
    if missing_cols:
        report.add("required_columns", "ERROR", t, len(missing_cols), f"missing columns: {missing_cols}")
        return report
    df = sessions

    # --- missing values ---------------------------------------------------------
    for c in REQUIRED_SESSION_COLUMNS:
        m = df[c].isna()
        report.add("missing_values", "ERROR", t, m, f"missing values in '{c}'", _examples(df, m, ["session_id", c]))

    # --- timestamps -------------------------------------------------------------
    dates = pd.to_datetime(df["date"], errors="coerce")
    m = dates.isna() & df["date"].notna()
    report.add("invalid_timestamp", "ERROR", t, m, "unparseable date", _examples(df, m, ["session_id", "date"]))
    if today is not None:
        m = dates > pd.Timestamp(today)
        report.add("future_timestamp", "ERROR", t, m, f"session dated after {today}",
                   _examples(df, m, ["session_id", "date"]))

    # --- ranges / physical plausibility ------------------------------------------
    for c in NON_NEGATIVE_COLUMNS:
        m = pd.to_numeric(df[c], errors="coerce") < 0
        report.add("negative_values", "ERROR", t, m, f"negative '{c}'", _examples(df, m, ["session_id", c]))
    ms = pd.to_numeric(df["max_speed"], errors="coerce")
    m = (ms < 0) | (ms > MAX_PLAUSIBLE_SPEED_KMH)
    report.add("impossible_speed", "ERROR", t, m, f"max_speed outside [0, {MAX_PLAUSIBLE_SPEED_KMH}] km/h",
               _examples(df, m, ["session_id", "max_speed"]))
    dur = pd.to_numeric(df["duration_minutes"], errors="coerce")
    m = (dur <= 0) | (dur > MAX_SESSION_MINUTES)
    report.add("invalid_duration", "ERROR", t, m, f"duration_minutes outside (0, {MAX_SESSION_MINUTES}]",
               _examples(df, m, ["session_id", "duration_minutes"]))
    m = df["high_speed_running_distance"] > df["total_distance"] + 1e-6
    report.add("hsr_exceeds_total", "ERROR", t, m, "HSR distance > total distance",
               _examples(df, m, ["session_id", "high_speed_running_distance", "total_distance"]))
    m = df["sprint_distance"] > df["high_speed_running_distance"] + 1e-6
    report.add("sprint_exceeds_hsr", "ERROR", t, m, "sprint distance > HSR distance",
               _examples(df, m, ["session_id", "sprint_distance", "high_speed_running_distance"]))
    m = (df["perceived_exertion"] < 0) | (df["perceived_exertion"] > 10)
    report.add("invalid_rpe", "ERROR", t, m, "RPE outside CR-10 range",
               _examples(df, m, ["session_id", "perceived_exertion"]))
    speed_per_min = df["total_distance"] / dur.where(dur > 0)
    m = speed_per_min > 250  # > 15 km/h average over a whole session is implausible
    report.add("implausible_average_speed", "WARNING", t, m, "average speed > 15 km/h across session",
               _examples(df, m, ["session_id", "total_distance", "duration_minutes"]))

    # --- duplicates -------------------------------------------------------------
    m = df["session_id"].duplicated(keep=False)
    report.add("duplicate_session_id", "ERROR", t, m, "duplicate session_id", _examples(df, m, ["session_id"]))
    key = ["athlete_id", "date", "session_type", "duration_minutes", "total_distance"]
    m = df.duplicated(subset=key, keep="first")
    report.add("duplicate_session", "ERROR", t, m, "identical session recorded twice",
               _examples(df, m, key))

    # --- chronological ordering --------------------------------------------------
    ordered = df.assign(_d=dates).sort_values("session_id")
    backwards = ordered.groupby("athlete_id")["_d"].diff().dt.days < 0
    report.add("chronological_order", "WARNING", t, backwards,
               "session_id order is not chronological within athlete (data will be re-sorted by date)")
    m = df["athlete_id"].astype(str).str.strip() != df["athlete_id"].astype(str)
    report.add("athlete_id_format", "ERROR", t, m, "athlete_id has leading/trailing whitespace")

    # --- record level (pydantic) --------------------------------------------------
    if record_level and report.ok:
        adapter = TypeAdapter(SessionRecord)
        bad, examples = 0, []
        records = df[[c for c in df.columns if c in SessionRecord.model_fields]].to_dict("records")
        for rec in records:
            try:
                adapter.validate_python(rec)
            except ValidationError as exc:
                bad += 1
                if len(examples) < 5:
                    examples.append({"session_id": rec.get("session_id"), "error": exc.errors()[0]["msg"]})
        report.add("record_schema", "ERROR", t, bad, "row failed SessionRecord schema", examples)
    return report


def validate_absences(absences: pd.DataFrame, report: ValidationReport | None = None) -> ValidationReport:
    report = report or ValidationReport()
    t = "absences"
    report.n_rows[t] = len(absences)
    missing_cols = [c for c in REQUIRED_ABSENCE_COLUMNS if c not in absences.columns]
    if missing_cols:
        report.add("required_columns", "ERROR", t, len(missing_cols), f"missing columns: {missing_cols}")
        return report
    df = absences.copy()
    onset = pd.to_datetime(df["onset_date"], errors="coerce")
    ret = pd.to_datetime(df["return_date"], errors="coerce")
    report.add("invalid_timestamp", "ERROR", t, onset.isna() | ret.isna(), "unparseable onset/return date")
    m = ret <= onset
    report.add("return_before_onset", "ERROR", t, m, "return_date <= onset_date",
               _examples(df, m, ["absence_id", "onset_date", "return_date"]))
    m = ~df["absence_type"].isin(["injury", "illness"])
    report.add("invalid_absence_type", "ERROR", t, m, "absence_type must be injury|illness")
    m = df["absence_id"].duplicated(keep=False)
    report.add("duplicate_absence_id", "ERROR", t, m, "duplicate absence_id")
    df["_on"], df["_ret"] = onset, ret
    df = df.sort_values(["athlete_id", "_on"])
    prev_ret = df.groupby("athlete_id")["_ret"].shift()
    m = df["_on"] < prev_ret
    report.add("overlapping_absences", "ERROR", t, m, "absence starts before the previous one ended",
               _examples(df, m, ["absence_id", "athlete_id", "onset_date"]))
    return report


def validate_athletes(athletes: pd.DataFrame, report: ValidationReport | None = None) -> ValidationReport:
    report = report or ValidationReport()
    t = "athletes"
    report.n_rows[t] = len(athletes)
    missing_cols = [c for c in REQUIRED_ATHLETE_COLUMNS if c not in athletes.columns]
    if missing_cols:
        report.add("required_columns", "ERROR", t, len(missing_cols), f"missing columns: {missing_cols}")
        return report
    m = athletes["athlete_id"].duplicated(keep=False)
    report.add("duplicate_athlete_id", "ERROR", t, m, "duplicate athlete_id", _examples(athletes, m, ["athlete_id"]))
    m = ~athletes["position"].isin(["GK", "CB", "FB", "CM", "W", "ST"])
    report.add("invalid_position", "ERROR", t, m, "unknown position", _examples(athletes, m, ["athlete_id", "position"]))
    m = pd.to_datetime(athletes["date_of_birth"], errors="coerce").isna()
    report.add("invalid_timestamp", "ERROR", t, m, "unparseable date_of_birth")
    return report


def validate_consistency(athletes: pd.DataFrame, sessions: pd.DataFrame, absences: pd.DataFrame,
                         report: ValidationReport) -> ValidationReport:
    known = set(athletes["athlete_id"])
    m = ~sessions["athlete_id"].isin(known)
    report.add("unknown_athlete_id", "ERROR", "sessions", m, "athlete_id not present in athletes table",
               _examples(sessions, m, ["session_id", "athlete_id"]))
    m = ~absences["athlete_id"].isin(known)
    report.add("unknown_athlete_id", "ERROR", "absences", m, "athlete_id not present in athletes table")
    if "position" in sessions.columns:
        pos = sessions["athlete_id"].map(athletes.set_index("athlete_id")["position"])
        m = sessions["position"].notna() & pos.notna() & (sessions["position"] != pos)
        report.add("position_mismatch", "WARNING", "sessions", m, "position differs from athletes table")
    # sessions logged while athlete recorded as absent (data-entry problem)
    if len(absences) and len(sessions):
        s = sessions[["athlete_id", "date"]].assign(date=pd.to_datetime(sessions["date"]))
        a = absences.assign(onset_date=pd.to_datetime(absences["onset_date"]),
                            return_date=pd.to_datetime(absences["return_date"]))
        merged = s.merge(a[["athlete_id", "onset_date", "return_date"]], on="athlete_id")
        m = (merged["date"] > merged["onset_date"]) & (merged["date"] < merged["return_date"])
        report.add("session_during_absence", "WARNING", "sessions", m,
                   "session recorded while athlete was absent")
    return report


def validate_dataset(athletes: pd.DataFrame, sessions: pd.DataFrame, absences: pd.DataFrame,
                     today: date | None = None, raise_on_error: bool = True,
                     record_level: bool = True) -> ValidationReport:
    report = ValidationReport()
    validate_athletes(athletes, report)
    validate_sessions(sessions, report, today=today, record_level=record_level)
    validate_absences(absences, report)
    if report.ok:
        validate_consistency(athletes, sessions, absences, report)
    for issue in report.warnings:
        logger.warning("[validation] %s.%s: %s rows – %s", issue.table, issue.check, issue.n_rows, issue.message)
    if raise_on_error and not report.ok:
        raise DataValidationError(report)
    return report


def main() -> None:
    import json
    import sys

    from src.config import load_config
    from src.data.repository import load_dataset

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ds = load_dataset(load_config())
    report = validate_dataset(ds.athletes, ds.sessions, ds.absences, raise_on_error=False)
    print(report.summary())
    print(json.dumps(report.n_rows))
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
