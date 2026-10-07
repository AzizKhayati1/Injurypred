"""Pydantic record schemas for raw input data.

These schemas validate *individual records* (e.g. a session posted to an
ingestion endpoint). Whole-table checks (duplicates, chronological ordering,
cross-table consistency) live in :mod:`src.data.validation`.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Physical plausibility limits. Usain Bolt's peak ≈ 44.7 km/h; elite footballers rarely exceed 37 km/h.
MAX_PLAUSIBLE_SPEED_KMH = 45.0
MAX_SESSION_MINUTES = 240.0
MAX_SESSION_DISTANCE_M = 20_000.0


class Position(str, Enum):
    GK = "GK"
    CB = "CB"
    FB = "FB"
    CM = "CM"
    W = "W"
    ST = "ST"


class SessionKind(str, Enum):
    match = "match"
    training = "training"


class SessionRecord(BaseModel):
    """One GPS-tracked session for one athlete."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    session_id: str = Field(min_length=1)
    athlete_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    date: date
    session_type: str
    match_or_training: SessionKind
    duration_minutes: float = Field(gt=0, le=MAX_SESSION_MINUTES)
    minutes_played: float = Field(ge=0, le=MAX_SESSION_MINUTES)
    total_distance: float = Field(ge=0, le=MAX_SESSION_DISTANCE_M)
    high_speed_running_distance: float = Field(ge=0)
    sprint_distance: float = Field(ge=0)
    sprint_count: int = Field(ge=0)
    accelerations: int = Field(ge=0)
    decelerations: int = Field(ge=0)
    max_speed: float = Field(ge=0, le=MAX_PLAUSIBLE_SPEED_KMH, description="km/h")
    player_load: float = Field(ge=0)
    perceived_exertion: float = Field(ge=0, le=10, description="CR-10 RPE")
    training_load: float | None = Field(default=None, ge=0, description="session-RPE load (RPE × min)")
    position: Position | None = None

    @model_validator(mode="after")
    def _consistency(self) -> SessionRecord:
        if self.high_speed_running_distance > self.total_distance + 1e-6:
            raise ValueError("high_speed_running_distance cannot exceed total_distance")
        if self.sprint_distance > self.high_speed_running_distance + 1e-6:
            raise ValueError("sprint_distance cannot exceed high_speed_running_distance "
                             "(sprint band is a subset of the HSR band)")
        if self.minutes_played > self.duration_minutes + 1e-6:
            raise ValueError("minutes_played cannot exceed duration_minutes")
        if self.sprint_count > 0 and self.sprint_distance == 0:
            raise ValueError("sprint_count > 0 but sprint_distance == 0")
        return self


class AbsenceRecord(BaseModel):
    """A time-loss absence (injury or illness)."""

    model_config = ConfigDict(extra="ignore")

    absence_id: str
    athlete_id: str
    onset_date: date
    return_date: date
    days_lost: int = Field(ge=1)
    absence_type: str = Field(pattern="^(injury|illness)$")

    @field_validator("return_date")
    @classmethod
    def _after_onset(cls, v: date, info):
        onset = info.data.get("onset_date")
        if onset is not None and v <= onset:
            raise ValueError("return_date must be after onset_date")
        return v


class AthleteRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    athlete_id: str
    team_id: str
    position: Position
    date_of_birth: date
    prior_injury_count: int = Field(ge=0)
