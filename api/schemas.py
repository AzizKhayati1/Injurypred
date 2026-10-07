"""Request/response schemas for the REST API (Pydantic v2)."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ATHLETE_ID = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$", examples=["A102"])
RiskCategory = Literal["LOW", "MODERATE", "HIGH", "VERY_HIGH"]


# ----------------------------------------------------------------------------- requests
class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"athlete_id": "A102", "prediction_date": "2026-09-15"}})

    athlete_id: str = ATHLETE_ID
    prediction_date: date | None = Field(
        default=None, description="Prediction made at the end of this day (data ≤ this date). Default: latest data.")
    include_explanation: bool = True
    top_k: int = Field(default=5, ge=1, le=15)


class BatchPredictRequest(BaseModel):
    athlete_ids: list[str] | None = Field(default=None, max_length=500, description="Default: all athletes")
    prediction_date: date | None = None
    include_explanation: bool = False
    top_k: int = Field(default=5, ge=1, le=15)


class ScenarioIn(BaseModel):
    name: str | None = Field(default=None, max_length=80)
    total_load_pct: float = Field(default=0, ge=-100, le=200, description="% change of all external load metrics")
    sprint_load_pct: float = Field(default=0, ge=-100, le=300, description="% change of sprint distance/count")
    hsr_pct: float = Field(default=0, ge=-100, le=300, description="% change of high-speed running distance")
    duration_pct: float = Field(default=0, ge=-90, le=200, description="% change of session duration (and volume)")
    rpe_delta: float = Field(default=0, ge=-5, le=5, description="change of session RPE (CR-10 points)")
    rest_day: bool = Field(default=False, description="replace the planned session with a recovery day")


class PlannedSessionIn(BaseModel):
    match_or_training: Literal["match", "training"] = "training"
    duration_minutes: float = Field(gt=0, le=240)
    minutes_played: float = Field(default=0, ge=0, le=240)
    total_distance: float = Field(ge=0, le=20000)
    high_speed_running_distance: float = Field(ge=0)
    sprint_distance: float = Field(ge=0)
    sprint_count: float = Field(default=0, ge=0)
    accelerations: float = Field(default=0, ge=0)
    decelerations: float = Field(default=0, ge=0)
    player_load: float = Field(ge=0)
    max_speed: float = Field(ge=0, le=45)
    perceived_exertion: float = Field(ge=0, le=10)

    @model_validator(mode="after")
    def _bands(self):
        if self.high_speed_running_distance > self.total_distance:
            raise ValueError("high_speed_running_distance cannot exceed total_distance")
        if self.sprint_distance > self.high_speed_running_distance:
            raise ValueError("sprint_distance cannot exceed high_speed_running_distance")
        return self


class SimulateRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "athlete_id": "A102", "prediction_date": "2026-09-15",
        "scenarios": [{"sprint_load_pct": 20}, {"sprint_load_pct": -20}, {"rest_day": True}]}})

    athlete_id: str = ATHLETE_ID
    prediction_date: date | None = None
    scenarios: list[ScenarioIn] | None = Field(default=None, max_length=20,
                                               description="Default: the presets in configs/config.yaml")
    planned_session: PlannedSessionIn | None = Field(
        default=None, description="Tomorrow's planned session. Default: the athlete's typical recent session.")


# ----------------------------------------------------------------------------- responses
class RiskFactor(BaseModel):
    feature: str
    label: str
    group: str | None = None
    value: float | None = None
    impact: float
    direction: Literal["increases_risk", "decreases_risk"]
    explanation: str


class BaselineMetric(BaseModel):
    current_7d: float | None
    baseline_weekly: float | None
    deviation_pct: float | None
    z_score: float | None
    status: str


class CalibrationInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    method: str | None = None
    test_brier: float | None = None
    test_ece: float | None = None
    test_roc_auc: float | None = None
    bin_range: list[float] | None = None
    bin_observed_rate: float | None = None
    bin_n: int | None = None
    interpretation: str | None = None


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    athlete_id: str
    prediction_date: date
    prediction_horizon_days: int
    risk_window: dict[str, date]
    risk_probability: float = Field(ge=0, le=1)
    risk_category: RiskCategory
    risk_percentile: float
    relative_risk: float
    base_rate: float
    thresholds: dict[str, float]
    top_risk_factors: list[RiskFactor]
    protective_factors: list[RiskFactor]
    baseline_comparison: dict[str, BaselineMetric]
    calibration: CalibrationInfo
    summary: str
    warnings: list[str]
    model_version: str
    model_type: str
    build_id: str
    data_origin: str
    disclaimer: str


class SkippedPrediction(BaseModel):
    athlete_id: str
    reason: str
    status_code: int


class BatchPredictResponse(BaseModel):
    prediction_date: date
    n_requested: int
    n_predicted: int
    summary: dict
    results: list[PredictResponse]
    skipped: list[SkippedPrediction]


class ScenarioResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    risk_probability: float
    risk_category: RiskCategory
    delta_probability: float
    delta_relative_pct: float | None
    feature_changes: dict


class SimulateResponse(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    athlete_id: str
    as_of_date: date
    simulated_day: date
    current_risk: dict
    planned_session: dict
    baseline: dict
    scenarios: list[ScenarioResult]
    disclaimer: str
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_version: str | None = None
    build_id: str | None = None
    n_athletes: int | None = None
    data_range: list[str] | None = None
    loaded_at: str | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    detail: str
