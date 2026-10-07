from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.routes.predictions import ERRORS
from api.schemas import SimulateRequest, SimulateResponse
from src.service import RiskService

router = APIRouter(tags=["simulation"])


@router.post("/simulate", response_model=SimulateResponse, responses=ERRORS,
             summary="What-if: recompute risk for modified training load tomorrow")
def simulate(req: SimulateRequest, svc: RiskService = Depends(get_service)):
    """Appends a planned session for the day after `prediction_date`, applies each scenario
    (e.g. +20 % sprint load, extra recovery day), recomputes features and re-scores.

    **Model simulation only** – not causal evidence and not medical advice."""
    scenarios = [s.model_dump() for s in req.scenarios] if req.scenarios else None
    planned = req.planned_session.model_dump() if req.planned_session else None
    return svc.simulate(req.athlete_id, req.prediction_date, scenarios, planned)
