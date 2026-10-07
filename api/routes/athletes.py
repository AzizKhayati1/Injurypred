from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Path, Query

from api.dependencies import get_service
from src.service import RiskService

router = APIRouter(prefix="/athletes", tags=["athletes"])
AID = Path(pattern=r"^[A-Za-z0-9_-]{1,32}$", examples=["A102"])


@router.get("", summary="List athletes with their latest availability and risk")
def list_athletes(team_id: str | None = Query(default=None, examples=["T1"]), svc: RiskService = Depends(get_service)):
    return {"athletes": svc.list_athletes(team_id)}


@router.get("/{athlete_id}", summary="Athlete profile, injury history and personal workload baseline")
def athlete(athlete_id: str = AID, svc: RiskService = Depends(get_service)):
    return svc.athlete_profile(athlete_id)


@router.get("/{athlete_id}/risk-history", summary="Daily predicted risk over a date range")
def risk_history(athlete_id: str = AID, start: date | None = None, end: date | None = None,
                 svc: RiskService = Depends(get_service)):
    return svc.risk_history(athlete_id, start, end)


@router.get("/{athlete_id}/workload", summary="Daily workload and rolling workload features")
def workload(athlete_id: str = AID, start: date | None = None, end: date | None = None,
             svc: RiskService = Depends(get_service)):
    return svc.workload(athlete_id, start, end)
