from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_service
from src.service import RiskService

router = APIRouter(prefix="/squad", tags=["squad"])


@router.get("/overview", summary="Squad-level availability and risk summary")
def overview(date: date | None = None, team_id: str | None = Query(default=None, examples=["T1"]),
             svc: RiskService = Depends(get_service)):
    return svc.squad_overview(date, team_id)


@router.get("/heatmap", summary="Athletes × dates matrix of predicted risk")
def heatmap(start: date | None = None, end: date | None = None, team_id: str | None = None,
            svc: RiskService = Depends(get_service)):
    return svc.heatmap(start, end, team_id)
