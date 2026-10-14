from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query

from api.dependencies import get_service
from src.service import RiskService

router = APIRouter(tags=["players & injuries"])
AID = Path(pattern=r"^[A-Za-z0-9_-]{1,32}$", examples=["A172"])


@router.get("/players", summary="Player cards: bio, club, availability status, latest risk")
def players(team_id: str | None = Query(default=None, examples=["T1"]), svc: RiskService = Depends(get_service)):
    return {"players": svc.list_players(team_id)}


@router.get("/players/{athlete_id}", summary="Player profile: bio, season/career stats, injury summary")
def player(athlete_id: str = AID, svc: RiskService = Depends(get_service)):
    return svc.player_profile(athlete_id)


@router.get("/players/{athlete_id}/injuries", summary="Clinical injury history (as recorded by medical staff)")
def injuries(athlete_id: str = AID, svc: RiskService = Depends(get_service)):
    return svc.injury_history(athlete_id)


@router.get("/players/{athlete_id}/prevention", summary="Evidence-based prevention recommendations (decision support)")
def prevention(athlete_id: str = AID, svc: RiskService = Depends(get_service)):
    return svc.prevention(athlete_id)


@router.get("/clubs", summary="Clubs with squad availability and risk")
def clubs(svc: RiskService = Depends(get_service)):
    return {"clubs": svc.clubs()}


@router.get("/injury-report", summary="League injury report: current absences, recent news, injury statistics")
def injury_report(svc: RiskService = Depends(get_service)):
    return svc.injury_report()


@router.get("/injury-catalogue", summary="Injury classification used for clinical records")
def injury_catalogue(svc: RiskService = Depends(get_service)):
    return {"diagnoses": svc.injury_catalogue()}
