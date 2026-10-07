from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from api.dependencies import get_service
from src.service import RiskService

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


def _dir(svc: RiskService) -> Path:
    return Path(svc.cfg["paths"]["reports_dir"]) / "monitoring"


@router.get("/drift", summary="Feature / prediction drift and performance report (JSON)")
def drift(refresh: bool = False, start: date | None = None, end: date | None = None,
          svc: RiskService = Depends(get_service)):
    path = _dir(svc) / "drift_report.json"
    if refresh or start or end or not path.exists():
        from src.monitoring.drift import run_monitoring

        return run_monitoring(service=svc, start=start, end=end)
    return json.loads(path.read_text())


@router.get("/drift/html", response_class=HTMLResponse, summary="Drift report (HTML)")
def drift_html(svc: RiskService = Depends(get_service)):
    path = _dir(svc) / "drift_report.html"
    if not path.exists():
        from src.monitoring.drift import run_monitoring

        run_monitoring(service=svc)
    return HTMLResponse(path.read_text())
