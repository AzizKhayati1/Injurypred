"""Dependency wiring: one RiskService per process, created at start-up."""

from __future__ import annotations

from fastapi import HTTPException, Request

from src.service import RiskService


def get_service(request: Request) -> RiskService:
    svc = getattr(request.app.state, "service", None)
    if svc is None:
        err = getattr(request.app.state, "startup_error", "service not initialised")
        raise HTTPException(status_code=503, detail=f"model service unavailable: {err}")
    return svc
