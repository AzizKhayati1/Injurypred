from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from src.service import RiskService

router = APIRouter(tags=["model"])


@router.get("/model/info", summary="Model version, lineage, training data and risk-engine configuration")
def model_info(svc: RiskService = Depends(get_service)):
    return svc.model_info()


@router.get("/model/metrics", summary="Validation/test metrics and model comparison")
def model_metrics(svc: RiskService = Depends(get_service)):
    return svc.model_metrics()


@router.get("/features", summary="Feature catalogue with definitions and training ranges")
def features(svc: RiskService = Depends(get_service)):
    return svc.feature_catalog()
