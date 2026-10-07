from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas import BatchPredictRequest, BatchPredictResponse, ErrorResponse, PredictRequest, PredictResponse
from src.service import RiskService

router = APIRouter(tags=["predictions"])

ERRORS = {404: {"model": ErrorResponse, "description": "Unknown athlete"},
          409: {"model": ErrorResponse, "description": "Athlete unavailable (currently absent)"},
          422: {"model": ErrorResponse, "description": "Invalid input / date out of range"}}


@router.post("/predict", response_model=PredictResponse, responses=ERRORS,
             summary="7-day availability risk for one athlete, with explanation")
def predict(req: PredictRequest, svc: RiskService = Depends(get_service)):
    """Estimate the probability that the athlete has a new injury/illness absence in the 7 days **after**
    `prediction_date`, using only data up to the end of that day. Statistical estimate – not a diagnosis."""
    return svc.predict(req.athlete_id, req.prediction_date, explain=req.include_explanation, top_k=req.top_k)


@router.post("/predict/batch", response_model=BatchPredictResponse, responses=ERRORS,
             summary="Availability risk for many athletes on one date")
def predict_batch(req: BatchPredictRequest, svc: RiskService = Depends(get_service)):
    return svc.predict_batch(req.athlete_ids, req.prediction_date, explain=req.include_explanation, top_k=req.top_k)
