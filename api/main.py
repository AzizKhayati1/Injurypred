"""PlayerLoad-Risk REST API.

    uvicorn api.main:app --reload            # http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import athletes, model, monitoring, predictions, simulation, squad
from api.schemas import HealthResponse
from src import __version__
from src.service import RiskService, ServiceError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")

DESCRIPTION = """
**PlayerLoad-Risk** estimates an athlete's *statistical* short-term (7-day) availability risk from GPS-style
training and match-load data, with SHAP explanations, athlete-specific baselines and what-if simulation.

> ⚠️ This project estimates statistical availability risk from workload patterns. **It is not a medical diagnostic
> system** and does not predict any individual injury with certainty. All bundled data is **synthetic**.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "service", None) is None:  # tests may inject a service
        try:
            app.state.service = RiskService()
        except Exception as exc:  # keep the process alive so /health can report the problem
            logger.exception("Failed to initialise RiskService")
            app.state.service = None
            app.state.startup_error = str(exc)
    yield


app = FastAPI(title="PlayerLoad-Risk API", version=__version__, description=DESCRIPTION, lifespan=lifespan,
              contact={"name": "PlayerLoad-Risk"}, license_info={"name": "MIT"})
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{1000 * (time.perf_counter() - t0):.1f}"
    return response


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse, tags=["health"], summary="Liveness / readiness")
def health(request: Request):
    svc = getattr(request.app.state, "service", None)
    if svc is None:
        return JSONResponse(status_code=503, content={"status": "unavailable", "model_loaded": False,
                                                      "detail": getattr(request.app.state, "startup_error", None)})
    return svc.health()


@app.get("/", include_in_schema=False)
def root():
    return {"name": "PlayerLoad-Risk API", "docs": "/docs", "health": "/health",
            "disclaimer": "Statistical availability-risk estimates from workload data; not a medical diagnosis."}


for r in (predictions.router, simulation.router, athletes.router, model.router, squad.router, monitoring.router):
    app.include_router(r)
