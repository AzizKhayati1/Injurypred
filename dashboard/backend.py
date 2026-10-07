"""Data access for the dashboard.

The dashboard is a *client*: by default it talks to the REST API
(``API_URL``, Docker: ``http://api:8000``). For local development it can run
the same :class:`~src.service.RiskService` in-process
(``DASHBOARD_BACKEND=local``). ``auto`` (default) tries the API first.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class BackendError(RuntimeError):
    pass


class ApiBackend:
    name = "api"

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base, timeout=60.0)

    def _req(self, method: str, path: str, **kw):
        try:
            r = self.client.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise BackendError(f"API unreachable at {self.base}: {exc}") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except Exception:
                detail = r.text
            raise BackendError(str(detail))
        return r.json()

    def health(self):
        return self._req("GET", "/health")

    def athletes(self):
        return self._req("GET", "/athletes")["athletes"]

    def athlete(self, aid):
        return self._req("GET", f"/athletes/{aid}")

    def predict(self, aid, date=None):
        return self._req("POST", "/predict", json={"athlete_id": aid, "prediction_date": date})

    def risk_history(self, aid, start=None, end=None):
        return self._req("GET", f"/athletes/{aid}/risk-history", params=_p(start=start, end=end))

    def workload(self, aid, start=None, end=None):
        return self._req("GET", f"/athletes/{aid}/workload", params=_p(start=start, end=end))

    def simulate(self, aid, date=None, scenarios=None, planned=None):
        return self._req("POST", "/simulate", json={"athlete_id": aid, "prediction_date": date,
                                                     "scenarios": scenarios, "planned_session": planned})

    def squad(self, date=None, team=None):
        return self._req("GET", "/squad/overview", params=_p(date=date, team_id=team))

    def heatmap(self, start=None, end=None, team=None):
        return self._req("GET", "/squad/heatmap", params=_p(start=start, end=end, team_id=team))

    def model_info(self):
        return self._req("GET", "/model/info")

    def model_metrics(self):
        return self._req("GET", "/model/metrics")

    def drift(self):
        return self._req("GET", "/monitoring/drift")


class LocalBackend:
    name = "local (in-process)"

    def __init__(self):
        from src.service import RiskService, ServiceError

        self._err = ServiceError
        self.svc = RiskService()

    def _call(self, fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except self._err as exc:
            raise BackendError(str(exc)) from exc

    def health(self):
        return self.svc.health()

    def athletes(self):
        return self.svc.list_athletes()

    def athlete(self, aid):
        return self._call(self.svc.athlete_profile, aid)

    def predict(self, aid, date=None):
        return self._call(self.svc.predict, aid, date)

    def risk_history(self, aid, start=None, end=None):
        return self._call(self.svc.risk_history, aid, start, end)

    def workload(self, aid, start=None, end=None):
        return self._call(self.svc.workload, aid, start, end)

    def simulate(self, aid, date=None, scenarios=None, planned=None):
        return self._call(self.svc.simulate, aid, date, scenarios, planned)

    def squad(self, date=None, team=None):
        return self._call(self.svc.squad_overview, date, team)

    def heatmap(self, start=None, end=None, team=None):
        return self._call(self.svc.heatmap, start, end, team)

    def model_info(self):
        return self.svc.model_info()

    def model_metrics(self):
        return self.svc.model_metrics()

    def drift(self):
        from src.monitoring.drift import run_monitoring

        path = Path(self.svc.cfg["paths"]["reports_dir"]) / "monitoring" / "drift_report.json"
        if path.exists():
            import json

            return json.loads(path.read_text())
        return run_monitoring(service=self.svc)


def _p(**kw):
    return {k: v for k, v in kw.items() if v is not None}


def make_backend():
    mode = os.environ.get("DASHBOARD_BACKEND", "auto")
    url = os.environ.get("API_URL", "http://localhost:8000")
    if mode in ("api", "auto"):
        api = ApiBackend(url)
        try:
            api.health()
            return api
        except BackendError:
            if mode == "api":
                raise
    return LocalBackend()
