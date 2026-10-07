"""API integration tests (FastAPI TestClient with an injected small service)."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client(service):
    app.state.service = service
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_contract(client, available_athlete_date):
    aid, date = available_athlete_date
    r = client.post("/predict", json={"athlete_id": aid, "prediction_date": date})
    assert r.status_code == 200, r.text
    body = r.json()
    for k in ("risk_probability", "risk_category", "risk_percentile", "prediction_horizon_days", "top_risk_factors",
              "protective_factors", "baseline_comparison", "model_version", "calibration", "disclaimer"):
        assert k in body
    assert 0 <= body["risk_probability"] <= 1
    assert body["risk_category"] in ("LOW", "MODERATE", "HIGH", "VERY_HIGH")
    assert body["prediction_horizon_days"] == 7
    for f in body["top_risk_factors"]:
        assert f["impact"] > 0 and f["direction"] == "increases_risk"
    for f in body["protective_factors"]:
        assert f["impact"] < 0 and f["direction"] == "decreases_risk"


def test_predict_errors(client, service):
    assert client.post("/predict", json={"athlete_id": "NOPE"}).status_code == 404
    assert client.post("/predict", json={"athlete_id": "A101", "prediction_date": "2099-01-01"}).status_code == 422
    assert client.post("/predict", json={"athlete_id": "bad id!"}).status_code == 422
    assert client.post("/predict", json={"prediction_date": "2025-01-01"}).status_code == 422
    f = service.features
    unavailable = f[~f["available"] & f["in_season"]].index[0]
    r = client.post("/predict", json={"athlete_id": unavailable[0], "prediction_date": str(unavailable[1].date())})
    assert r.status_code == 409


def test_batch(client, available_athlete_date):
    _, date = available_athlete_date
    r = client.post("/predict/batch", json={"prediction_date": date})
    assert r.status_code == 200
    body = r.json()
    assert body["n_predicted"] + len(body["skipped"]) == body["n_requested"]


def test_simulate(client, available_athlete_date):
    aid, date = available_athlete_date
    r = client.post("/simulate", json={"athlete_id": aid, "prediction_date": date,
                                       "scenarios": [{"sprint_load_pct": 20}, {"rest_day": True}]})
    assert r.status_code == 200, r.text
    assert len(r.json()["scenarios"]) == 2
    bad = client.post("/simulate", json={"athlete_id": aid, "scenarios": [{"sprint_load_pct": 5000}]})
    assert bad.status_code == 422


def test_simulate_with_planned_session_validation(client, available_athlete_date):
    aid, date = available_athlete_date
    planned = {"duration_minutes": 70, "total_distance": 5000, "high_speed_running_distance": 6000,
               "sprint_distance": 10, "player_load": 500, "max_speed": 30, "perceived_exertion": 6}
    assert client.post("/simulate", json={"athlete_id": aid, "planned_session": planned}).status_code == 422
    planned["high_speed_running_distance"] = 300
    r = client.post("/simulate", json={"athlete_id": aid, "prediction_date": date, "planned_session": planned})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("path", ["/athletes", "/athletes/{aid}", "/athletes/{aid}/risk-history",
                                  "/athletes/{aid}/workload", "/model/info", "/model/metrics", "/features",
                                  "/squad/overview", "/squad/heatmap"])
def test_get_endpoints(client, available_athlete_date, path):
    aid, _ = available_athlete_date
    r = client.get(path.format(aid=aid))
    assert r.status_code == 200, r.text


def test_risk_history_range(client, available_athlete_date):
    aid, date = available_athlete_date
    r = client.get(f"/athletes/{aid}/risk-history", params={"end": date})
    hist = r.json()["history"]
    assert hist[-1]["date"] == date and len(hist) == 91


def test_openapi_documents_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/health", "/predict", "/predict/batch", "/athletes/{athlete_id}", "/athletes/{athlete_id}/risk-history",
              "/athletes/{athlete_id}/workload", "/simulate", "/model/info", "/model/metrics", "/features"):
        assert p in paths
