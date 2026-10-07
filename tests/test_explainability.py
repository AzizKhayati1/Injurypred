import numpy as np
import pandas as pd

from src.explainability.narrative import FORBIDDEN_PHRASES, build_factors, describe_feature
from src.explainability.shap_explainer import ModelExplainer
from src.features.catalog import FEATURE_NAMES


def test_shap_contributions_are_additive(service):
    model = service.model
    X = service.features[service.features["eligible"]].sample(50, random_state=0)
    exp = ModelExplainer(model, background=service.bundle["background"])
    contrib, base = exp.explain(X)
    raw = model.estimator.predict_proba(X[model.features])[:, 1]
    if exp.output_space.startswith("log_odds"):
        recon = 1 / (1 + np.exp(-(contrib.sum(axis=1).to_numpy() + base)))
        assert np.allclose(recon, raw, atol=1e-3)


def test_factor_split_by_sign():
    row = pd.Series({n: 1.0 for n in FEATURE_NAMES})
    contrib = pd.Series(0.0, index=FEATURE_NAMES)
    contrib["acwr_sprint_load"], contrib["rest_days_7d"] = 0.3, -0.2
    pos, neg = build_factors(row, contrib, top_k=5)
    assert [f["feature"] for f in pos] == ["acwr_sprint_load"]
    assert [f["feature"] for f in neg] == ["rest_days_7d"]
    assert pos[0]["direction"] == "increases_risk" and neg[0]["direction"] == "decreases_risk"


def test_language_policy_on_all_features(service, available_athlete_date):
    aid, date = available_athlete_date
    row = service.features.loc[(aid, pd.Timestamp(date))]
    for name in FEATURE_NAMES:
        for direction in ("increases_risk", "decreases_risk"):
            text = describe_feature(name, row, row, direction).lower()
            assert not any(p in text for p in FORBIDDEN_PHRASES), text
            assert "associated with" in text or "no personal baseline" in text


def test_prediction_explanation_language(service, available_athlete_date):
    aid, date = available_athlete_date
    out = service.predict(aid, date)
    text = " ".join([out["summary"], out["disclaimer"]] + [f["explanation"] for f in
                                                           out["top_risk_factors"] + out["protective_factors"]])
    assert "will get injured" not in text.lower()
    assert "not a medical" in text.lower()
