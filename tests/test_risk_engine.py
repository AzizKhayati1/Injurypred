import numpy as np
import pytest

from src.risk.engine import RiskEngine, resolve_thresholds

RISK_CFG = {"mode": "relative", "relative_multipliers": {"MODERATE": 1.5, "HIGH": 2.5, "VERY_HIGH": 4.0},
            "absolute_thresholds": {"MODERATE": 0.2, "HIGH": 0.4, "VERY_HIGH": 0.7}}


def test_relative_thresholds():
    th = resolve_thresholds(RISK_CFG, 0.08)
    assert th == pytest.approx({"MODERATE": 0.12, "HIGH": 0.2, "VERY_HIGH": 0.32})


def test_absolute_thresholds():
    th = resolve_thresholds({**RISK_CFG, "mode": "absolute"}, 0.08)
    assert th == {"MODERATE": 0.2, "HIGH": 0.4, "VERY_HIGH": 0.7}


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        resolve_thresholds({**RISK_CFG, "mode": "absolute",
                            "absolute_thresholds": {"MODERATE": 0.5, "HIGH": 0.4, "VERY_HIGH": 0.7}}, 0.1)


@pytest.mark.parametrize("p,cat", [(0.0, "LOW"), (0.199, "LOW"), (0.2, "MODERATE"), (0.39, "MODERATE"),
                                   (0.4, "HIGH"), (0.7, "VERY_HIGH"), (1.0, "VERY_HIGH")])
def test_categorize_boundaries(p, cat):
    e = RiskEngine({"MODERATE": 0.2, "HIGH": 0.4, "VERY_HIGH": 0.7}, 0.1, np.linspace(0, 1, 101))
    assert e.categorize(p) == cat
    assert e.categorize_many(np.array([p]))[0] == cat


def test_assessment_fields():
    e = RiskEngine({"MODERATE": 0.12, "HIGH": 0.2, "VERY_HIGH": 0.32}, 0.08, np.linspace(0, 0.5, 201))
    a = e.assess(0.24).to_dict()
    assert a["risk_category"] == "HIGH"
    assert a["relative_risk"] == pytest.approx(3.0)
    assert 0 <= a["risk_percentile"] <= 100
    assert e.percentile(0.5) == 100.0
