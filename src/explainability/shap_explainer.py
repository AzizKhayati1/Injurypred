"""SHAP-based local explanations for any :class:`~src.models.estimators.RiskModel`.

* Tree models (LightGBM, XGBoost, Random Forest) → ``shap.TreeExplainer``
* Linear models (logistic regression, discrete-time hazard) → ``shap.LinearExplainer``

Contributions are additive in the model's *raw* output space (log-odds for
boosting / logistic models, probability for random forests), i.e. before
calibration. Calibration is a monotone transform, so the sign and ranking of
contributions are preserved; we report the raw values as ``impact`` and state
the space in ``output_space``.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelExplainer:
    def __init__(self, risk_model, background: pd.DataFrame | None = None, max_background: int = 200):
        import shap

        self.model = risk_model
        self.features = risk_model.features
        est = risk_model.estimator
        self.pre = Pipeline(est.steps[:-1]) if isinstance(est, Pipeline) and len(est.steps) > 1 else None
        self.final = risk_model.final_estimator
        kind = type(self.final).__name__
        self.kind = kind
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if kind in ("LGBMClassifier", "XGBClassifier", "RandomForestClassifier"):
                self.explainer = shap.TreeExplainer(self.final)
                self.output_space = "probability" if kind == "RandomForestClassifier" else "log_odds"
            elif kind == "LogisticRegression":
                if background is None:
                    raise ValueError("linear explainer needs a background sample")
                bg = background[self.features].astype("float64").sample(
                    min(max_background, len(background)), random_state=0)
                self.explainer = shap.LinearExplainer(self.final, self._transform(bg))
                self.output_space = "log_odds" + ("_daily_hazard" if risk_model.hazard else "")
            else:  # pragma: no cover
                raise TypeError(f"no explainer for {kind}")

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features].astype("float64")
        return self.pre.transform(X) if self.pre is not None else X

    def explain(self, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Return (contributions frame n × features, base value)."""
        Xt = self._transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = self.explainer.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        base = self.explainer.expected_value
        base = float(np.ravel(base)[-1])
        return pd.DataFrame(sv, index=X.index, columns=self.features), base

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        contrib, _ = self.explain(X)
        return contrib.abs().mean().sort_values(ascending=False)
