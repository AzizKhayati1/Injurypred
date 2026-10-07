"""Model definitions and the serving wrapper.

All models expose the same interface through :class:`RiskModel`:
``predict_proba(features_frame) -> calibrated P(event in next 7 days)``.

Class imbalance (≈7 % positives) is handled by *re-weighting* during fitting
(``class_weight="balanced"`` / ``scale_pos_weight``) so that the learners do
not ignore the minority class. Re-weighting inflates predicted probabilities,
so every model is **re-calibrated on the validation period** (Platt scaling on
the logit by default) before its probabilities are shown to anyone.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class ProbabilityCalibrator:
    """Platt (logistic on the logit) or isotonic calibration of raw scores."""

    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.model_ = None

    def fit(self, raw: np.ndarray, y: np.ndarray) -> ProbabilityCalibrator:
        if self.method == "none":
            return self
        if self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(raw, y)
        else:
            self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw).reshape(-1, 1), y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            return np.asarray(raw, dtype=float)
        if self.method == "isotonic":
            return self.model_.predict(raw)
        return self.model_.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]


class RiskModel:
    """A fitted estimator + calibrator + the exact feature list it expects."""

    def __init__(self, name: str, estimator, features: list[str], calibrator: ProbabilityCalibrator | None = None,
                 horizon: int = 7, hazard: bool = False):
        self.name = name
        self.estimator = estimator
        self.features = list(features)
        self.calibrator = calibrator
        self.horizon = horizon
        self.hazard = hazard

    def _matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.features if f not in X.columns]
        if missing:
            raise KeyError(f"missing features: {missing[:10]}")
        return X[self.features].astype("float64")

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        p = self.estimator.predict_proba(self._matrix(X))[:, 1]
        if self.hazard:  # daily hazard → P(≥1 event within horizon) under a constant hazard
            p = 1.0 - (1.0 - p) ** self.horizon
        return p

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(X)
        return self.calibrator.transform(raw) if self.calibrator is not None else raw

    def fit_calibrator(self, X: pd.DataFrame, y: np.ndarray, method: str) -> RiskModel:
        self.calibrator = ProbabilityCalibrator(method).fit(self.predict_raw(X), np.asarray(y))
        return self

    @property
    def final_estimator(self):
        return self.estimator.steps[-1][1] if isinstance(self.estimator, Pipeline) else self.estimator


def _pos_weight(y: np.ndarray, setting) -> float:
    """Positive-class weight. ``auto`` = n_neg/n_pos (fully balanced), ``sqrt`` = its square root."""
    pos = max(int(np.sum(y)), 1)
    ratio = (len(y) - pos) / pos
    if setting in (None, "none"):
        return 1.0
    if setting == "auto":
        return float(ratio)
    if setting == "sqrt":
        return float(np.sqrt(ratio))
    return float(setting)


def build_estimator(name: str, params: dict, y_train: np.ndarray, seed: int):
    params = {k: v for k, v in params.items() if k not in ("enabled", "early_stopping_rounds")}
    if name in ("logistic_regression", "discrete_time_hazard"):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=params.get("C", 0.1), class_weight=params.get("class_weight", "balanced"),
                                       max_iter=params.get("max_iter", 2000))),
        ])
    if name == "random_forest":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clf", RandomForestClassifier(random_state=seed, n_jobs=-1, **params)),
        ])
    if name == "lightgbm":
        import lightgbm as lgb

        params["scale_pos_weight"] = _pos_weight(y_train, params.get("scale_pos_weight", "auto"))
        return lgb.LGBMClassifier(random_state=seed, verbose=-1, n_jobs=4, **params)
    if name == "xgboost":
        import xgboost as xgb

        params["scale_pos_weight"] = _pos_weight(y_train, params.get("scale_pos_weight", "auto"))
        return xgb.XGBClassifier(random_state=seed, n_jobs=4, eval_metric="auc", tree_method="hist", **params)
    raise ValueError(f"unknown model: {name}")


def fit_model(name: str, params: dict, features: list[str], train: pd.DataFrame, valid: pd.DataFrame | None,
              target: str, seed: int, calibration: str = "sigmoid", horizon: int = 7) -> RiskModel:
    """Fit one model on ``train``.

    ``valid`` is used for early stopping (boosting) and probability calibration
    only – never for fitting model weights. With ``valid=None`` boosting models
    use ``params["n_estimators"]`` as-is and no calibrator is fitted.
    """
    hazard = name == "discrete_time_hazard"
    tr = train[train["observable_1d"]] if hazard else train
    y_tr = tr["event_next_1d" if hazard else target].to_numpy()
    X_tr = tr[features].astype("float64")
    est = build_estimator(name, params, y_tr, seed)
    has_valid = valid is not None and len(valid) > 0
    if name == "lightgbm" and has_valid:
        import lightgbm as lgb

        est.fit(X_tr, y_tr, eval_set=[(valid[features].astype("float64"), valid[target].to_numpy())],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(params.get("early_stopping_rounds", 300), verbose=False)])
    elif name == "xgboost" and has_valid:
        est.set_params(early_stopping_rounds=params.get("early_stopping_rounds", 300))
        est.fit(X_tr, y_tr, eval_set=[(valid[features].astype("float64"), valid[target].to_numpy())], verbose=False)
    else:
        est.fit(X_tr, y_tr)
    model = RiskModel(name, est, features, horizon=horizon, hazard=hazard)
    if has_valid:
        model.fit_calibrator(valid, valid[target].to_numpy(), calibration)
    return model


def best_iteration(model: RiskModel) -> int | None:
    est = model.final_estimator
    for attr in ("best_iteration_", "best_iteration"):
        it = getattr(est, attr, None)
        if isinstance(it, int | np.integer) and it > 0:
            return int(it) + (1 if attr == "best_iteration" else 0)
    return None
