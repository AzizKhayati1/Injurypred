"""Small shared helpers."""

from __future__ import annotations

import functools
import math

import numpy as np
import pandas as pd


def clean_json(obj):
    """Recursively convert numpy/pandas scalars to Python and NaN/inf to None (strict-JSON safe)."""
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean_json(v) for v in obj]
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NaT or obj is pd.NA:
        return None
    return obj


def json_safe(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return clean_json(fn(*args, **kwargs))
    return wrapper
