"""Unit tests for the pure workload calculations."""

import numpy as np
import pandas as pd
import pytest

from src.features import workload as W


@pytest.fixture
def s():
    return pd.Series(np.arange(1, 36, dtype=float), index=pd.date_range("2025-01-01", periods=35))


def test_rolling_windows_are_trailing_and_inclusive(s):
    r7 = W.rolling_sum(s, 7)
    assert r7.iloc[6] == sum(range(1, 8))
    assert r7.iloc[34] == sum(range(29, 36))
    assert W.rolling_sum(s, 28).iloc[34] == sum(range(8, 36))


def test_acwr_coupled_matches_definition(s):
    acwr = W.acwr_rolling(s, 7, 28)
    expected = sum(range(29, 36)) / (sum(range(8, 36)) / 4)
    assert acwr.iloc[34] == pytest.approx(expected)


def test_acwr_constant_load_is_one():
    c = pd.Series(100.0, index=pd.date_range("2025-01-01", periods=60))
    assert W.acwr_rolling(c).iloc[-1] == pytest.approx(1.0)
    assert W.acwr_uncoupled(c).iloc[-1] == pytest.approx(1.0)
    assert W.acwr_ewma(c).iloc[-1] == pytest.approx(1.0)


def test_acwr_uncoupled_excludes_acute_week(s):
    val = W.acwr_uncoupled(s, 7, 28).iloc[34]
    chronic_weekly = sum(range(8, 29)) / 3
    assert val == pytest.approx(sum(range(29, 36)) / chronic_weekly)


def test_acwr_zero_chronic_is_nan_not_inf():
    z = pd.Series(0.0, index=pd.date_range("2025-01-01", periods=30))
    assert W.acwr_rolling(z).isna().all()


def test_ewma_recursive_form():
    x = pd.Series([10.0, 0.0, 0.0, 20.0])
    lam = 2 / (7 + 1)
    e = [10.0]
    for v in x.iloc[1:]:
        e.append(lam * v + (1 - lam) * e[-1])
    assert np.allclose(W.ewma(x, 7).values, e)


def test_monotony_and_strain():
    week = pd.Series([100, 200, 100, 200, 100, 200, 0], dtype=float)
    m = W.monotony(week, 7).iloc[-1]
    assert m == pytest.approx(week.mean() / week.std(ddof=0))
    assert W.strain(week, 7).iloc[-1] == pytest.approx(week.sum() * m)
    flat = pd.Series([100.0] * 7)
    assert np.isnan(W.monotony(flat, 7).iloc[-1])  # sd = 0 → undefined, not infinite


def test_week_over_week_change():
    x = pd.Series([100.0] * 7 + [150.0] * 7)
    assert W.week_over_week_change(x, 7).iloc[-1] == pytest.approx(0.5)


def test_consecutive_true_and_days_since():
    flag = pd.Series([True, True, False, True, True, True, False])
    assert W.consecutive_true(flag).tolist() == [1, 2, 0, 1, 2, 3, 0]
    ds = W.days_since(pd.Series([False, True, False, False, True, False]))
    assert ds.iloc[0] != ds.iloc[0]  # NaN before first event
    assert ds.iloc[1:].tolist() == [0, 1, 2, 0, 1]


def test_consecutive_true_dataframe():
    df = pd.DataFrame({"a": [True, True, False], "b": [False, True, True]})
    out = W.consecutive_true(df)
    assert out["a"].tolist() == [1, 2, 0]
    assert out["b"].tolist() == [0, 1, 2]


def test_forward_window_sum_looks_only_forward():
    x = pd.Series([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    fwd = W.forward_window_sum(x, 3)
    assert fwd.tolist()[:3] == [1.0, 1.0, 0.0]  # t=2 itself is NOT in (t, t+3]
    assert fwd.iloc[-1] != fwd.iloc[-1]          # no future after the last day
