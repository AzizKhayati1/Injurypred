"""Pure workload calculations.

Every function takes a **complete daily calendar** (rest days = 0 load) as a
``pd.Series`` (one athlete) or a ``pd.DataFrame`` (dates × athletes) and returns
an object of the same shape. All windows are *trailing* and include the current
day ``t`` only — values at ``t`` never depend on data after ``t``.

Definitions
-----------
* Rolling acute / chronic load: trailing sums over 7 / 28 days.
* Coupled ACWR (Hulin et al., 2014; Gabbett, 2016)::

      ACWR_t = sum(L[t-6..t]) / (sum(L[t-27..t]) / 4)

  The acute week is part of the chronic window ("mathematical coupling",
  Lolli et al., 2019), so the two terms are not independent.
* Uncoupled ACWR: chronic = mean weekly load over days t-27..t-7.
* EWMA ACWR (Williams et al., 2017):
  ``EWMA_t = λ·L_t + (1-λ)·EWMA_{t-1}``, ``λ = 2/(N+1)``.
* Monotony = mean(daily load, 7 d) / sd(daily load, 7 d); Strain = weekly load ×
  monotony (Foster, 1998).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Frame = pd.Series | pd.DataFrame


def rolling_sum(x: Frame, window: int, min_periods: int = 1) -> Frame:
    return x.rolling(window, min_periods=min_periods).sum()


def rolling_mean(x: Frame, window: int, min_periods: int = 1) -> Frame:
    return x.rolling(window, min_periods=min_periods).mean()


def ewma(x: Frame, span: int) -> Frame:
    """Exponentially weighted moving average with λ = 2/(span+1) (recursive form)."""
    return x.ewm(span=span, adjust=False).mean()


def safe_ratio(num: Frame, den: Frame, floor: float = 1e-6) -> Frame:
    """num/den, NaN where the denominator is (near) zero instead of ±inf."""
    den = den.where(den.abs() > floor)
    return num / den


def acwr_rolling(x: Frame, acute: int = 7, chronic: int = 28, floor: float = 1e-6) -> Frame:
    """Coupled rolling-average ACWR: acute weekly load / average weekly load over the chronic window."""
    weeks = chronic / acute
    return safe_ratio(rolling_sum(x, acute), rolling_sum(x, chronic) / weeks, floor)


def acwr_uncoupled(x: Frame, acute: int = 7, chronic: int = 28, floor: float = 1e-6) -> Frame:
    """Uncoupled ACWR: the chronic window excludes the acute week (days t-27 … t-7)."""
    weeks = (chronic - acute) / acute
    chronic_load = rolling_sum(x.shift(acute), chronic - acute)
    return safe_ratio(rolling_sum(x, acute), chronic_load / weeks, floor)


def acwr_ewma(x: Frame, acute_span: int = 7, chronic_span: int = 28, floor: float = 1e-6) -> Frame:
    return safe_ratio(ewma(x, acute_span), ewma(x, chronic_span), floor)


def monotony(x: Frame, window: int = 7, cap: float = 10.0) -> Frame:
    mean = x.rolling(window, min_periods=window).mean()
    sd = x.rolling(window, min_periods=window).std(ddof=0)
    return safe_ratio(mean, sd).clip(upper=cap)


def strain(x: Frame, window: int = 7) -> Frame:
    return rolling_sum(x, window) * monotony(x, window)


def week_over_week_change(x: Frame, window: int = 7, floor: float = 1e-6, clip: float = 5.0) -> Frame:
    """(this week − previous week) / previous week, using trailing 7-day sums."""
    this_week = rolling_sum(x, window)
    prev_week = this_week.shift(window)
    return safe_ratio(this_week - prev_week, prev_week, floor).clip(-1.0, clip)


def consecutive_true(flag: Frame) -> Frame:
    """Length of the run of True values ending at each day (0 if False)."""
    f = flag.astype(bool).astype(int)
    groups = (f == 0).cumsum()
    if isinstance(f, pd.Series):
        return f.groupby(groups).cumsum()
    return f.apply(lambda col: col.groupby(groups[col.name]).cumsum())


def days_since(flag: Frame, cap: float | None = None) -> Frame:
    """Days since the most recent True (0 if True today). NaN before the first True."""
    f = flag.astype(bool)
    idx = pd.Series(np.arange(len(f)), index=f.index)
    if isinstance(f, pd.Series):
        last = idx.where(f).ffill()
        out = idx - last
    else:
        pos = pd.DataFrame(np.repeat(idx.values[:, None], f.shape[1], axis=1), index=f.index, columns=f.columns)
        last = pos.where(f).ffill()
        out = pos - last
    return out.clip(upper=cap) if cap is not None else out


def forward_window_sum(x: Frame, horizon: int) -> Frame:
    """Sum over the *future* window (t+1 … t+horizon). Used for LABELS ONLY."""
    reversed_ = x.iloc[::-1]
    fwd = reversed_.rolling(horizon, min_periods=1).sum().iloc[::-1]  # t … t+h-1
    return fwd.shift(-1)                                               # t+1 … t+h
