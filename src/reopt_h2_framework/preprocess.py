"""Generic preparation steps for the hourly inputs (``docs/data_pipeline.md``).

None of these functions downloads data; they turn records the user has obtained into model inputs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def to_hourly(records: pd.DataFrame, time_col: str, value_col: str, how: str = "mean",
              expected_hours: Optional[int] = None, records_per_hour: Optional[int] = None) -> pd.Series:
    """Aggregate interval records (e.g. 5- or 15-minute prices) to hourly values.

    ``how`` is "mean" (prices, power) or "sum" (energy per interval). Missing time stamps and hours
    with no record raise an error rather than being dropped or filled silently. ``records_per_hour``
    (e.g. 4 for 15-minute data) also requires every hour to be complete; ``expected_hours`` checks
    the length of the result.
    """
    if how not in ("mean", "sum"):
        raise ValueError("how must be 'mean' or 'sum'")
    t = pd.to_datetime(records[time_col])
    if t.isna().any():
        raise ValueError(f"{time_col} has {int(t.isna().sum())} missing time stamps")
    values = pd.to_numeric(records[value_col], errors="raise")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{value_col} has non-finite values")
    s = pd.Series(values.to_numpy(), index=t).sort_index()
    if s.index.duplicated().any():
        raise ValueError(f"{time_col} has duplicated time stamps")
    hourly = s.resample("h").agg(how)
    counts = s.resample("h").count()
    if (counts == 0).any():
        raise ValueError(f"{int((counts == 0).sum())} hours have no record, first {counts[counts == 0].index[0]}")
    if records_per_hour is not None and (counts != records_per_hour).any():
        bad = counts[counts != records_per_hour]
        raise ValueError(f"{len(bad)} hours do not have {records_per_hour} records, first {bad.index[0]}")
    if expected_hours is not None and len(hourly) != expected_hours:
        raise ValueError(f"expected {expected_hours} hours, got {len(hourly)}")
    return hourly


def drop_leap_day(hourly: pd.Series) -> pd.Series:
    """Remove 29 February from an hourly series with a DatetimeIndex (8784 -> 8760 hours)."""
    idx = hourly.index
    return hourly[~((idx.month == 2) & (idx.day == 29))]


def capacity_factor(power_kw, rated_kw: float, clip: bool = True) -> np.ndarray:
    """Production factor f^p = power / rated power. Values above 1 usually mean a wrong rated power
    or wrong units, so they raise unless ``clip=True`` and they exceed 1 by less than 2 %."""
    if not np.isfinite(rated_kw) or rated_kw <= 0:
        raise ValueError("rated_kw must be positive")
    cf = np.asarray(power_kw, dtype=float) / rated_kw
    if not np.all(np.isfinite(cf)):
        raise ValueError("power has non-finite values")
    if cf.min() < -1e-9:
        raise ValueError("power has negative values")
    if cf.max() > 1.02 or (not clip and cf.max() > 1.0):
        raise ValueError(f"capacity factor reaches {cf.max():.3f}; check rated_kw and units")
    return np.clip(cf, 0.0, 1.0)


def availability_from_downtime(downtime_hours_per_year: float) -> float:
    """Availability share 1 - downtime / 8760 (e.g. from an operation-and-maintenance model)."""
    if not np.isfinite(downtime_hours_per_year) or not 0 <= downtime_hours_per_year <= 8760:
        raise ValueError("downtime must be between 0 and 8760 hours per year")
    return 1.0 - downtime_hours_per_year / 8760.0


def apply_availability(cf, availability: float) -> np.ndarray:
    """Scale a production factor by an availability share (expected-value treatment of downtime)."""
    if not np.isfinite(availability) or not 0 <= availability <= 1:
        raise ValueError("availability must be in [0, 1]")
    return np.asarray(cf, dtype=float) * availability


def price_per_kwh(price_per_mwh) -> np.ndarray:
    """$/MWh (as published by most markets) to $/kWh."""
    return np.asarray(price_per_mwh, dtype=float) / 1000.0
