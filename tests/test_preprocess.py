import numpy as np
import pandas as pd
import pytest

from reopt_h2_framework.preprocess import (apply_availability, availability_from_downtime, capacity_factor,
                                           drop_leap_day, price_per_kwh, to_hourly)


def test_to_hourly_mean_and_sum():
    t = pd.date_range("2022-01-01", periods=24, freq="15min")
    df = pd.DataFrame({"time": t, "price": np.arange(24.0)})
    mean = to_hourly(df, "time", "price", expected_hours=6)
    np.testing.assert_allclose(mean.to_numpy(), [1.5, 5.5, 9.5, 13.5, 17.5, 21.5])
    total = to_hourly(df, "time", "price", how="sum")
    assert total.iloc[0] == 6.0


def test_to_hourly_refuses_gaps_and_duplicates():
    t = pd.date_range("2022-01-01", periods=24, freq="15min")
    df = pd.DataFrame({"time": t, "v": 1.0}).drop(index=range(4, 8))
    with pytest.raises(ValueError, match="have no record"):
        to_hourly(df, "time", "v")
    dup = pd.DataFrame({"time": [t[0], t[0]], "v": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicated"):
        to_hourly(dup, "time", "v")


def test_to_hourly_refuses_missing_time_stamps_and_incomplete_hours():
    t = pd.Series(pd.date_range("2022-01-01", periods=8, freq="15min"))
    df = pd.DataFrame({"time": t, "e": 100.0})
    df.loc[3, "time"] = pd.NaT
    with pytest.raises(ValueError, match="missing time stamps"):
        to_hourly(df, "time", "e", how="sum")
    partial = pd.DataFrame({"time": pd.date_range("2022-01-01", periods=7, freq="15min"), "e": 1.0})
    with pytest.raises(ValueError, match="do not have 4 records"):
        to_hourly(partial, "time", "e", how="sum", records_per_hour=4)
    assert len(to_hourly(partial, "time", "e", how="sum")) == 2


def test_drop_leap_day():
    s = pd.Series(1.0, index=pd.date_range("2020-01-01", periods=8784, freq="h"))
    assert len(drop_leap_day(s)) == 8760


def test_capacity_factor():
    np.testing.assert_allclose(capacity_factor([0, 50, 100, 101], 100.0), [0, 0.5, 1.0, 1.0])
    with pytest.raises(ValueError, match="check rated_kw"):
        capacity_factor([0, 150], 100.0)
    with pytest.raises(ValueError, match="negative"):
        capacity_factor([-5, 10], 100.0)


def test_availability():
    assert availability_from_downtime(876) == pytest.approx(0.9)
    np.testing.assert_allclose(apply_availability([0.5, 1.0], 0.9), [0.45, 0.9])
    with pytest.raises(ValueError):
        apply_availability([0.5], 1.1)
    np.testing.assert_allclose(price_per_kwh([30.0]), [0.03])
