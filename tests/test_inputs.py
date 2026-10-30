import numpy as np
import pytest

from reopt_h2_framework import CONFIGURATIONS, Configuration, Instance
from reopt_h2_framework.synthetic import illustrative_parameters, synthetic_instance, synthetic_series


def test_configuration_rules():
    with pytest.raises(ValueError, match="fuel cell needs the electrolyzer"):
        Configuration(electrolyzer=False, fuel_cell=True)
    with pytest.raises(ValueError, match="CSP plant supplies the electrolyzer"):
        Configuration(electrolyzer=False, fuel_cell=False, tank=False, csp=True)
    with pytest.raises(ValueError, match="at least one of wind, pv or csp"):
        Configuration(wind=False)
    assert CONFIGURATIONS["wind+battery"].generators == ("wind",)
    assert CONFIGURATIONS["wind+electrolyzer+tank+fuel_cell"].generators == ("wind", "fuel_cell")


def test_hour_weight():
    inst = synthetic_instance(hours=48)
    assert inst.hours == 48
    assert inst.hour_weight == pytest.approx(8760 / 48)


@pytest.mark.parametrize("change, message", [
    ({"wind_cf": np.full(48, 1.2)}, r"wind_cf must be in \[0, 1\]"),
    ({"export_price": np.full(47, 0.03)}, "same length"),
    ({"site_load": np.full(48, -1.0)}, "non-negative"),
    ({"outage": np.full(48, 0.5)}, "outage must be 0 or 1"),
    ({"hydrogen_price": np.r_[np.full(47, 3.0), np.nan]}, "non-finite"),
    ({"bogus": np.ones(48)}, "unknown series"),
])
def test_series_validation(change, message):
    series = synthetic_series(48)
    series.update(change)
    with pytest.raises(ValueError, match=message):
        Instance.create(Configuration(), illustrative_parameters(), series)


def test_missing_series_and_parameters():
    series = synthetic_series(48)
    del series["wind_cf"]
    with pytest.raises(ValueError, match="wind_cf is missing"):
        Instance.create(Configuration(), illustrative_parameters(), series)
    p = illustrative_parameters()
    values = {k: v for k, v in p.values.items() if k != ("electrolyzer", "kwh_per_kg")}
    from reopt_h2_framework import Parameters
    with pytest.raises(ValueError, match="electrolyzer"):
        Instance.create(Configuration(), Parameters(values), synthetic_series(48))


def test_fixed_sizes_validation():
    with pytest.raises(ValueError, match="not a candidate"):
        Instance.create(CONFIGURATIONS["wind+battery"], illustrative_parameters(), synthetic_series(48), {"tank": 10.0})
    with pytest.raises(ValueError, match="finite and non-negative"):
        Instance.create(Configuration(), illustrative_parameters(), synthetic_series(48), {"wind": -1.0})
    inst = Instance.create(Configuration(), illustrative_parameters(), synthetic_series(48), {"battery_kw": 1000.0})
    assert inst.fixed_sizes["battery_kw"] == 1000.0
