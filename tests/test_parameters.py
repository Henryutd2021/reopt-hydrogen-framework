import math

import pytest

from reopt_h2_framework import PARAMETERS, Parameters, canonical_units, effective_capital_cost, pwf
from reopt_h2_framework.parameters import FinancialFactors
from reopt_h2_framework.synthetic import illustrative_parameters


def test_pwf_matches_explicit_sum():
    for n, d, e in ((25, 0.07, 0.02), (10, 0.05, 0.0), (20, 0.03, 0.03)):
        explicit = sum(((1 + e) / (1 + d)) ** y for y in range(1, n + 1))
        assert pwf(n, d, e) == pytest.approx(explicit, rel=1e-12)
    assert pwf(0, 0.05) == 0.0


def test_financial_factors():
    f = FinancialFactors.from_parameters(illustrative_parameters())
    assert f.f_om == pytest.approx(pwf(25, 0.07, 0.02))
    assert f.f_e == pytest.approx(pwf(25, 0.07, 0.01))
    assert f.f_tow == 0.25
    assert f.f_pi(10) == pytest.approx(pwf(10, 0.07, 0.0))
    assert f.f_pi(40) == pytest.approx(pwf(25, 0.07, 0.0))      # limited to the analysis period


def test_every_parameter_has_a_unit():
    units = canonical_units()
    assert len(units) == len(PARAMETERS)
    assert units["electrolyzer.kwh_per_kg"] == "kWh/kg"
    assert units["fuel_cell.kg_per_kwh"] == "kg/kWh"


@pytest.mark.parametrize("values, message", [
    ({"wind.nope": 1.0}, "unknown parameter"),
    ({"wind.capital_cost": float("nan")}, "finite"),
    ({"wind.capital_cost": -1.0}, "below its lower bound"),
    ({"battery.charge_efficiency": 1.2}, r"\(0, 1\]"),
    ({"battery.initial_soc": 1.5}, r"\[0, 1\]"),
    ({"nodot": 1.0}, "component"),
    ({"financial.tax_rate": 25.0}, r"tax_rate must be in \[0, 1\)"),
    ({"electrolyzer.kwh_per_kg": 0.0}, "must be positive"),
    ({"fuel_cell.kg_per_kwh": 0.0}, "must be positive"),
])
def test_invalid_parameters(values, message):
    with pytest.raises(ValueError, match=message):
        Parameters.from_mapping(values)


def test_required_and_defaults():
    p = Parameters.from_mapping({"wind.capital_cost": 1000.0})
    assert p.get("wind", "variable_om") == 0.0
    assert p.get("wind", "levelization") == 1.0
    with pytest.raises(ValueError, match="missing required parameter wind.max_size"):
        p.get("wind", "max_size")
    with pytest.raises(ValueError, match="missing required parameters for wind"):
        p.require("wind")


def test_replace_and_scaled():
    p = illustrative_parameters()
    assert p.replace(electrolyzer__capital_cost=1.0).get("electrolyzer", "capital_cost") == 1.0
    assert p.scaled({"wind.capital_cost": 0.5}).get("wind", "capital_cost") == pytest.approx(650.0)


def test_effective_capital_cost():
    assert effective_capital_cost(1000.0, 0.25, 0.07) == pytest.approx(1000.0)
    assert effective_capital_cost(1000.0, 0.25, 0.07, itc=0.3) == pytest.approx(700.0)
    macrs = (0.2, 0.32, 0.192, 0.1152, 0.1152, 0.0576)
    basis = 1000.0 * (1 - 0.15)
    shield = 0.25 * sum(basis * s / 1.07 ** (y + 1) for y, s in enumerate(macrs))
    assert effective_capital_cost(1000.0, 0.25, 0.07, itc=0.3, macrs=macrs) == pytest.approx(700.0 - shield)
    with pytest.raises(ValueError, match="sum to 1"):
        effective_capital_cost(1000.0, 0.25, 0.07, macrs=(0.5,))
    with pytest.raises(ValueError, match="itc must be"):
        effective_capital_cost(1000.0, 0.25, 0.07, itc=1.5)
    with pytest.raises(ValueError, match="finite and non-negative"):
        effective_capital_cost(1000.0, 0.25, 0.07, macrs=(float("nan"), 1.0))
    with pytest.raises(ValueError, match="would be negative"):
        effective_capital_cost(100.0, 0.25, 0.0, itc=1.0, macrs=(1.0,))
    assert math.isfinite(effective_capital_cost(0.0, 0.0, 0.0))
