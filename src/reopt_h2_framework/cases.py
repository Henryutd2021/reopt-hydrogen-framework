"""System configurations and sensitivity cases (``docs/scenarios.md``).

The configurations are those compared in the studies built on this framework: wind with battery,
electrolyzer, hydrogen tank and fuel cell in four combinations; CSP with thermal storage supplying a
high-temperature electrolyzer at constant or variable rate; PV supplying a low-temperature electrolyzer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .inputs import Configuration, Instance, sizable
from .model import ModelOptions, build_model
from .results import summarize
from .solve import solve

CONFIGURATIONS: Dict[str, Configuration] = {
    "wind+battery": Configuration(battery=True, electrolyzer=False, tank=False, fuel_cell=False),
    "wind+battery+electrolyzer": Configuration(battery=True, electrolyzer=True, tank=False, fuel_cell=False),
    "wind+electrolyzer+tank+fuel_cell": Configuration(battery=False, electrolyzer=True, tank=True, fuel_cell=True),
    "wind+battery+electrolyzer+tank+fuel_cell": Configuration(battery=True, electrolyzer=True, tank=True, fuel_cell=True),
    "csp+tes+electrolyzer": Configuration(wind=False, battery=False, electrolyzer=True, tank=False, fuel_cell=False, csp=True),
    "pv+electrolyzer": Configuration(wind=False, pv=True, battery=False, electrolyzer=True, tank=False, fuel_cell=False),
}
DEFAULT_OPTIONS: Dict[str, ModelOptions] = {
    "csp+tes+electrolyzer": ModelOptions(electrolyzer_supply="csp_heat"),
}
# The solar-hydrogen configurations produce hydrogen only: no energy is delivered to the grid.
DEFAULT_SERIES: Dict[str, Dict[str, list]] = {
    "csp+tes+electrolyzer": {"delivery_target": [0.0]},
    "pv+electrolyzer": {"delivery_target": [0.0]},
}


@dataclass(frozen=True)
class Case:
    """One model run: a configuration and options applied to a base instance, with optional changes."""

    name: str
    configuration: Optional[Configuration] = None
    options: ModelOptions = ModelOptions()
    parameter_factors: Mapping[str, float] = field(default_factory=dict)    # {"component.field": factor}
    parameter_values: Mapping[str, float] = field(default_factory=dict)     # {"component.field": value}
    series: Mapping[str, object] = field(default_factory=dict)              # replaced hourly series (a one-element
                                                                             # sequence is repeated for every hour)

    def apply(self, base: Instance) -> Instance:
        """The case's instance. Configuration, parameters and series are merged first and validated once,
        so a case may add a technology together with its parameters and series."""
        config = self.configuration if self.configuration is not None else base.config
        params = base.params
        if self.parameter_values:
            params = params.replace(**{k.replace(".", "__"): v for k, v in self.parameter_values.items()})
        if self.parameter_factors:
            params = params.scaled(self.parameter_factors)
        series = dict(base.series)
        for k, v in self.series.items():
            a = np.asarray(v, dtype=float).ravel()
            series[k] = np.full(base.hours, a[0]) if a.size == 1 else a
        sizes = {k: v for k, v in base.fixed_sizes.items() if k in sizable(config)}
        return Instance.create(config, params, series, sizes, self.name)


def configuration_cases(names: Iterable[str] = tuple(CONFIGURATIONS)[:4]) -> list:
    return [Case(name=n, configuration=CONFIGURATIONS[n], options=DEFAULT_OPTIONS.get(n, ModelOptions()),
                 series=DEFAULT_SERIES.get(n, {})) for n in names]


def hydrogen_price_cases(prices: Iterable[float], options: ModelOptions = ModelOptions()) -> list:
    """Constant hydrogen sale prices ($/kg)."""
    return [Case(name=f"h2_price_{p:g}", options=options, series={"hydrogen_price": [p]}) for p in prices]


def run_cases(base: Instance, cases: Iterable[Case], solver: str = "highs",
              solver_options: Optional[Mapping[str, object]] = None,
              on_result: Optional[Callable] = None) -> pd.DataFrame:
    """Solve every case on ``base`` and return one row per case: objective, sizes and indicators."""
    rows = []
    for case in cases:
        inst = case.apply(base)
        pm = build_model(inst, case.options)
        res = solve(pm, solver=solver, options=solver_options)
        s = summarize(pm)
        row = {"case": case.name, "objective ($)": res.objective, **s.sizes.to_dict(), **s.indicators.to_dict(),
               "hydrogen produced (kg/yr)": s.annual["hydrogen produced (kg/yr)"],
               "unused wind and PV energy (kWh/yr)": s.annual["unused wind and PV energy (kWh/yr)"]}
        rows.append(row)
        if on_result is not None:
            on_result(case, pm, s)
    return pd.DataFrame(rows).set_index("case")
