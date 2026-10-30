"""Model instance: which technologies are candidates, the hourly series, and the parameters.

Hourly series (one value per hour, ``docs/data_pipeline.md`` gives the schemas):

    wind_cf               f^p_th of wind, kWh per kW of rated power (0..1)
    pv_cf                 f^p_th of PV, kWh per kW of rated power (0..1)
    export_price          c^e_h, $/kWh paid for energy delivered to the grid
    hydrogen_price        c^hts_h, $/kg paid for hydrogen sold
    delivery_target       δ^d_h, kW: energy to be delivered to the grid in each hour, Eq. (21)
    interconnection_limit δ̄^g_h, kW: limit on delivery to the grid, Eq. (23)
    site_load             L_h, kW: site electric load (``delivery="site_load"``)
    purchase_price        c^p_h, $/kWh paid for grid electricity (site load, electrolyzer, CSP parasitics)
    outage                1 in hours without grid connection, else 0 (``delivery="site_load"``)
    csp_thermal_cf        kWh_th per kW_th of CSP size delivered by the solar field and receiver
    csp_parasitic         kWh of electricity per kW_th of CSP size needed by the plant (parasitic load)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .parameters import FinancialFactors, Parameters

SERIES = ("wind_cf", "pv_cf", "export_price", "hydrogen_price", "delivery_target", "interconnection_limit",
          "site_load", "purchase_price", "outage", "csp_thermal_cf", "csp_parasitic")
SERIES_UNITS = {"wind_cf": "-", "pv_cf": "-", "export_price": "$/kWh", "hydrogen_price": "$/kg",
                "delivery_target": "kW", "interconnection_limit": "kW", "site_load": "kW",
                "purchase_price": "$/kWh", "outage": "0/1", "csp_thermal_cf": "kWh_th/kW_th",
                "csp_parasitic": "kWh/kW_th"}


@dataclass(frozen=True)
class Configuration:
    """Candidate technologies. A technology that is False is not in the model (its size is zero)."""

    wind: bool = True
    pv: bool = False
    battery: bool = True
    electrolyzer: bool = True
    tank: bool = True
    fuel_cell: bool = True
    csp: bool = False          # CSP plant with thermal storage (TES) supplying the electrolyzer with heat

    def __post_init__(self) -> None:
        if self.fuel_cell and not self.electrolyzer:
            raise ValueError("a fuel cell needs the electrolyzer (its only hydrogen source)")
        if self.csp and not self.electrolyzer:
            raise ValueError("the CSP plant supplies the electrolyzer; enable the electrolyzer")
        if not (self.wind or self.pv or self.csp):
            raise ValueError("at least one of wind, pv or csp is needed")

    @property
    def generators(self) -> Tuple[str, ...]:
        return tuple(g for g in ("wind", "pv", "fuel_cell") if getattr(self, g))

    @property
    def components(self) -> Tuple[str, ...]:
        names = ["financial"] + list(self.generators)
        names += [c for c in ("battery", "electrolyzer", "tank") if getattr(self, c)]
        if self.csp:
            names += ["csp", "tes"]
        return tuple(names)


def sizable(config: Configuration) -> Tuple[str, ...]:
    """Names accepted in ``fixed_sizes``: generators, battery_kw, battery_kwh, electrolyzer, tank, csp, tes."""
    names = list(config.generators)
    if config.battery:
        names += ["battery_kw", "battery_kwh"]
    names += [c for c in ("electrolyzer", "tank") if getattr(config, c)]
    if config.csp:
        names += ["csp", "tes"]
    return tuple(names)


@dataclass(frozen=True)
class Instance:
    """Everything the model needs. Build it with :meth:`Instance.create`, which validates the inputs."""

    config: Configuration
    params: Parameters
    series: Mapping[str, np.ndarray]
    fixed_sizes: Mapping[str, float] = field(default_factory=dict)   # e.g. {"wind": 50_000} (kW)
    name: str = "case"

    @classmethod
    def create(cls, config: Configuration, params: Parameters, series: Mapping[str, object],
               fixed_sizes: Optional[Mapping[str, float]] = None, name: str = "case") -> "Instance":
        arrays: Dict[str, np.ndarray] = {}
        lengths = set()
        for key, values in series.items():
            if key not in SERIES:
                raise ValueError(f"unknown series {key!r}; known: {SERIES}")
            a = np.asarray(values, dtype=float).ravel()
            if a.size == 0:
                raise ValueError(f"series {key} is empty")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"series {key} has non-finite values")
            arrays[key] = a
            lengths.add(a.size)
        if len(lengths) != 1:
            raise ValueError(f"all series must have the same length, got {sorted(lengths)}")
        for key in ("wind_cf", "pv_cf"):
            if key in arrays and (arrays[key].min() < 0 or arrays[key].max() > 1):
                raise ValueError(f"{key} must be in [0, 1]")
        for key in ("delivery_target", "interconnection_limit", "site_load", "csp_thermal_cf", "csp_parasitic"):
            if key in arrays and arrays[key].min() < 0:
                raise ValueError(f"{key} must be non-negative")
        if "outage" in arrays and not np.isin(arrays["outage"], (0.0, 1.0)).all():
            raise ValueError("outage must be 0 or 1")
        for g in ("wind", "pv"):
            if getattr(config, g) and f"{g}_cf" not in arrays:
                raise ValueError(f"{g} is a candidate but series {g}_cf is missing")
        if config.csp and "csp_thermal_cf" not in arrays:
            raise ValueError("csp is a candidate but series csp_thermal_cf is missing")
        if "hydrogen_price" not in arrays and config.electrolyzer:
            raise ValueError("series hydrogen_price is required with the electrolyzer")
        for comp in config.components:
            params.require(comp)
        sizes = dict(fixed_sizes or {})
        for tech, size in sizes.items():
            if tech not in sizable(config):
                raise ValueError(f"fixed size for {tech!r}, which is not a candidate; candidates: {sizable(config)}")
            if not np.isfinite(size) or size < 0:
                raise ValueError(f"fixed size of {tech} must be finite and non-negative")
        return cls(config=config, params=params, series=arrays, fixed_sizes=sizes, name=name)

    @property
    def hours(self) -> int:
        return int(next(iter(self.series.values())).size)

    @property
    def hour_weight(self) -> float:
        """W = 8760 / |H|: hourly sums are scaled to one year when the horizon is shorter (1 for a full year)."""
        return 8760.0 / self.hours

    @property
    def factors(self) -> FinancialFactors:
        return FinancialFactors.from_parameters(self.params)

    def s(self, key: str, default: Optional[float] = None) -> np.ndarray:
        """Series ``key``; a constant ``default`` if it is absent (error if no default)."""
        if key in self.series:
            return self.series[key]
        if default is None:
            raise ValueError(f"series {key} is required for this model")
        return np.full(self.hours, float(default))

    def with_series(self, **updates: object) -> "Instance":
        merged = {**self.series, **updates}
        return Instance.create(self.config, self.params, merged, self.fixed_sizes, self.name)

    def with_params(self, params: Parameters) -> "Instance":
        return Instance.create(self.config, params, self.series, self.fixed_sizes, self.name)

    def with_config(self, config: Configuration) -> "Instance":
        sizes = {k: v for k, v in self.fixed_sizes.items() if k in sizable(config)}
        return Instance.create(config, self.params, self.series, sizes, self.name)

    def renamed(self, name: str) -> "Instance":
        return replace(self, name=name)
