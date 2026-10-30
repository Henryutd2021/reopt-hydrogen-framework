"""Synthetic example data: an invented site with illustrative round-number costs.

The numbers describe no real site, market or technology vintage. They exist to show how the model
behaves and how to read its output. For real studies, build inputs as in ``docs/data_pipeline.md``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .inputs import Configuration, Instance
from .parameters import Parameters


def illustrative_parameters() -> Parameters:
    """Round illustrative values for every component (units: ``parameters.canonical_units()``)."""
    return Parameters.from_mapping({
        "financial.analysis_years": 25, "financial.discount_rate": 0.07, "financial.om_escalation": 0.02,
        "financial.energy_escalation": 0.01, "financial.tax_rate": 0.25,
        "wind.capital_cost": 1300, "wind.fixed_om": 40, "wind.max_size": 100_000,
        "pv.capital_cost": 1000, "pv.fixed_om": 20, "pv.max_size": 100_000,
        "fuel_cell.capital_cost": 1200, "fuel_cell.fixed_om": 35, "fuel_cell.max_size": 50_000,
        "fuel_cell.kg_per_kwh": 0.06,
        "battery.power_cost": 300, "battery.energy_cost": 250, "battery.om_per_kwh": 10,
        "battery.charge_efficiency": 0.95, "battery.discharge_efficiency": 0.95, "battery.initial_soc": 0.1,
        "battery.min_soc": 0.1, "battery.max_kw": 100_000, "battery.max_kwh": 400_000,
        "electrolyzer.capital_cost": 40_000, "electrolyzer.fixed_om": 2_000, "electrolyzer.variable_om": 0.2,
        "electrolyzer.kwh_per_kg": 52, "electrolyzer.heat_kwh_per_kg": 8, "electrolyzer.max_size": 5_000,
        "tank.capital_cost": 500, "tank.fixed_om": 10, "tank.max_size": 200_000, "tank.initial_fill": 0.0,
        "csp.capital_cost": 900, "csp.fixed_om": 25, "csp.max_size": 1_000_000, "csp.heat_per_kwh_e": 1.0,
        "tes.energy_cost": 25, "tes.fixed_om": 0.5, "tes.max_kwh": 10_000_000,
        "tes.charge_efficiency": 0.99, "tes.discharge_efficiency": 0.99, "tes.initial_soc": 0.0,
    })


def synthetic_series(hours: int = 72, seed: int = 0, start_hour: int = 0) -> Dict[str, np.ndarray]:
    """Hourly series of an invented site: wind and PV capacity factors, prices, a delivery target,
    a site load with one outage block, and CSP heat and parasitic profiles. The first hour is
    ``start_hour`` o'clock."""
    if hours < 24:
        raise ValueError("synthetic_series needs at least 24 hours")
    if not 0 <= start_hour < 24:
        raise ValueError("start_hour must be in 0..23")
    rng = np.random.default_rng(seed)
    h = np.arange(hours)
    day = ((h + start_hour) % 24) / 24.0
    # wind: windy at the start, then lulls and windy spells with a two-day period, a night-time maximum and noise
    wind = 0.45 + 0.35 * np.cos(2 * np.pi * h / 48) + 0.08 * np.cos(2 * np.pi * day) + 0.05 * rng.normal(size=hours)
    wind = np.clip(wind, 0.0, 0.95)
    sun = np.clip(np.sin(np.pi * (day * 24 - 6) / 12), 0.0, None)
    cloud = np.clip(1 - 0.4 * rng.random(hours // 24 + 2), 0.5, 1.0)[(h + start_hour) // 24]
    pv = np.round(0.85 * sun * cloud, 4)
    price = np.round(0.035 + 0.02 * np.cos(2 * np.pi * (day - 0.75)) + 0.006 * rng.normal(size=hours), 4)
    price = np.clip(price, 0.005, None)
    load = np.round(8_000 + 2_500 * np.cos(2 * np.pi * (day - 0.6)), 1)
    outage = np.zeros(hours)
    outage[30:36] = 1.0
    csp_heat = np.round(1.1 * sun * cloud, 4)
    parasitic = np.round(0.01 + 0.02 * (sun == 0), 4)
    return {
        "wind_cf": np.round(wind, 4), "pv_cf": pv, "export_price": price,
        "hydrogen_price": np.full(hours, 3.0), "delivery_target": np.full(hours, 12_000.0),
        "interconnection_limit": np.full(hours, 40_000.0), "site_load": load,
        "purchase_price": np.round(price + 0.01, 4), "outage": outage,
        "csp_thermal_cf": csp_heat, "csp_parasitic": parasitic,
    }


def synthetic_instance(config: Configuration = Configuration(), hours: int = 72, seed: int = 0,
                       wind_kw: float = 50_000, pv_kw: float = 100_000, csp_kw_th: float = 300_000,
                       start_hour: int = 0, name: str = "synthetic") -> Instance:
    """An instance of the synthetic site. The plant sizes of wind, PV and CSP are fixed (a site-specific
    study of a given plant); storage, electrolyzer, tank and fuel cell are sized by the model."""
    series = synthetic_series(hours, seed, start_hour)
    if not config.wind:                      # solar-hydrogen configurations deliver no energy to the grid
        series["delivery_target"] = np.zeros(hours)
    fixed = {}
    for tech, size in (("wind", wind_kw), ("pv", pv_kw), ("csp", csp_kw_th)):
        if getattr(config, tech):
            fixed[tech] = size
    return Instance.create(config, illustrative_parameters(), series, fixed, name)
