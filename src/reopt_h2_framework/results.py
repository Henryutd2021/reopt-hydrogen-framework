"""Result tables of a solved model: sizes, annual quantities, cost terms, indicators, hourly dispatch,
and balance residuals recomputed from the solution.

Annual quantities are hourly sums scaled by W = 8760/|H|. Costs are present values in $, as in Eq. (1).
Indicators are computed after the solve and are not part of the optimization; their definitions are
in ``docs/formulation.md`` (indicators).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from .model import PlanningModel
from .parameters import pwf

ElectricityValue = Union[str, float]


def _val(pm: PlanningModel, name: str) -> np.ndarray:
    return np.asarray(pm.v[name].solution) + 0.0 if name in pm.v else np.array(0.0)   # + 0.0 turns -0.0 into 0.0


def _hourly(pm: PlanningModel, name: str, t=None) -> np.ndarray:
    n = pm.instance.hours
    if name not in pm.v:
        return np.zeros(n)
    sol = pm.v[name].solution
    if t is not None:
        return np.asarray(sol.sel(t=t)) if "t" in sol.dims and t in sol.indexes["t"] else np.zeros(n)
    if "t" in sol.dims:
        sol = sol.sum("t")
    if "h" in sol.dims and sol.sizes["h"] == n + 1:
        sol = sol.isel(h=slice(1, None))
    return np.asarray(sol)


@dataclass
class Summary:
    sizes: pd.Series
    annual: pd.Series
    costs: pd.Series
    indicators: pd.Series
    hourly: pd.DataFrame
    residuals: Dict[str, float]

    def to_csv(self, directory) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.sizes.rename("value").to_csv(d / "sizes.csv", index_label="item")
        self.annual.rename("value").to_csv(d / "annual.csv", index_label="item")
        self.costs.rename("value").to_csv(d / "costs.csv", index_label="item")
        self.indicators.rename("value").to_csv(d / "indicators.csv", index_label="item")
        self.hourly.to_csv(d / "hourly.csv", index_label="hour")
        (d / "balance_residuals.json").write_text(json.dumps(self.residuals, indent=2) + "\n")
        return d


def summarize(pm: PlanningModel, electricity_value: ElectricityValue = "export_price") -> Summary:
    """Tables of a solved model. ``electricity_value`` prices the electricity used by the electrolyzer
    in the LCOH indicator: "export_price" (c^e_h, the default), "zero", or a number in $/kWh."""
    inst, p, cfg = pm.instance, pm.instance.params, pm.instance.config
    W, n, fin = inst.hour_weight, inst.hours, inst.factors
    if pm.model.status != "ok":
        raise ValueError("the model has no solution; solve it first")

    sizes = {}
    for g in cfg.generators:
        sizes[f"{g} (kW)"] = float(pm.v["size"].solution.sel(t=g)) + 0.0
    for name, label in (("bkW", "battery power (kW)"), ("bkWh", "battery energy (kWh)"),
                        ("X_kg", "electrolyzer (kg/h)"), ("X_tank", "hydrogen tank (kg)"),
                        ("X_csp", "CSP (kW_th)"), ("X_tes", "thermal storage (kWh_th)")):
        if name in pm.v:
            sizes[label] = float(_val(pm, name))

    # hourly dispatch
    hourly = pd.DataFrame(index=pd.RangeIndex(1, n + 1, name="hour"))
    gen_total = np.zeros(n)
    for g in cfg.generators:
        f = inst.s(f"{g}_cf") if g != "fuel_cell" else np.ones(n)
        out = f * p.get(g, "levelization") * _hourly(pm, "rp", g)
        hourly[f"{g}_generation_kw"] = out
        gen_total += out
    for name, col in (("ptg", "production_to_grid_kw"), ("pts", "production_to_battery_kw"),
                      ("pte", "production_to_electrolyzer_kw"), ("ptc", "curtailed_kw"), ("ptl", "production_to_load_kw"),
                      ("dfs", "battery_discharge_kw"), ("stg", "battery_to_grid_kw"), ("ste", "battery_to_electrolyzer_kw"),
                      ("stl", "battery_to_load_kw"), ("se", "battery_energy_kwh"), ("grid_to_load", "grid_to_load_kw"),
                      ("grid_to_electrolyzer", "grid_to_electrolyzer_kw"), ("p_e", "hydrogen_produced_kg"),
                      ("sell", "hydrogen_sold_kg"), ("etf", "hydrogen_to_fuel_cell_kg"), ("sh", "tank_level_kg"),
                      ("q_direct", "csp_heat_to_electrolyzer_kwh_th"), ("q_discharge", "tes_discharge_kwh_th"),
                      ("q_dump", "csp_heat_unused_kwh_th"), ("s_tes", "tes_energy_kwh_th"), ("shortfall", "shortfall_kw")):
        if name in pm.v:
            hourly[col] = _hourly(pm, name)
    delivered = hourly.get("production_to_grid_kw", 0) + hourly.get("battery_to_grid_kw", 0)
    hourly["delivered_to_grid_kw"] = delivered
    # available wind and PV energy (size × production factor × levelization) and the part not used
    available = np.zeros(n)
    for g in ("wind", "pv"):
        if g in cfg.generators:
            available += inst.s(f"{g}_cf") * p.get(g, "levelization") * sizes[f"{g} (kW)"]
    hourly["available_wind_pv_kw"] = available
    used = sum(hourly[f"{g}_generation_kw"] - _hourly(pm, "ptc", g) for g in ("wind", "pv") if g in cfg.generators)
    hourly["unused_wind_pv_kw"] = available - (used if not isinstance(used, int) else 0)
    if cfg.csp:
        hourly["grid_to_csp_parasitic_kw"] = inst.s("csp_parasitic", 0.0) * sizes["CSP (kW_th)"]

    def ann(col):                               # annual total of an hourly column
        return float(W * hourly[col].sum()) if col in hourly else 0.0

    annual = {
        "generation (kWh/yr)": float(W * gen_total.sum()),
        "available wind and PV energy (kWh/yr)": ann("available_wind_pv_kw"),
        "unused wind and PV energy (kWh/yr)": ann("unused_wind_pv_kw"),
        "explicit curtailment (kWh/yr)": ann("curtailed_kw"),
        "delivered to grid (kWh/yr)": ann("delivered_to_grid_kw"),
        "battery discharge (kWh/yr)": ann("battery_discharge_kw"),
        "electricity to electrolyzer (kWh/yr)": ann("production_to_electrolyzer_kw") + ann("battery_to_electrolyzer_kw")
        + ann("grid_to_electrolyzer_kw"),
        "grid purchases (kWh/yr)": ann("grid_to_load_kw") + ann("grid_to_electrolyzer_kw") + ann("grid_to_csp_parasitic_kw"),
        "hydrogen produced (kg/yr)": ann("hydrogen_produced_kg"),
        "hydrogen sold (kg/yr)": ann("hydrogen_sold_kg"),
        "hydrogen to fuel cell (kg/yr)": ann("hydrogen_to_fuel_cell_kg"),
        "CSP heat to electrolyzer (kWh_th/yr)": ann("csp_heat_to_electrolyzer_kwh_th") + ann("tes_discharge_kwh_th"),
    }
    if "fuel_cell_generation_kw" in hourly:
        annual["fuel cell generation (kWh/yr)"] = ann("fuel_cell_generation_kw")

    costs = {k: float(expr.solution.sum()) if hasattr(expr, "solution") else float(expr)
             for k, expr in pm.cost_terms.items()}
    for k in ("capital", "om", "revenue", "incentives", "purchases"):
        costs.setdefault(k, 0.0)
    costs["objective"] = float(pm.model.objective.value)
    costs["check: capital + om - revenue - incentives + purchases"] = (
        costs["capital"] + costs["om"] - costs["revenue"] - costs["incentives"] + costs["purchases"])

    # indicators
    f_h = pwf(fin.years, fin.discount_rate, 0.0)
    ind = {}
    if annual["available wind and PV energy (kWh/yr)"] > 0:
        ind["unused share of available wind and PV energy (-)"] = (
            annual["unused wind and PV energy (kWh/yr)"] / annual["available wind and PV energy (kWh/yr)"])
    kg = sizes.get("electrolyzer (kg/h)", 0.0)
    h2 = annual["hydrogen produced (kg/yr)"]
    if kg > 0:
        ind["electrolyzer capacity factor (-)"] = h2 / (8760.0 * kg)
    if h2 > 0:
        ind["LCOH ($/kg)"] = _lcoh(pm, hourly, sizes, f_h, electricity_value)
    for g in ("wind", "pv"):
        col = f"{g}_generation_kw"
        if col in hourly and hourly[col].sum() > 0:
            size = sizes[f"{g} (kW)"]
            built = float(pm.v["z_sigma"].solution.sel(t=g)) if "z_sigma" in pm.v and g in pm.v["z_sigma"].indexes["t"] else 0.0
            cost = (p.get(g, "capital_cost") * size + p.get(g, "fixed_capital_cost") * built
                    + fin.f_om * (p.get(g, "fixed_om") * size + W * p.get(g, "variable_om") * _hourly(pm, "rp", g).sum()))
            ind[f"LCOE of {g} generation ($/kWh)"] = cost / (f_h * W * hourly[col].sum())

    return Summary(sizes=pd.Series(sizes, dtype=float), annual=pd.Series(annual, dtype=float),
                   costs=pd.Series(costs, dtype=float), indicators=pd.Series(ind, dtype=float),
                   hourly=hourly, residuals=balance_residuals(pm))


def _lcoh(pm, hourly, sizes, f_h, electricity_value) -> float:
    inst, p, fin, W = pm.instance, pm.instance.params, pm.instance.factors, pm.instance.hour_weight
    kg, tank = sizes.get("electrolyzer (kg/h)", 0.0), sizes.get("hydrogen tank (kg)", 0.0)
    cap = p.get("electrolyzer", "capital_cost") * kg + (p.get("tank", "capital_cost") * tank if tank else 0.0)
    om = p.get("electrolyzer", "fixed_om") * kg + p.get("electrolyzer", "variable_om") * W * _hourly(pm, "rp_e").sum()
    if tank:
        om += p.get("tank", "fixed_om") * tank
    energy = 0.0
    onsite = (hourly.get("production_to_electrolyzer_kw", 0) + hourly.get("battery_to_electrolyzer_kw", 0))
    if isinstance(electricity_value, str):
        if electricity_value == "export_price":
            price = inst.s("export_price", 0.0)
        elif electricity_value == "zero":
            price = np.zeros(inst.hours)
        else:
            raise ValueError("electricity_value must be 'export_price', 'zero' or a number ($/kWh)")
    else:
        if not np.isfinite(electricity_value) or electricity_value < 0:
            raise ValueError("electricity_value must be finite and non-negative")
        price = np.full(inst.hours, float(electricity_value))
    energy += W * float((price * np.asarray(onsite)).sum())
    if "grid_to_electrolyzer_kw" in hourly:
        energy += W * float((inst.s("purchase_price") * hourly["grid_to_electrolyzer_kw"]).sum())
    if "CSP (kW_th)" in sizes:
        csp, tes = sizes["CSP (kW_th)"], sizes["thermal storage (kWh_th)"]
        cap += p.get("csp", "capital_cost") * csp + p.get("tes", "energy_cost") * tes
        om += p.get("csp", "fixed_om") * csp + p.get("tes", "fixed_om") * tes
        energy += W * float((inst.s("purchase_price") * inst.s("csp_parasitic", 0.0)).sum()) * csp
    h2 = W * hourly["hydrogen_produced_kg"].sum()
    return (cap + fin.f_om * om + fin.f_e * energy) / (f_h * h2)


def balance_residuals(pm: PlanningModel) -> Dict[str, float]:
    """Largest absolute residual of the main balances, recomputed from the solution values."""
    inst, p, cfg = pm.instance, pm.instance.params, pm.instance.config
    n = inst.hours
    res: Dict[str, float] = {}
    gen = np.zeros(n)
    for g in cfg.generators:
        f = inst.s(f"{g}_cf") if g != "fuel_cell" else np.ones(n)
        gen += f * p.get(g, "levelization") * _hourly(pm, "rp", g)
    alloc = sum(_hourly(pm, k) for k in ("pts", "ptg", "pte", "ptc", "ptl"))
    if cfg.generators:
        res["eq22_generation_allocation"] = float(np.abs(gen - alloc).max())
    if cfg.battery:
        se = np.asarray(pm.v["se"].solution)
        flow = p.get("battery", "charge_efficiency") * _hourly(pm, "pts") - _hourly(pm, "dfs") / p.get("battery", "discharge_efficiency")
        res["eq06_soc_balance"] = float(np.abs(se[1:] - se[:-1] - flow).max())
        out = sum(_hourly(pm, k) for k in ("stg", "ste", "stl"))
        res["eq20_energy_balance"] = float(np.abs(gen + _hourly(pm, "dfs") - alloc - out).max())
    if pm.options.delivery == "target" and (cfg.generators or cfg.battery):
        delivered = _hourly(pm, "ptg") + _hourly(pm, "stg") + _hourly(pm, "shortfall")
        res["eq21_delivery"] = float(np.abs(delivered - inst.s("delivery_target")).max())
    if pm.options.delivery == "site_load":
        served = _hourly(pm, "ptl") + _hourly(pm, "stl") + _hourly(pm, "grid_to_load")
        res["site_load_balance"] = float(np.abs(served - inst.s("site_load")).max())
    if cfg.electrolyzer:
        p_e = _hourly(pm, "p_e")
        out = _hourly(pm, "sell") + _hourly(pm, "etf")
        if cfg.tank:
            sh = np.asarray(pm.v["sh"].solution)
            res["eq15_hydrogen_storage"] = float(np.abs(sh[1:] - sh[:-1] - p_e + out).max())
        else:
            res["eq15_hydrogen_storage"] = float(np.abs(p_e - out).max())
        if pm.options.electrolyzer_supply == "electric":
            supply = sum(_hourly(pm, k) for k in ("pte", "ste", "grid_to_electrolyzer"))
            res["eq13_electrolyzer_energy"] = float(np.abs(p.get("electrolyzer", "kwh_per_kg") * p_e - supply).max())
        if cfg.fuel_cell:
            fc = p.get("fuel_cell", "levelization") * _hourly(pm, "rp", "fuel_cell")
            res["eq17_fuel_cell_hydrogen"] = float(np.abs(_hourly(pm, "etf") - p.get("fuel_cell", "kg_per_kwh") * fc).max())
    return res
