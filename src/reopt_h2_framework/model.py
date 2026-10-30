"""The hydrogen-extended sizing and dispatch model (mixed-integer linear program), written with linopy.

Decisions: sizes of wind, PV, fuel cell, battery (power and energy), electrolyzer, hydrogen tank and,
optionally, a CSP plant with thermal storage; and, for every hour, production, its allocation
(to storage, grid, electrolyzer, curtailment, site load), storage operation, hydrogen production,
storage, sales and use in the fuel cell.

Constraint names carry the equation numbers of the Journal of Mechanical Design article
(``eq02_max_power_rating`` ... ``eq23_interconnection``); ``docs/formulation.md`` gives the mathematics,
the conventions of this implementation and the options (site load with grid purchases and outages,
constant-rate electrolysis, heat supply from a CSP plant, and a shortfall extension).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from .inputs import Instance

DELIVERY_MODES = ("target", "site_load")
ELECTROLYZER_OPERATION = ("variable", "constant")
ELECTROLYZER_SUPPLY = ("electric", "csp_heat")
ON_OFF = ("binary", "fixed_on")


@dataclass(frozen=True)
class ModelOptions:
    """Model variants (``docs/formulation.md``).

    delivery              "target": energy delivered to the grid equals δ^d_h in every hour, Eq. (21);
                          "site_load": a site load is served by on-site supply and grid purchases,
                          exports are optional, and grid exchange is zero in outage hours.
    electrolyzer_operation "variable" (hourly production up to capacity) or "constant" (at capacity in every hour).
    electrolyzer_supply   "electric" (Eq. 13) or "csp_heat" (heat from the CSP plant and its thermal storage).
    on_off                "binary": Z^to_th of Eq. (2) is binary, as stated; "fixed_on": Z^to_th = 1.
                          Z^to_th appears only in Eq. (2), so both give the same optimum; "fixed_on" is smaller.
    shortfall_penalty     extension ($/kWh), "target" only: a non-negative shortfall in Eq. (21) at this price.
    """

    delivery: str = "target"
    electrolyzer_operation: str = "variable"
    electrolyzer_supply: str = "electric"
    on_off: str = "binary"
    shortfall_penalty: Optional[float] = None

    def __post_init__(self) -> None:
        for name, allowed in (("delivery", DELIVERY_MODES), ("electrolyzer_operation", ELECTROLYZER_OPERATION),
                              ("electrolyzer_supply", ELECTROLYZER_SUPPLY), ("on_off", ON_OFF)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        p = self.shortfall_penalty
        if p is not None:
            if not np.isfinite(p) or p < 0:
                raise ValueError("shortfall_penalty must be finite and non-negative")
            if self.delivery != "target":
                raise ValueError("shortfall_penalty applies to delivery='target' only")


@dataclass
class PlanningModel:
    """A built linopy model, the instance and options it was built from, and its variables by name."""

    model: linopy.Model
    instance: Instance
    options: ModelOptions
    v: Dict[str, linopy.Variable] = field(default_factory=dict)
    cost_terms: Dict[str, linopy.LinearExpression] = field(default_factory=dict)


def _size_var(m, inst, name, comp, lower_field, upper_field, label):
    """Scalar size variable with its bounds, or fixed to ``inst.fixed_sizes[name]``."""
    p = inst.params
    if name in inst.fixed_sizes:
        lo = hi = float(inst.fixed_sizes[name])
    else:
        lo = p.get(comp, lower_field) if lower_field else 0.0
        hi = p.get(comp, upper_field)
        if lo > hi:
            raise ValueError(f"{comp}: lower bound {lo} exceeds upper bound {hi}")
    return m.add_variables(lower=lo, upper=hi, name=label), hi


def build_model(inst: Instance, options: Optional[ModelOptions] = None) -> PlanningModel:
    """Build the linopy model of one instance."""
    opt = options or ModelOptions()
    cfg, p, fin = inst.config, inst.params, inst.factors
    W = inst.hour_weight
    n = inst.hours
    site = opt.delivery == "site_load"
    heat = opt.electrolyzer_supply == "csp_heat"
    if heat and not cfg.csp:
        raise ValueError("electrolyzer_supply='csp_heat' needs the CSP plant in the configuration")
    if cfg.csp and not heat:
        raise ValueError("the CSP plant only supplies heat to the electrolyzer; use electrolyzer_supply='csp_heat'")
    if cfg.csp and site and inst.s("outage", 0.0).max() > 0 and inst.s("csp_parasitic", 0.0).max() > 0:
        raise ValueError("CSP parasitic load is bought from the grid; outage hours with delivery='site_load' "
                         "are not supported together with the CSP plant")
    if opt.electrolyzer_operation == "constant" and not cfg.electrolyzer:
        raise ValueError("electrolyzer_operation='constant' needs the electrolyzer")
    electric_ely = cfg.electrolyzer and not heat

    H = pd.RangeIndex(1, n + 1, name="h")
    H0 = pd.RangeIndex(0, n + 1, name="h")       # storage levels include the initial period h = 0

    def hs(a):                                  # hourly data array
        return xr.DataArray(np.asarray(a, dtype=float), coords=[H])

    gens = cfg.generators
    G = pd.Index(gens, name="t")

    m = linopy.Model()
    v: Dict[str, linopy.Variable] = {}
    tax = 1.0 - fin.f_tow
    capital, fixed_om, variable_om, revenue, purchases, incentives = [], [], [], [], [], []

    # ---- generators t in T: wind, PV, fuel cell (rated production X^rp_th, kW)
    if gens:
        lo, hi, cm, omf, omp, f1 = [], [], [], [], [], []
        for g in gens:
            if g in inst.fixed_sizes:
                lo.append(inst.fixed_sizes[g]); hi.append(inst.fixed_sizes[g])
            else:
                lo.append(p.get(g, "min_size")); hi.append(p.get(g, "max_size"))
                if lo[-1] > hi[-1]:
                    raise ValueError(f"{g}: min_size exceeds max_size")
            cm.append(p.get(g, "capital_cost")); omf.append(p.get(g, "fixed_om"))
            omp.append(p.get(g, "variable_om")); f1.append(p.get(g, "levelization"))
        bmax = xr.DataArray(hi, coords=[G])
        size = m.add_variables(lower=xr.DataArray(lo, coords=[G]), upper=bmax, name="size")
        fp = xr.DataArray(np.vstack([inst.s(f"{g}_cf") if g != "fuel_cell" else np.ones(n) for g in gens]),
                          coords=[G, H])
        lev = xr.DataArray(f1, coords=[G])
        rp = m.add_variables(lower=0, coords=[G, H], name="rp")
        gen = fp * lev * rp                                    # actual generation f^p_th f^1_t X^rp_th
        v.update(size=size, rp=rp)
        capital.append((xr.DataArray(cm, coords=[G]) * size).sum())
        fixed_om.append((xr.DataArray(omf, coords=[G]) * size).sum())
        variable_om.append(W * (xr.DataArray(omp, coords=[G]) * rp).sum())
        # Eq. (2): rated production limited to the maximum power rating when on
        if opt.on_off == "binary":
            on = m.add_variables(binary=True, coords=[G, H], name="on")
            v["on"] = on
            m.add_constraints(rp - bmax * on <= 0, name="eq02_max_power_rating")
        else:
            m.add_constraints(rp <= bmax, name="eq02_max_power_rating")
        # Convention: rated production does not exceed the size (Eq. 19 states it for the fuel cell)
        m.add_constraints(rp - size <= 0, name="rated_production_le_size")
        # Fixed (y-intercept) capital cost with a build indicator Z^σ_t
        cb = {g: p.get(g, "fixed_capital_cost") for g in gens}
        built = [g for g in gens if cb[g] > 0]
        if built:
            B1 = pd.Index(built, name="t")
            z_sig = m.add_variables(binary=True, coords=[B1], name="z_sigma")
            v["z_sigma"] = z_sig
            m.add_constraints(size.sel(t=B1) - bmax.sel(t=B1) * z_sig <= 0, name="build_indicator")
            capital.append((xr.DataArray([cb[g] for g in built], coords=[B1]) * z_sig).sum())
        # Allocation of production
        ptg = m.add_variables(lower=0, coords=[G, H], name="ptg")      # X^ptg_th, to the grid
        ptc = m.add_variables(lower=0, coords=[G, H], name="ptc")      # X^ptc_th, curtailed
        alloc = ptg + ptc
        v.update(ptg=ptg, ptc=ptc)
        if cfg.battery:
            pts = m.add_variables(lower=0, coords=[G, H], name="pts")  # X^pts_bth, to the battery
            alloc = alloc + pts
            v["pts"] = pts
        if electric_ely:
            pte = m.add_variables(lower=0, coords=[G, H], name="pte")  # X^pte_th, to the electrolyzer
            alloc = alloc + pte
            v["pte"] = pte
        if site:
            ptl = m.add_variables(lower=0, coords=[G, H], name="ptl")  # to the site load (site_load mode)
            alloc = alloc + ptl
            v["ptl"] = ptl
        # Eq. (22): generation allocated to storage, grid, electrolyzer and curtailment; each generator's
        # production is allocated to its own flows (convention), which summed over t gives Eq. (22)
        m.add_constraints(gen.sum("t") - alloc.sum("t") == 0, name="eq22_generation_allocation")
        m.add_constraints(gen - alloc == 0, name="allocation_by_generator")
        revenue.append(W * (hs(inst.s("export_price", 0.0)) * ptg.sum("t")).sum())

    # ---- battery b in B
    if cfg.battery:
        bkw, _ = _size_var(m, inst, "battery_kw", "battery", "min_kw", "max_kw", "bkW")
        bkwh, _ = _size_var(m, inst, "battery_kwh", "battery", "min_kwh", "max_kwh", "bkWh")
        se = m.add_variables(lower=0, coords=[H0], name="se")          # X^se_bh, kWh
        dfs = m.add_variables(lower=0, coords=[H], name="dfs")         # X^dfs_bh, kW
        stg = m.add_variables(lower=0, coords=[H], name="stg")         # X^stg_bh, to the grid
        v.update(bkW=bkw, bkWh=bkwh, se=se, dfs=dfs, stg=stg)
        eta_p, eta_m = p.get("battery", "charge_efficiency"), p.get("battery", "discharge_efficiency")
        charge = v["pts"].sum("t") if gens else 0
        m.add_constraints(se.sel(h=0) - p.get("battery", "initial_soc") * bkwh == 0, name="eq03_initial_soc")
        m.add_constraints(se.sel(h=H) - se.shift(h=1).sel(h=H) - eta_p * charge + (1 / eta_m) * dfs == 0,
                          name="eq06_soc_balance")
        m.add_constraints(se.sel(h=H) - p.get("battery", "min_soc") * bkwh >= 0, name="eq07_min_soc")
        m.add_constraints(bkw - charge - dfs >= 0, name="eq08_storage_power")
        m.add_constraints(bkwh - se.sel(h=H) >= 0, name="eq09_storage_energy")
        capital.append(p.get("battery", "power_cost") * bkw
                       + (p.get("battery", "energy_cost") + p.get("battery", "om_per_kwh")) * bkwh)
        out = stg
        if electric_ely:
            ste = m.add_variables(lower=0, coords=[H], name="ste")     # X^ste_bh, to the electrolyzer
            out = out + ste
            v["ste"] = ste
        if site:
            stl = m.add_variables(lower=0, coords=[H], name="stl")     # to the site load
            out = out + stl
            v["stl"] = stl
        # Eq. (20), with Eq. (22): discharge is split between grid, electrolyzer (and site load)
        lhs = (gen.sum("t") if gens else 0) + dfs
        rhs = (alloc.sum("t") if gens else 0) + out
        m.add_constraints(lhs - rhs == 0, name="eq20_energy_balance")
        revenue.append(W * (hs(inst.s("export_price", 0.0)) * stg).sum())

    # ---- delivery to the grid: Eqs. (21) and (23), or the site load
    exports = None
    if gens:
        exports = v["ptg"].sum("t")
    if cfg.battery:
        exports = v["stg"] if exports is None else exports + v["stg"]
    outage = inst.s("outage", 0.0) if site else np.zeros(n)
    limit = inst.s("interconnection_limit", np.inf).copy()
    limit[outage == 1] = 0.0
    if exports is not None:
        if opt.delivery == "target":
            target = hs(inst.s("delivery_target"))
            lhs = exports
            if opt.shortfall_penalty is not None:
                short = m.add_variables(lower=0, coords=[H], name="shortfall")
                v["shortfall"] = short
                lhs = lhs + short
                purchases.append(W * opt.shortfall_penalty * short.sum())
            m.add_constraints(lhs == target, name="eq21_delivery")
        finite = np.isfinite(limit)
        if finite.any():
            m.add_constraints(exports <= hs(np.where(finite, limit, 0.0)), name="eq23_interconnection",
                              mask=hs(finite.astype(float)) > 0)
    elif opt.delivery == "target" and inst.s("delivery_target", 0.0).max() > 0:
        raise ValueError("delivery_target is positive but no generator or battery can deliver energy")

    if site:
        price = hs(inst.s("purchase_price"))
        grid_upper = hs(np.where(outage == 1, 0.0, np.inf))
        gl = m.add_variables(lower=0, upper=grid_upper, coords=[H], name="grid_to_load")
        v["grid_to_load"] = gl
        served = gl
        if gens:
            served = served + v["ptl"].sum("t")
        if cfg.battery:
            served = served + v["stl"]
        m.add_constraints(served == hs(inst.s("site_load")), name="site_load_balance")
        purchases.append(W * (price * gl).sum())

    # ---- electrolyzer e and hydrogen
    if cfg.electrolyzer:
        kg, kg_max = _size_var(m, inst, "electrolyzer", "electrolyzer", "min_size", "max_size", "X_kg")
        rp_e = m.add_variables(lower=0, coords=[H], name="rp_e")       # X^rp_eh, kg/h (rated)
        p_e = m.add_variables(lower=0, coords=[H], name="p_e")         # X^p_eh, kg/h (actual)
        sell = m.add_variables(lower=0, coords=[H], name="sell")       # X^sell_eh, kg
        v.update(X_kg=kg, rp_e=rp_e, p_e=p_e, sell=sell)
        if opt.on_off == "binary":
            on_e = m.add_variables(binary=True, coords=[H], name="on_e")
            v["on_e"] = on_e
            m.add_constraints(rp_e - kg_max * on_e <= 0, name="eq02_max_power_rating_electrolyzer")
        else:
            m.add_constraints(rp_e <= kg_max, name="eq02_max_power_rating_electrolyzer")
        lev_e = p.get("electrolyzer", "levelization")
        m.add_constraints(p_e - lev_e * rp_e == 0, name="eq12_electrolyzer_output")       # f^p_eh = 1
        me = p.get("electrolyzer", "kwh_per_kg")
        if electric_ely:
            supply = 0
            if gens:
                supply = v["pte"].sum("t")
            if cfg.battery:
                supply = supply + v["ste"]
            if site:
                gte = m.add_variables(lower=0, upper=hs(np.where(outage == 1, 0.0, np.inf)), coords=[H],
                                      name="grid_to_electrolyzer")
                v["grid_to_electrolyzer"] = gte
                supply = supply + gte
                purchases.append(W * (hs(inst.s("purchase_price")) * gte).sum())
            m.add_constraints(me * p_e - supply == 0, name="eq13_electrolyzer_energy")
        m.add_constraints(rp_e - kg <= 0, name="eq14_electrolyzer_capacity")
        if opt.electrolyzer_operation == "constant":
            m.add_constraints(rp_e - kg == 0, name="constant_rate_electrolysis")
        capital.append(p.get("electrolyzer", "capital_cost") * kg)
        fixed_om.append(p.get("electrolyzer", "fixed_om") * kg)
        variable_om.append(W * p.get("electrolyzer", "variable_om") * rp_e.sum())
        revenue.append(W * (hs(inst.s("hydrogen_price")) * sell).sum())

        h2_out = sell
        if cfg.fuel_cell:
            etf = m.add_variables(lower=0, coords=[H], name="etf")     # X^etf_eh, kg to the fuel cell
            v["etf"] = etf
            mf = p.get("fuel_cell", "kg_per_kwh")
            m.add_constraints(etf - mf * gen.sel(t="fuel_cell") == 0, name="eq17_fuel_cell_hydrogen")
            h2_out = h2_out + etf
        if cfg.tank:
            xt, _ = _size_var(m, inst, "tank", "tank", None, "max_size", "X_tank")
            sh = m.add_variables(lower=0, coords=[H0], name="sh")      # X^sh_th, kg in the tank
            v.update(X_tank=xt, sh=sh)
            m.add_constraints(sh.sel(h=0) - p.get("tank", "initial_fill") * xt == 0, name="tank_initial_level")
            m.add_constraints(sh.sel(h=H) - sh.shift(h=1).sel(h=H) - p_e + h2_out == 0,
                              name="eq15_hydrogen_storage")
            m.add_constraints(sh.sel(h=H) - xt <= 0, name="eq16_tank_capacity")
            capital.append(p.get("tank", "capital_cost") * xt)
            fixed_om.append(p.get("tank", "fixed_om") * xt)
        else:
            m.add_constraints(p_e - h2_out == 0, name="eq15_hydrogen_storage")   # no tank: X^sh = 0

        # production incentive of the electrolyzer, Eqs. (10)-(11), per kg
        _incentive(m, v, inst, "electrolyzer", kg, kg_max, rp_e, 1.0, W, fin, incentives)

    # production incentives of generators, Eqs. (10)-(11)
    for g in gens:
        _incentive(m, v, inst, g, v["size"].sel(t=g), float(bmax.sel(t=g)), v["rp"].sel(t=g),
                   fp.sel(t=g).values, W, fin, incentives)

    # ---- CSP plant with thermal storage supplying the electrolyzer (heat)
    if cfg.csp:
        csp, _ = _size_var(m, inst, "csp", "csp", None, "max_size", "X_csp")
        xtes, _ = _size_var(m, inst, "tes", "tes", None, "max_kwh", "X_tes")
        q_dir = m.add_variables(lower=0, coords=[H], name="q_direct")
        q_ch = m.add_variables(lower=0, coords=[H], name="q_charge")
        q_dis = m.add_variables(lower=0, coords=[H], name="q_discharge")
        q_dump = m.add_variables(lower=0, coords=[H], name="q_dump")
        s_tes = m.add_variables(lower=0, coords=[H0], name="s_tes")
        v.update(X_csp=csp, X_tes=xtes, q_direct=q_dir, q_charge=q_ch, q_discharge=q_dis, q_dump=q_dump, s_tes=s_tes)
        m.add_constraints(hs(inst.s("csp_thermal_cf")) * csp - q_dir - q_ch - q_dump == 0, name="csp_heat_balance")
        m.add_constraints(s_tes.sel(h=0) - p.get("tes", "initial_soc") * xtes == 0, name="tes_initial")
        m.add_constraints(s_tes.sel(h=H) - s_tes.shift(h=1).sel(h=H) - p.get("tes", "charge_efficiency") * q_ch
                          + (1 / p.get("tes", "discharge_efficiency")) * q_dis == 0, name="tes_balance")
        m.add_constraints(s_tes.sel(h=H) - xtes <= 0, name="tes_capacity")
        heat_per_kg = p.get("electrolyzer", "heat_kwh_per_kg") + p.get("csp", "heat_per_kwh_e") * me
        if heat_per_kg <= 0:
            raise ValueError("with electrolyzer_supply='csp_heat', electrolyzer.heat_kwh_per_kg + "
                             "csp.heat_per_kwh_e * electrolyzer.kwh_per_kg must be positive")
        m.add_constraints(heat_per_kg * v["p_e"] - q_dir - q_dis == 0, name="electrolyzer_heat_supply")
        capital.append(p.get("csp", "capital_cost") * csp + p.get("tes", "energy_cost") * xtes)
        fixed_om.append(p.get("csp", "fixed_om") * csp + p.get("tes", "fixed_om") * xtes)
        parasitic = (hs(inst.s("purchase_price")) * hs(inst.s("csp_parasitic", 0.0))).sum().item()
        if parasitic:
            purchases.append(W * parasitic * csp)

    # ---- objective, Eq. (1) (+ purchases and the shortfall extension)
    def total(parts):
        out = 0
        for x in parts:
            out = out + x
        return out

    terms = {
        "capital": total(capital),
        "om": tax * fin.f_om * (total(fixed_om) + total(variable_om)),
        "revenue": tax * fin.f_e * total(revenue),
        "incentives": tax * total(incentives),
        "purchases": tax * fin.f_e * total(purchases),
    }
    m.add_objective(terms["capital"] + terms["om"] - terms["revenue"] - terms["incentives"] + terms["purchases"],
                    sense="min")
    pm = PlanningModel(model=m, instance=inst, options=opt, v=v)
    pm.cost_terms = {k: x for k, x in terms.items() if not isinstance(x, (int, float))}
    return pm


def _incentive(m, v, inst, tech, size, size_upper, rp, fp, W, fin, out) -> None:
    """Eqs. (10)-(11): production incentive X^pi_t (present value, $) and its eligibility Z^pi_t."""
    p = inst.params
    rate = p.get(tech, "incentive_rate")
    if rate <= 0:
        return
    coef = rate * fin.f_pi(p.get(tech, "incentive_years")) * p.get(tech, "incentive_levelization")
    fp = np.broadcast_to(np.asarray(fp, dtype=float), (inst.hours,))
    big_m = coef * W * float(fp.sum()) * size_upper        # largest incentive the size bound allows
    fpx = xr.DataArray(fp, coords=[rp.indexes["h"]])
    x_pi = m.add_variables(lower=0, name=f"X_pi_{tech}")
    z_pi = m.add_variables(binary=True, name=f"Z_pi_{tech}")
    v[f"X_pi_{tech}"], v[f"Z_pi_{tech}"] = x_pi, z_pi
    m.add_constraints(x_pi - big_m * z_pi <= 0, name=f"eq10_incentive_cap_{tech}")
    m.add_constraints(x_pi - coef * W * (fpx * rp).sum() <= 0, name=f"eq10_incentive_production_{tech}")
    m.add_constraints(size + size_upper * z_pi <= p.get(tech, "incentive_max_size") + size_upper,
                      name=f"eq11_incentive_size_{tech}")
    out.append(x_pi)
