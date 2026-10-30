import numpy as np
import pytest

from reopt_h2_framework import (CONFIGURATIONS, ModelOptions, build_model, configuration_cases,
                                hydrogen_price_cases, run_cases, solve, summarize)
from reopt_h2_framework.synthetic import synthetic_instance

from conftest import gurobi_available

TOL = 1e-6


def _solved(inst, **opts):
    pm = build_model(inst, ModelOptions(**opts))
    res = solve(pm)
    return pm, res, summarize(pm)


@pytest.mark.parametrize("name", list(CONFIGURATIONS)[:4])
def test_wind_configurations_balance(name):
    inst = synthetic_instance(CONFIGURATIONS[name], hours=48)
    pm, res, s = _solved(inst)
    assert res.termination == "optimal"
    assert max(s.residuals.values()) < TOL
    # Eq. (21): delivered energy equals the target in every hour
    np.testing.assert_allclose(s.hourly["delivered_to_grid_kw"], inst.s("delivery_target"), atol=1e-6)
    # cost terms add up to the objective
    assert s.costs["check: capital + om - revenue - incentives + purchases"] == pytest.approx(res.objective, rel=1e-9, abs=1e-3)


def test_binary_and_fixed_on_give_the_same_optimum(full_instance):
    _, a, _ = _solved(full_instance, on_off="binary")
    _, b, _ = _solved(full_instance, on_off="fixed_on")
    assert a.objective == pytest.approx(b.objective, rel=1e-6)


def test_storage_and_hydrogen_bounds(full_instance):
    pm, _, s = _solved(full_instance)
    p = full_instance.params
    se = np.asarray(pm.v["se"].solution)[1:]
    bkwh = float(pm.v["bkWh"].solution)
    assert se.max() <= bkwh + TOL
    assert se.min() >= p.get("battery", "min_soc") * bkwh - TOL                       # Eq. (7)
    flows = s.hourly["production_to_battery_kw"] + s.hourly["battery_discharge_kw"]
    assert flows.max() <= float(pm.v["bkW"].solution) + TOL                            # Eq. (8)
    rp_e = np.asarray(pm.v["rp_e"].solution)
    assert rp_e.max() <= float(pm.v["X_kg"].solution) + TOL                            # Eq. (14)
    sh = np.asarray(pm.v["sh"].solution)[1:]
    assert sh.max() <= float(pm.v["X_tank"].solution) + TOL                            # Eq. (16)


def test_no_tank_means_no_hydrogen_storage():
    inst = synthetic_instance(CONFIGURATIONS["wind+battery+electrolyzer"], hours=48)
    _, _, s = _solved(inst)
    np.testing.assert_allclose(s.hourly["hydrogen_produced_kg"], s.hourly["hydrogen_sold_kg"], atol=1e-6)


def test_fuel_cell_case_uses_stored_hydrogen():
    inst = synthetic_instance(CONFIGURATIONS["wind+electrolyzer+tank+fuel_cell"], hours=48)
    pm, _, s = _solved(inst)
    assert s.sizes["fuel_cell (kW)"] > 0 and s.sizes["hydrogen tank (kg)"] > 0
    mf = inst.params.get("fuel_cell", "kg_per_kwh")
    np.testing.assert_allclose(s.hourly["hydrogen_to_fuel_cell_kg"], mf * s.hourly["fuel_cell_generation_kw"], atol=1e-6)


def test_site_load_with_outage():
    inst = synthetic_instance(CONFIGURATIONS["wind+battery+electrolyzer+tank+fuel_cell"], hours=48)
    pm, _, s = _solved(inst, delivery="site_load")
    out = inst.s("outage") == 1
    assert out.any()
    assert np.abs(s.hourly["grid_to_load_kw"].to_numpy()[out]).max() < TOL
    assert np.abs(s.hourly["grid_to_electrolyzer_kw"].to_numpy()[out]).max() < TOL
    assert np.abs(s.hourly["delivered_to_grid_kw"].to_numpy()[out]).max() < TOL
    assert s.residuals["site_load_balance"] < TOL
    assert s.costs["purchases"] > 0


def test_interconnection_limit():
    inst = synthetic_instance(CONFIGURATIONS["wind+battery"], hours=48, name="lim")
    limit = np.full(48, 30_000.0)
    pm, _, s = _solved(inst.with_series(interconnection_limit=limit), delivery="site_load")
    assert s.hourly["delivered_to_grid_kw"].max() <= 30_000 + TOL


def test_constant_rate_electrolysis():
    inst = synthetic_instance(CONFIGURATIONS["wind+battery+electrolyzer"], hours=48)
    pm, _, s = _solved(inst, electrolyzer_operation="constant")
    rp_e = np.asarray(pm.v["rp_e"].solution)
    np.testing.assert_allclose(rp_e, float(pm.v["X_kg"].solution), atol=1e-6)


def test_csp_heat_supply():
    inst = synthetic_instance(CONFIGURATIONS["csp+tes+electrolyzer"], hours=48)
    inst = inst.with_series(hydrogen_price=np.full(48, 6.0))
    pm, _, s = _solved(inst, electrolyzer_supply="csp_heat")
    p = inst.params
    assert s.sizes["electrolyzer (kg/h)"] > 0
    per_kg = p.get("electrolyzer", "heat_kwh_per_kg") + p.get("csp", "heat_per_kwh_e") * p.get("electrolyzer", "kwh_per_kg")
    heat = s.hourly["csp_heat_to_electrolyzer_kwh_th"] + s.hourly["tes_discharge_kwh_th"]
    np.testing.assert_allclose(heat, per_kg * s.hourly["hydrogen_produced_kg"], atol=1e-5)
    s_tes = np.asarray(pm.v["s_tes"].solution)
    flow = (p.get("tes", "charge_efficiency") * np.asarray(pm.v["q_charge"].solution)
            - np.asarray(pm.v["q_discharge"].solution) / p.get("tes", "discharge_efficiency"))
    np.testing.assert_allclose(s_tes[1:] - s_tes[:-1], flow, atol=1e-5)
    assert "LCOH ($/kg)" in s.indicators


def test_production_incentive_eligibility(full_instance):
    p = full_instance.params
    eligible = full_instance.with_params(p.replace(wind__incentive_rate=0.02, wind__incentive_years=10,
                                                   wind__incentive_max_size=60_000))
    pm, _, _ = _solved(eligible, on_off="fixed_on")
    assert float(pm.v["Z_pi_wind"].solution) == pytest.approx(1.0)
    x_pi = float(pm.v["X_pi_wind"].solution)
    fin = eligible.factors
    rp = np.asarray(pm.v["rp"].solution.sel(t="wind"))
    bound = 0.02 * fin.f_pi(10) * eligible.hour_weight * float((eligible.s("wind_cf") * rp).sum())
    assert x_pi == pytest.approx(bound, rel=1e-6)                                      # Eq. (10)
    too_big = full_instance.with_params(p.replace(wind__incentive_rate=0.02, wind__incentive_years=10,
                                                  wind__incentive_max_size=40_000))
    pm2, _, _ = _solved(too_big, on_off="fixed_on")
    assert float(pm2.v["X_pi_wind"].solution) == pytest.approx(0.0, abs=1e-6)        # Eq. (11): 50 MW > 40 MW


def test_fixed_capital_cost_needs_build_indicator(full_instance):
    p = full_instance.params.replace(fuel_cell__fixed_capital_cost=5e6)
    pm, _, s = _solved(full_instance.with_params(p), on_off="fixed_on")
    z = float(pm.v["z_sigma"].solution.sel(t="fuel_cell"))
    assert (z > 0.5) == (s.sizes["fuel_cell (kW)"] > TOL)


def test_shortfall_extension():
    inst = synthetic_instance(CONFIGURATIONS["wind+battery"], hours=48)
    inst = inst.with_params(inst.params.replace(battery__max_kw=1000.0, battery__max_kwh=4000.0))
    with pytest.raises(Exception):
        solve(build_model(inst))                                                        # target cannot be met
    pm, _, s = _solved(inst, shortfall_penalty=1.0)
    assert s.hourly["shortfall_kw"].sum() > 0
    assert s.residuals["eq21_delivery"] < TOL


def test_lcoh_definition():
    inst = synthetic_instance(CONFIGURATIONS["wind+battery+electrolyzer"], hours=48)
    pm, _, s = _solved(inst, on_off="fixed_on")
    zero = summarize(pm, electricity_value="zero").indicators["LCOH ($/kg)"]
    p, fin, W = inst.params, inst.factors, inst.hour_weight
    from reopt_h2_framework import pwf
    kg = s.sizes["electrolyzer (kg/h)"]
    rp_e = np.asarray(pm.v["rp_e"].solution).sum()
    num = p.get("electrolyzer", "capital_cost") * kg + fin.f_om * (p.get("electrolyzer", "fixed_om") * kg
                                                                   + p.get("electrolyzer", "variable_om") * W * rp_e)
    den = pwf(fin.years, fin.discount_rate) * s.annual["hydrogen produced (kg/yr)"]
    assert zero == pytest.approx(num / den, rel=1e-9)
    assert s.indicators["LCOH ($/kg)"] > zero
    with pytest.raises(ValueError):
        summarize(pm, electricity_value="market")


def test_options_validation(full_instance):
    with pytest.raises(ValueError):
        ModelOptions(delivery="island")
    with pytest.raises(ValueError, match="applies to delivery=.target. only"):
        ModelOptions(delivery="site_load", shortfall_penalty=1.0)
    with pytest.raises(ValueError, match="needs the CSP plant"):
        build_model(full_instance, ModelOptions(electrolyzer_supply="csp_heat"))
    csp = synthetic_instance(CONFIGURATIONS["csp+tes+electrolyzer"], hours=48)
    with pytest.raises(ValueError, match="only supplies heat"):
        build_model(csp, ModelOptions())


def test_run_cases_and_hydrogen_price_sweep():
    base = synthetic_instance(hours=48)
    df = run_cases(base, configuration_cases(["wind+battery", "wind+battery+electrolyzer"]),
                   solver_options={"mip_rel_gap": 1e-6})
    assert list(df.index) == ["wind+battery", "wind+battery+electrolyzer"]
    sweep = run_cases(base, hydrogen_price_cases([2.0, 6.0], ModelOptions(on_off="fixed_on")))
    assert sweep.loc["h2_price_6", "electrolyzer (kg/h)"] >= sweep.loc["h2_price_2", "electrolyzer (kg/h)"] - TOL


@pytest.mark.skipif(not gurobi_available(), reason="gurobipy with a licence is not available")
def test_gurobi_matches_highs(full_instance):
    a = solve(build_model(full_instance, ModelOptions(on_off="fixed_on")))
    b = solve(build_model(full_instance, ModelOptions(on_off="fixed_on")), solver="gurobi")
    assert a.objective == pytest.approx(b.objective, rel=1e-5)


def _tiny_wind(cf=1.0, target=10.0, hours=24, **extra):
    from reopt_h2_framework import Configuration, Instance, Parameters
    values = {"financial.analysis_years": 1, "financial.discount_rate": 0.0, "wind.capital_cost": 0.0,
              "wind.fixed_om": 0.0, "wind.variable_om": 0.001, "wind.max_size": 1000.0,
              "battery.power_cost": 100.0, "battery.energy_cost": 100.0, "battery.max_kw": 1000.0,
              "battery.max_kwh": 1000.0, "battery.initial_soc": 0.0}
    values.update(extra)
    series = {"wind_cf": np.full(hours, cf), "delivery_target": np.full(hours, target), "export_price": np.zeros(hours)}
    return Instance.create(Configuration(battery=True, electrolyzer=False, tank=False, fuel_cell=False),
                           Parameters.from_mapping(values), series, {"wind": 100.0})


def test_unused_available_energy_counts_reduced_production():
    pm, _, s = _solved(_tiny_wind(), on_off="fixed_on")
    # 100 kW at a capacity factor of 1, 10 kW delivered: nine tenths of the available energy is not used
    assert s.indicators["unused share of available wind and PV energy (-)"] == pytest.approx(0.9, abs=1e-6)
    assert s.annual["explicit curtailment (kWh/yr)"] == pytest.approx(0.0, abs=1e-3)


def test_lcoe_includes_fixed_capital_cost():
    inst = _tiny_wind(target=1.0, **{"wind.fixed_capital_cost": 1000.0, "wind.variable_om": 0.0,
                                     "wind.max_size": 1.0})
    inst = _with_wind_size(inst, 1.0)
    pm, res, s = _solved(inst, on_off="fixed_on")
    assert res.objective == pytest.approx(1000.0, rel=1e-9)
    assert s.indicators["LCOE of wind generation ($/kWh)"] == pytest.approx(1000.0 / 8760.0, rel=1e-9)


def _with_wind_size(inst, kw):
    from reopt_h2_framework import Instance
    return Instance.create(inst.config, inst.params, inst.series, {"wind": kw})


def test_csp_with_site_outages_is_refused():
    inst = synthetic_instance(CONFIGURATIONS["csp+tes+electrolyzer"], hours=48)
    with pytest.raises(ValueError, match="not supported together with the CSP plant"):
        build_model(inst, ModelOptions(electrolyzer_supply="csp_heat", delivery="site_load"))


def test_csp_parasitics_are_reported_as_purchases():
    inst = synthetic_instance(CONFIGURATIONS["csp+tes+electrolyzer"], hours=48).with_series(hydrogen_price=np.full(48, 6.0))
    pm, _, s = _solved(inst, electrolyzer_supply="csp_heat")
    expected = inst.hour_weight * float((inst.s("csp_parasitic") * s.sizes["CSP (kW_th)"]).sum())
    assert s.annual["grid purchases (kWh/yr)"] == pytest.approx(expected, rel=1e-9)
    assert s.costs["purchases"] > 0


def test_case_adds_a_technology_with_its_inputs():
    from reopt_h2_framework import Case
    base = _tiny_wind(cf=0.6)
    case = Case("with electrolyzer", configuration=CONFIGURATIONS["wind+battery+electrolyzer"],
                options=ModelOptions(on_off="fixed_on"),
                parameter_values={"electrolyzer.capital_cost": 1000.0, "electrolyzer.fixed_om": 10.0,
                                  "electrolyzer.kwh_per_kg": 50.0, "electrolyzer.max_size": 10.0},
                series={"hydrogen_price": [5.0]})
    inst = case.apply(base)
    assert inst.config.electrolyzer and inst.fixed_sizes == {"wind": 100.0}
    pm, res, s = _solved(inst, on_off="fixed_on")
    assert res.termination == "optimal"


def test_curtailment_is_attributed_to_each_generator():
    inst = synthetic_instance(CONFIGURATIONS["wind+electrolyzer+tank+fuel_cell"], hours=48)
    pm, _, s = _solved(inst)
    wind_ptc = np.asarray(pm.v["ptc"].solution.sel(t="wind"))
    assert (wind_ptc <= s.hourly["wind_generation_kw"].to_numpy() + TOL).all()
    share = s.indicators["unused share of available wind and PV energy (-)"]
    assert 0.0 <= share <= 1.0


def test_csp_needs_a_positive_heat_demand():
    inst = synthetic_instance(CONFIGURATIONS["csp+tes+electrolyzer"], hours=48)
    p = inst.params.replace(electrolyzer__heat_kwh_per_kg=0.0, csp__heat_per_kwh_e=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        build_model(inst.with_params(p), ModelOptions(electrolyzer_supply="csp_heat"))
