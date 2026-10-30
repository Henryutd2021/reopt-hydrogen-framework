"""Compare system configurations and run sensitivity cases on synthetic data.

    python examples/configurations.py

1. the four wind configurations: wind+battery; +electrolyzer; electrolyzer+tank+fuel cell instead of
   the battery; all together;
2. hydrogen sale price;
3. technology cost factors (e.g. a future cost year);
4. a different wind resource profile;
5. wind availability after downtime (e.g. from an operation-and-maintenance model).
"""

from reopt_h2_framework import Case, ModelOptions, configuration_cases, hydrogen_price_cases, run_cases
from reopt_h2_framework.preprocess import apply_availability, availability_from_downtime
from reopt_h2_framework.synthetic import synthetic_instance, synthetic_series

base = synthetic_instance(hours=72)
fast = ModelOptions(on_off="fixed_on")    # same optimum as the binary form of Eq. (2), smaller model
cols = ["objective ($)", "battery energy (kWh)", "electrolyzer (kg/h)", "hydrogen tank (kg)", "fuel_cell (kW)",
        "LCOH ($/kg)", "unused share of available wind and PV energy (-)"]


def show(title, df):
    print(f"\n== {title}\n" + df.reindex(columns=[c for c in cols if c in df.columns]).round(3).to_string())


show("configurations", run_cases(base, configuration_cases()))
show("hydrogen sale price ($/kg)", run_cases(base, hydrogen_price_cases([2.0, 3.0, 4.0, 5.0], fast)))

future = {"electrolyzer.capital_cost": 0.6, "fuel_cell.capital_cost": 0.7, "battery.energy_cost": 0.7}
show("cost factors", run_cases(base, [Case("today", options=fast), Case("lower costs", options=fast,
                                                                      parameter_factors=future)]))

other_site = synthetic_series(72, seed=7)["wind_cf"] * 0.85
show("wind profile", run_cases(base, [Case("site A", options=fast),
                                      Case("site B", options=fast, series={"wind_cf": other_site})]))

cases = []
for label, downtime in (("downtime 300 h/yr", 300.0), ("downtime 900 h/yr", 900.0)):
    cf = apply_availability(base.s("wind_cf"), availability_from_downtime(downtime))
    cases.append(Case(label, options=fast, series={"wind_cf": cf}))
show("wind availability", run_cases(base, cases))
print("\nThe data are synthetic: the numbers illustrate the model, not any real site.")
