"""Minimal example: size a battery, electrolyzer, hydrogen tank and fuel cell next to a given wind farm
on synthetic data (three days), with HiGHS. Runs in a few seconds.

    python examples/minimal.py
"""

from reopt_h2_framework import CONFIGURATIONS, ModelOptions, build_model, solve, summarize
from reopt_h2_framework.synthetic import synthetic_instance

inst = synthetic_instance(CONFIGURATIONS["wind+battery+electrolyzer+tank+fuel_cell"], hours=72)
pm = build_model(inst, ModelOptions())            # delivery to the grid follows the target profile, Eq. (21)
res = solve(pm)                                   # HiGHS; solve(pm, solver="gurobi") also works
s = summarize(pm)

print(f"objective (present value of life-cycle cost): {res.objective:,.0f} $")
print("\nsizes\n" + s.sizes.round(1).to_string())
print("\nannual quantities\n" + s.annual.round(0).to_string())
print("\nindicators\n" + s.indicators.round(3).to_string())
print("\nlargest balance residual:", max(s.residuals.values()))
print("\nThe data are synthetic: the numbers illustrate the model, not any real site.")
