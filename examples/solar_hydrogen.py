"""Solar hydrogen on synthetic data:

* a CSP plant with thermal storage supplying a high-temperature electrolyzer with heat, at variable
  or constant production rate;
* PV supplying a low-temperature electrolyzer at variable rate.

    python examples/solar_hydrogen.py
"""

import numpy as np

from reopt_h2_framework import CONFIGURATIONS, ModelOptions, build_model, solve, summarize
from reopt_h2_framework.synthetic import synthetic_instance

HOURS, PRICE = 72, 5.0                         # synthetic horizon and a constant hydrogen price, $/kg

for label, config, options in (
    ("CSP + TES, variable rate", "csp+tes+electrolyzer", ModelOptions(electrolyzer_supply="csp_heat")),
    ("CSP + TES, constant rate", "csp+tes+electrolyzer",
     ModelOptions(electrolyzer_supply="csp_heat", electrolyzer_operation="constant")),
    ("PV, variable rate", "pv+electrolyzer", ModelOptions()),
):
    # The horizon starts at 8:00, so that the solar field delivers heat before the first night;
    # the thermal storage starts empty.
    inst = synthetic_instance(CONFIGURATIONS[config], hours=HOURS, start_hour=8)
    inst = inst.with_series(hydrogen_price=np.full(HOURS, PRICE))
    pm = build_model(inst, options)
    solve(pm)
    s = summarize(pm)
    print(f"\n== {label}\n" + s.sizes.round(1).to_string())
    print(f"hydrogen produced (kg/yr)          {s.annual['hydrogen produced (kg/yr)']:,.0f}")
    print(f"electrolyzer capacity factor (-)   {s.indicators['electrolyzer capacity factor (-)']:.3f}")
print("\nThe data are synthetic: the numbers illustrate the model, not any real site.")
