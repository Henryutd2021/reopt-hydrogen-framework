"""From raw interval records to an input directory, with the generic preparation helpers.

The "raw" records here are invented (15-minute prices and 10-minute wind power of a 2 MW turbine).
With real data, replace them by the records described in docs/data_pipeline.md.

    python examples/prepare_inputs.py outputs/prepared
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from reopt_h2_framework import CONFIGURATIONS, Instance, ModelOptions, build_model, solve, summarize, write_inputs
from reopt_h2_framework.preprocess import capacity_factor, price_per_kwh, to_hourly
from reopt_h2_framework.synthetic import illustrative_parameters

out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/prepared")
rng = np.random.default_rng(1)
hours = 48
start = pd.Timestamp("2021-03-01")

prices = pd.DataFrame({"interval_start": pd.date_range(start, periods=hours * 4, freq="15min")})
prices["price_per_mwh"] = 30 + 15 * np.sin(np.arange(len(prices)) / 16) + rng.normal(0, 3, len(prices))
turbine = pd.DataFrame({"time": pd.date_range(start, periods=hours * 6, freq="10min")})
turbine["power_kw"] = np.clip(900 + 700 * np.sin(np.arange(len(turbine)) / 40) + rng.normal(0, 150, len(turbine)), 0, 2000)

price_h = price_per_kwh(to_hourly(prices, "interval_start", "price_per_mwh", expected_hours=hours))
wind_cf = capacity_factor(to_hourly(turbine, "time", "power_kw", expected_hours=hours), rated_kw=2000.0)

series = {"wind_cf": wind_cf, "export_price": np.clip(price_h, 0, None), "hydrogen_price": np.full(hours, 3.0),
          "delivery_target": np.full(hours, 10_000.0)}
inst = Instance.create(CONFIGURATIONS["wind+battery+electrolyzer"], illustrative_parameters(), series,
                       {"wind": 50_000.0}, name="prepared")
write_inputs(inst, out, ModelOptions(on_off="fixed_on"))
print(f"wrote {out} (parameters.csv, hourly.csv, case.json)")

pm = build_model(inst, ModelOptions(on_off="fixed_on"))
solve(pm)
print(summarize(pm).sizes.round(1).to_string())
