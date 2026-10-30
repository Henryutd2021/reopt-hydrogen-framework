# Data pipeline

This page describes how the inputs of the model are built, the rules applied while building them, and the format of an input directory. The repository contains no real data. The synthetic example (`reopt_h2_framework.synthetic`) generates invented data in the same format.

## 1. Sources

| Input | Typical public source | What is taken | Model input |
|---|---|---|---|
| Wind resource | NREL WIND Toolkit, through SAM or REopt's wind module, for the site and turbine of the study; or measured turbine or farm power | hourly power of the reference turbine or farm | `wind_cf` |
| Solar resource for PV | NREL NSRDB with PVWatts or SAM | hourly AC output per kW_dc | `pv_cf` |
| Solar resource for CSP | NREL NSRDB with SAM's molten-salt power-tower model through PySAM (site from the study, default plant configuration, heliostat field and tower optimized by SAM for the site) | hourly thermal output delivered by the receiver, and the plant's parasitic electricity | `csp_thermal_cf`, `csp_parasitic` |
| Electricity prices | wholesale market prices of the relevant ISO/RTO (e.g. hourly or 15-minute settlement point prices), or the output of a market simulation | price of each hour | `export_price`, `purchase_price` |
| Site load | DOE Commercial Reference Building profiles (as provided by REopt) scaled to an annual consumption, or metered load | hourly load | `site_load` |
| Hydrogen price | a scenario value | \$/kg | `hydrogen_price` |
| Delivery target | the stabilized output the system must deliver to the grid, chosen for the study | kW per hour | `delivery_target` |
| Interconnection limit | the interconnection capacity of the site | kW | `interconnection_limit` |
| Technology costs and performance | NREL Annual Technology Baseline (ATB) for generation and storage; hydrogen technology cost references for electrolyzer, tank and fuel cell | capital, O&M, efficiency, conversion rates | `parameters.csv` |
| Financial assumptions | the study's analysis period, discount rate, escalation, tax rate and incentives | – | `parameters.csv` |

**Terms of use.** Every provider has its own terms. NREL data products are generally public. Market price data are published by each ISO/RTO under its own terms of use. Check them before redistributing derived inputs. This repository ships none.

## 2. Preparation steps

1. **Time base.** Choose one calendar year and one time zone convention (local standard time is usual for market data) and align every series to it. Hour 1 is the first hour of 1 January.
2. **Interval data to hours.** Average prices and power over the intervals of each hour, and sum energies (`preprocess.to_hourly`).
   - Missing time stamps and hours without records are errors, not silent fills.
   - With `records_per_hour` (e.g. 4 for 15-minute data), incomplete hours are errors too.
   - Decide gap filling explicitly and record it.
3. **Leap years.** Drop 29 February (`preprocess.drop_leap_day`) so that every series has 8,760 hours.
4. **Units.** Prices published in \$/MWh are divided by 1,000 (`preprocess.price_per_kwh`). Hydrogen quantities are in kg and electrolyzer capacity in kg/h; heat is in kWh_th.
5. **Production factors.** Divide hourly power by the rated power of the same plant or turbine (`preprocess.capacity_factor`). A maximum above 1 points to a wrong rated power or wrong units. The helper refuses values more than 2 % above 1 and clips the rest.
6. **Availability.** To represent expected downtime from an operation-and-maintenance study, scale the production factor by the availability share (`preprocess.availability_from_downtime`, `preprocess.apply_availability`). Enter the corresponding O&M cost as `wind.fixed_om`.
7. **CSP profiles.** Run the SAM model of the plant for the same year and weather file. Divide the hourly thermal output and parasitic load by the design thermal output, so that the model can scale the plant.
8. **Costs.** Convert capital costs to the units of `parameters.canonical_units()`. For example, an electrolyzer cost per kW of input power, times the electricity per kg ($`m_e`$), gives the cost per kg/h of capacity. Apply investment incentives and depreciation with `effective_capital_cost`.
9. **Checks before solving.** `Instance.create` checks equal lengths, finite values, ranges of production factors, non-negative loads, 0/1 outage flags, and the presence of every series and parameter the configuration needs. `read_inputs` also checks the units of every parameter and that hours run 1..|H| without gaps.

## 3. Input directory format

```
my_case/
├── parameters.csv
├── hourly.csv
└── case.json
```

### `parameters.csv`

| Column | Content |
|---|---|
| `name` | `<component>.<field>`, e.g. `electrolyzer.kwh_per_kg` |
| `value` | number |
| `unit` | must equal `canonical_units()[name]`; other units are refused |

Components: `financial`, `wind`, `pv`, `fuel_cell`, `battery`, `electrolyzer`, `tank`, `csp`, `tes`. The full list of fields, symbols, units and defaults is `reopt_h2_framework.PARAMETERS`; the [formulation](formulation.md) maps them to the symbols of Eq. (1)–(35). Only components of the chosen configuration need values. Required fields without a default (e.g. capital costs, size bounds, conversion rates) must be given.

### `hourly.csv`

One row per hour, `hour` = 1..|H|, then any of these columns:

| Column | Unit | Needed for |
|---|---|---|
| `wind_cf` | kWh/kW (0–1) | wind |
| `pv_cf` | kWh/kW (0–1) | PV |
| `export_price` | \$/kWh | revenue of energy delivered to the grid (default 0) |
| `hydrogen_price` | \$/kg | electrolyzer |
| `delivery_target` | kW | `delivery="target"` |
| `interconnection_limit` | kW | optional, Eq. (23) |
| `site_load` | kW | `delivery="site_load"` |
| `purchase_price` | \$/kWh | `delivery="site_load"`, CSP parasitics |
| `outage` | 0/1 | optional, `delivery="site_load"` |
| `csp_thermal_cf` | kWh_th/kW_th | CSP |
| `csp_parasitic` | kWh/kW_th | optional, CSP |

### `case.json`

```json
{
  "name": "wind-hydrogen",
  "configuration": {"wind": true, "pv": false, "battery": true, "electrolyzer": true,
                    "tank": true, "fuel_cell": true, "csp": false},
  "fixed_sizes": {"wind": 50000},
  "options": {"delivery": "target", "electrolyzer_operation": "variable",
              "electrolyzer_supply": "electric", "on_off": "binary", "shortfall_penalty": null}
}
```

- **`fixed_sizes`** fixes the size of a plant that is given in the study, such as an existing wind farm or a solar plant of known capacity. Accepted keys: candidate generators, `battery_kw`, `battery_kwh`, `electrolyzer`, `tank`, `csp`, `tes`.
- **`options`** are those of `ModelOptions` ([formulation](formulation.md), Part B).

`write_inputs(instance, directory, options)` writes this format, and `read_inputs(directory)` reads and validates it. `examples/prepare_inputs.py` goes from raw interval records to an input directory.
