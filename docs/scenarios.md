# Configurations and sensitivity cases

The studies built on this framework share one structure. They compare system configurations at a site, then vary one group of inputs at a time around a chosen configuration. This page lists both, and shows how to express each in code. It contains no study results; for those, see the articles.

## Configurations

`reopt_h2_framework.CONFIGURATIONS` holds the configurations compared in the studies. `configuration_cases()` turns them into cases with their default options.

| Name | Components sized by the model | Typical use |
|---|---|---|
| `wind+battery` | battery | a wind farm with conventional storage that delivers a stabilized output (base configuration) |
| `wind+battery+electrolyzer` | battery, electrolyzer | surplus wind converted to hydrogen for sale |
| `wind+electrolyzer+tank+fuel_cell` | electrolyzer, tank, fuel cell | hydrogen storage and a fuel cell in place of the battery |
| `wind+battery+electrolyzer+tank+fuel_cell` | all of the above | battery and hydrogen storage combined |
| `csp+tes+electrolyzer` | thermal storage, electrolyzer (CSP plant size given) | a CSP plant supplying a high-temperature electrolyzer with heat, at variable or constant rate |
| `pv+electrolyzer` | electrolyzer (PV plant size given) | PV supplying a low-temperature electrolyzer at variable rate |

The examples fix the size of the wind or solar plant through `fixed_sizes`. To size the plant as well, leave it free. Its bounds are `wind.min_size` and `wind.max_size` (likewise for PV); a CSP plant has a lower bound of zero and the upper bound `csp.max_size`.

Two settings define how the system meets electricity needs ([formulation](formulation.md)):

- **`delivery="target"`** (Eq. 21): the system delivers a stabilized output $\delta^{d}_h$ to the grid in every hour.
- **`delivery="site_load"`** (Part B1): the system serves a site load together with the grid, sells surplus energy, and must serve the load alone during outage hours. A benchmark is a grid-connected wind farm serving a load. It is extended step by step with an electrolyzer and tank, and then a fuel cell, and tested against a block of randomly chosen outage hours.

## Sensitivity families

| Family | What varies | In code |
|---|---|---|
| Hydrogen sale price | a constant \$/kg over a range | `hydrogen_price_cases([...])` |
| Wind resource | the hourly wind profile of sites in different market regions | `Case(series={"wind_cf": profile})` |
| Electricity prices | the price year or market region | `Case(series={"export_price": ..., "purchase_price": ...})` |
| Future technology costs | cost trajectories of a future year (e.g. ATB) for electrolyzer, fuel cell, battery and wind | `Case(parameter_factors={"electrolyzer.capital_cost": f, ...})` or `parameter_values` |
| Electrolyzer operation | variable or constant production rate | `ModelOptions(electrolyzer_operation="constant")` |
| Operation and maintenance | availability and O&M cost of the wind farm under different maintenance strategies (for example, from a separate O&M simulation) | `preprocess.apply_availability`, `wind.fixed_om` |
| Wind share and incentives | wind size relative to the electrolyzer, and production incentives for electricity or hydrogen | `fixed_sizes`, `<tech>.incentive_rate` |

`run_cases(base, cases)` solves a list of cases and returns one row per case. Each row holds the objective, the sizes, the indicators (unused share of available wind and PV energy, electrolyzer capacity factor, LCOH, LCOE) and annual hydrogen production. `examples/configurations.py` runs one example of each family on synthetic data.

## Reading the outputs

- **Sizes:** the capacities chosen for the components (`summary.sizes`).
- **Life-cycle cost:** the objective of Eq. (1); its parts are in `summary.costs`.
- **Hourly dispatch:** `summary.hourly`, e.g. for plots of the stabilized output, storage levels and hydrogen production.
- **Indicators:** `summary.indicators`, with the definitions given in the [formulation](formulation.md#indicators-computed-after-the-solve-resultssummarize).
