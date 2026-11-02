# REopt hydrogen framework: sizing and dispatch of wind, solar, storage and hydrogen systems

Author: [Honglin Li](https://henryutd2021.github.io/)

A documented, runnable implementation of the hydrogen-extended techno-economic optimization model in:

> Honglin Li and Jie Zhang. **Towards sustainable integration: techno-economic analysis and future perspectives of co-located wind and hydrogen energy systems.** *Journal of Mechanical Design* 146(2) (2024) 020903. [doi:10.1115/1.4063971](https://doi.org/10.1115/1.4063971)

and used, with the options described below, in the related studies listed under [Citation](#citation).

The model decides the sizes of a battery, an electrolyzer, a hydrogen tank and a fuel cell next to a wind or solar plant, and how to run them hour by hour over a year. It minimizes the system's life-cycle cost: capital and operation-and-maintenance costs, net of the revenue from electricity delivered to the grid, hydrogen sales and production incentives. It belongs to the REopt formulation family, NREL's techno-economic optimization model for distributed energy systems, and extends it with hydrogen production, storage and reconversion.

## Research question

How do electrolyzers, hydrogen storage and fuel cells change the cost, curtailment and firm output of a renewable plant? Which configuration and sizes are economic under given resource, price and cost conditions?

## System and scope

| Aspect | Content |
|---|---|
| Components | wind, PV, battery, electrolyzer, hydrogen tank, fuel cell; optionally a CSP plant with thermal storage supplying a high-temperature electrolyzer with heat |
| Decisions | component sizes (kW, kWh, kg/h, kg); hourly production, its allocation (to storage, grid, electrolyzer, curtailment), storage operation, hydrogen production, storage, sale and use in the fuel cell |
| Electricity delivery | a stabilized output delivered to the grid in every hour (Eq. 21), or a site load served together with the grid, with outage hours ([options](docs/formulation.md#part-b-options-used-in-related-studies)) |
| Time | hourly; one representative year over a multi-year analysis period (present-worth factors) |
| Economics | capital costs after investment incentives, fixed and variable O&M, energy export and hydrogen revenue, production incentives, taxes |
| Model type | mixed-integer linear program (linopy; HiGHS by default, Gurobi optional) |

The complete mathematics, keyed to Eqs. (1)–(35) of the article, is in [docs/formulation.md](docs/formulation.md).

## What is in this repository

| Path | Content |
|---|---|
| [`docs/formulation.md`](docs/formulation.md) | sets, parameters, variables, objective and every constraint, Eqs. (1)–(35); conventions; options for site load and outages, constant-rate electrolysis and CSP heat supply; indicator definitions |
| [`docs/data_pipeline.md`](docs/data_pipeline.md) | public sources for each input, the preparation steps, and the input file formats |
| [`docs/scenarios.md`](docs/scenarios.md) | the system configurations and sensitivity families of the studies, and how to set each up |
| [`docs/architecture.md`](docs/architecture.md) | modules, data flow, solvers, problem size, output tables, tests |
| [`src/reopt_h2_framework/`](src/reopt_h2_framework/) | the implementation: parameters and validation, model, solver interface, result tables, configurations and case runs, preprocessing helpers, input files, command line |
| [`examples/`](examples/) | `minimal.py`, `configurations.py` (four configurations and five sensitivity families), `solar_hydrogen.py` (CSP with thermal storage, and PV) and `prepare_inputs.py` (raw interval records to an input directory); all on synthetic data |
| [`tests/`](tests/) | pytest suite: parameters, inputs, input files, preprocessing, model constraints and balances, options, case runs, command line |

## Quick start

```bash
git clone https://github.com/Henryutd2021/reopt-hydrogen-framework.git
cd reopt-hydrogen-framework
python3 -m venv .venv && source .venv/bin/activate    # needs Python 3.10+ (check: python3 --version); if older, use e.g. python3.12
pip install -e ".[test]"
python examples/minimal.py                              # synthetic three-day example, HiGHS, a few seconds
pytest
```

With conda instead: `conda env create -f environment.yml && conda activate reopt-h2-framework && pip install -e .`

The command line works on input directories:

```bash
reopt-h2-framework synthetic my_inputs/ --configuration wind+battery+electrolyzer+tank+fuel_cell
reopt-h2-framework run my_inputs/ --out-dir outputs/base
reopt-h2-framework run my_inputs/ --out-dir outputs/site --delivery site_load
```

In Python:

```python
from reopt_h2_framework import ModelOptions, build_model, read_inputs, solve, summarize

inst, options = read_inputs("my_inputs/")              # your own data, see docs/data_pipeline.md
pm = build_model(inst, options)
solve(pm)                                              # or solve(pm, solver="gurobi")
summary = summarize(pm)
summary.to_csv("outputs/base")                         # sizes, annual, costs, indicators, hourly
```

**About the example data:** `reopt_h2_framework.synthetic` generates an invented site (a few days of wind, PV, CSP heat, prices and load, with round illustrative costs). It shows how the model behaves and how to read its output. Its numbers say nothing about any real site, market or technology.

## Using your own data

1. Build `parameters.csv`, `hourly.csv` and `case.json` as described in [docs/data_pipeline.md](docs/data_pipeline.md). `reopt_h2_framework.preprocess` and `examples/prepare_inputs.py` cover the generic steps. Units are checked when the files are read.
2. Fix the size of the plant you study (e.g. the wind farm) in `case.json` → `fixed_sizes`, and choose the configuration and options.
3. Run the configurations and sensitivity cases of [docs/scenarios.md](docs/scenarios.md) with `run_cases`.

A full year (8,760 hours) of the largest configuration has about 193,000 variables and 184,000 constraints; see [docs/architecture.md](docs/architecture.md) for sizes and solver notes.

## What this repository does not contain

This repository presents the modelling framework. It does not contain the code, the input data or the results used for the published articles. For their results, see the articles.

## Citation

If you use this framework, please cite the *Journal of Mechanical Design* article above. Citation metadata for it, for this software and for the related studies are in [`CITATION.cff`](CITATION.cff). The related studies are:

- H. Li, J. Rahman, J. Zhang. Optimal planning of co-located wind energy and hydrogen plants: a techno-economic analysis. *Journal of Physics: Conference Series* 2265 (2022) 042063. [doi:10.1088/1742-6596/2265/4/042063](https://doi.org/10.1088/1742-6596/2265/4/042063)
- H. Li, J. Kiviluoma, J. Zhang. Techno-economic analysis for co-located solar and hydrogen plants. *2023 19th International Conference on the European Energy Market (EEM)*, 2023. [doi:10.1109/EEM58374.2023.10161786](https://doi.org/10.1109/EEM58374.2023.10161786)
- H. Li, M. Li, J. Carroll, J. Zhang. Techno-economic analysis incorporating intelligent operation and maintenance management: a case study of an integrated offshore wind and hydrogen energy system. *Journal of Physics: Conference Series* 2767 (2024) 062016. [doi:10.1088/1742-6596/2767/6/062016](https://doi.org/10.1088/1742-6596/2767/6/062016)

The formulation family:

- D. Cutler, D. Olis, E. Elgqvist, X. Li, N. Laws, N. DiOrio, H. Walker, K. Anderson. *REopt: A Platform for Energy System Integration and Optimization.* NREL/TP-7A40-70022, National Renewable Energy Laboratory, 2017. [doi:10.2172/1395453](https://doi.org/10.2172/1395453)

This framework is an independent implementation. It contains no REopt source code and is not affiliated with or endorsed by NREL.

## Licence

- Documentation (`README.md` and `docs/`): [CC BY 4.0](LICENSE-docs).
- Everything else in the repository (code, tests, examples, configuration and metadata files): [MIT](LICENSE).
