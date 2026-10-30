"""Read and write the input directory format (schemas in ``docs/data_pipeline.md``).

An input directory holds three files:

    parameters.csv   name, value, unit    (units must equal ``parameters.canonical_units()``)
    hourly.csv       hour, <series ...>   (series names and units: ``inputs.SERIES_UNITS``)
    case.json        {"name", "configuration": {...}, "fixed_sizes": {...}, "options": {...}}

``hourly.csv`` has one row per hour, ``hour`` = 1..|H| without gaps.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .inputs import SERIES, Configuration, Instance
from .model import ModelOptions
from .parameters import Parameters, canonical_units


def read_parameters(path) -> Parameters:
    df = pd.read_csv(path)
    for col in ("name", "value", "unit"):
        if col not in df.columns:
            raise ValueError(f"{Path(path).name}: missing column {col!r}")
    if df["name"].duplicated().any():
        raise ValueError(f"{Path(path).name}: duplicated parameters {sorted(df.loc[df['name'].duplicated(), 'name'])}")
    units = canonical_units()
    values = {}
    for r in df.itertuples(index=False):
        name = str(r.name)
        if name not in units:
            raise ValueError(f"{Path(path).name}: unknown parameter {name!r}")
        if str(r.unit) != units[name]:
            raise ValueError(f"{Path(path).name}: {name} must be given in {units[name]}, not {r.unit!r}")
        values[name] = float(r.value)
    return Parameters.from_mapping(values)


def write_parameters(params: Parameters, path) -> None:
    units = canonical_units()
    rows = [{"name": k, "value": v, "unit": units[k]} for k, v in params.as_mapping().items()]
    pd.DataFrame(rows, columns=["name", "value", "unit"]).to_csv(path, index=False)


def read_inputs(directory) -> Tuple[Instance, ModelOptions]:
    """Read an input directory into a validated instance and its model options."""
    d = Path(directory)
    params = read_parameters(d / "parameters.csv")
    hourly = pd.read_csv(d / "hourly.csv")
    if "hour" not in hourly.columns:
        raise ValueError("hourly.csv: missing column 'hour'")
    if not np.array_equal(hourly["hour"].to_numpy(), np.arange(1, len(hourly) + 1)):
        raise ValueError(f"hourly.csv: 'hour' must run 1..{len(hourly)} without gaps")
    unknown = set(hourly.columns) - set(SERIES) - {"hour"}
    if unknown:
        raise ValueError(f"hourly.csv: unknown columns {sorted(unknown)}")
    series = {c: hourly[c].to_numpy(dtype=float) for c in hourly.columns if c != "hour"}
    case = json.loads((d / "case.json").read_text())
    unknown_keys = set(case) - {"name", "configuration", "fixed_sizes", "options"}
    if unknown_keys:
        raise ValueError(f"case.json: unknown keys {sorted(unknown_keys)}")
    config = Configuration(**case.get("configuration", {}))
    options = ModelOptions(**case.get("options", {}))
    inst = Instance.create(config, params, series, case.get("fixed_sizes", {}), case.get("name", d.name))
    return inst, options


def write_inputs(inst: Instance, directory, options: ModelOptions = ModelOptions()) -> Path:
    """Write an instance (and options) as an input directory."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    write_parameters(inst.params, d / "parameters.csv")
    df = pd.DataFrame({"hour": np.arange(1, inst.hours + 1)})
    for k in SERIES:
        if k in inst.series:
            df[k] = inst.series[k]
    df.to_csv(d / "hourly.csv", index=False)
    case = {"name": inst.name, "configuration": asdict(inst.config), "fixed_sizes": dict(inst.fixed_sizes),
            "options": asdict(options)}
    (d / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    return d
