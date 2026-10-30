"""Command line: ``reopt-h2-framework synthetic DIR`` and ``reopt-h2-framework run DIR --out-dir OUT``."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from typing import Optional, Sequence

from .cases import CONFIGURATIONS, DEFAULT_OPTIONS
from .io import read_inputs, write_inputs
from .model import DELIVERY_MODES, ELECTROLYZER_OPERATION, ON_OFF, ModelOptions, build_model
from .results import summarize
from .solve import solve
from .synthetic import synthetic_instance


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="reopt-h2-framework", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    syn = sub.add_parser("synthetic", help="write a synthetic input directory")
    syn.add_argument("directory")
    syn.add_argument("--configuration", choices=sorted(CONFIGURATIONS), default="wind+battery+electrolyzer+tank+fuel_cell")
    syn.add_argument("--hours", type=int, default=72)
    syn.add_argument("--seed", type=int, default=0)

    run = sub.add_parser("run", help="solve an input directory and write result tables")
    run.add_argument("directory")
    run.add_argument("--out-dir", required=True)
    run.add_argument("--solver", default="highs")
    run.add_argument("--delivery", choices=DELIVERY_MODES, help="override case.json")
    run.add_argument("--electrolyzer-operation", choices=ELECTROLYZER_OPERATION, help="override case.json")
    run.add_argument("--on-off", choices=ON_OFF, help="override case.json")
    run.add_argument("--time-limit", type=float, help="solver time limit, seconds")

    a = ap.parse_args(argv)
    if a.command == "synthetic":
        cfg = CONFIGURATIONS[a.configuration]
        inst = synthetic_instance(cfg, hours=a.hours, seed=a.seed, name=a.configuration)
        write_inputs(inst, a.directory, DEFAULT_OPTIONS.get(a.configuration, ModelOptions()))
        print(f"wrote {a.directory}")
        return 0

    inst, opt = read_inputs(a.directory)
    changes = {k: v for k, v in (("delivery", a.delivery), ("electrolyzer_operation", a.electrolyzer_operation),
                                 ("on_off", a.on_off)) if v is not None}
    opt = replace(opt, **changes)
    pm = build_model(inst, opt)
    solver_opts = {}
    if a.time_limit is not None:
        solver_opts = {"time_limit": a.time_limit} if a.solver == "highs" else {"TimeLimit": a.time_limit}
    res = solve(pm, solver=a.solver, options=solver_opts)
    s = summarize(pm)
    out = s.to_csv(a.out_dir)
    print(f"{inst.name}: objective {res.objective:,.0f} $ (present value); tables in {out}")
    print(s.sizes.round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
