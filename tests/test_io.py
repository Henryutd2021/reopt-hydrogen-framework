import json

import numpy as np
import pandas as pd
import pytest

from reopt_h2_framework import ModelOptions, read_inputs, write_inputs
from reopt_h2_framework.cli import main
from reopt_h2_framework.synthetic import synthetic_instance


def test_round_trip(tmp_path):
    inst = synthetic_instance(hours=48)
    write_inputs(inst, tmp_path, ModelOptions(on_off="fixed_on"))
    back, opt = read_inputs(tmp_path)
    assert opt.on_off == "fixed_on"
    assert back.config == inst.config
    assert dict(back.fixed_sizes) == dict(inst.fixed_sizes)
    assert back.params.as_mapping() == inst.params.as_mapping()
    for k, v in inst.series.items():
        np.testing.assert_allclose(back.series[k], v)


def _edit_csv(path, fn):
    df = pd.read_csv(path)
    fn(df)
    df.to_csv(path, index=False)


def test_wrong_unit_is_refused(tmp_path):
    write_inputs(synthetic_instance(hours=48), tmp_path)

    def wrong_unit(df):
        df.loc[df["name"] == "wind.capital_cost", "unit"] = "$/MW"

    _edit_csv(tmp_path / "parameters.csv", wrong_unit)
    with pytest.raises(ValueError, match=r"wind.capital_cost must be given in \$/kW"):
        read_inputs(tmp_path)


def test_hour_gap_and_unknown_column(tmp_path):
    write_inputs(synthetic_instance(hours=48), tmp_path)

    def gap(df):
        df["hour"] = df["hour"].replace(10, 99)

    _edit_csv(tmp_path / "hourly.csv", gap)
    with pytest.raises(ValueError, match="without gaps"):
        read_inputs(tmp_path)
    write_inputs(synthetic_instance(hours=48), tmp_path)

    def extra(df):
        df["extra"] = 1.0

    _edit_csv(tmp_path / "hourly.csv", extra)
    with pytest.raises(ValueError, match="unknown columns"):
        read_inputs(tmp_path)


def test_unknown_case_key(tmp_path):
    write_inputs(synthetic_instance(hours=48), tmp_path)
    case = json.loads((tmp_path / "case.json").read_text())
    case["solver"] = "x"
    (tmp_path / "case.json").write_text(json.dumps(case))
    with pytest.raises(ValueError, match="unknown keys"):
        read_inputs(tmp_path)


def test_cli(tmp_path, capsys):
    inputs, out = tmp_path / "in", tmp_path / "out"
    assert main(["synthetic", str(inputs), "--hours", "48", "--configuration", "wind+battery+electrolyzer"]) == 0
    assert main(["run", str(inputs), "--out-dir", str(out), "--on-off", "fixed_on"]) == 0
    for f in ("sizes.csv", "annual.csv", "costs.csv", "indicators.csv", "hourly.csv", "balance_residuals.json"):
        assert (out / f).exists()
    residuals = json.loads((out / "balance_residuals.json").read_text())
    assert max(residuals.values()) < 1e-6
    assert "objective" in capsys.readouterr().out
