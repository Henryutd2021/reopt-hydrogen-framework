"""Hydrogen-extended sizing and dispatch of wind, solar, storage, electrolyzer, hydrogen tank and fuel cell
(framework of Li & Zhang, J. Mech. Des. 146 (2024) 020903, and related studies)."""

from .cases import CONFIGURATIONS, Case, configuration_cases, hydrogen_price_cases, run_cases
from .inputs import SERIES, SERIES_UNITS, Configuration, Instance
from .io import read_inputs, write_inputs
from .model import ModelOptions, PlanningModel, build_model
from .parameters import PARAMETERS, FinancialFactors, Parameters, canonical_units, effective_capital_cost, pwf
from .results import Summary, balance_residuals, summarize
from .solve import SolveError, SolveResult, solve

__version__ = "1.0.0"
__all__ = [
    "CONFIGURATIONS", "Case", "configuration_cases", "hydrogen_price_cases", "run_cases",
    "SERIES", "SERIES_UNITS", "Configuration", "Instance", "read_inputs", "write_inputs",
    "ModelOptions", "PlanningModel", "build_model",
    "PARAMETERS", "FinancialFactors", "Parameters", "canonical_units", "effective_capital_cost", "pwf",
    "Summary", "balance_residuals", "summarize", "SolveError", "SolveResult", "solve",
]
