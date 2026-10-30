import pytest

from reopt_h2_framework import CONFIGURATIONS
from reopt_h2_framework.synthetic import synthetic_instance


@pytest.fixture(scope="session")
def full_instance():
    return synthetic_instance(CONFIGURATIONS["wind+battery+electrolyzer+tank+fuel_cell"], hours=48)


def gurobi_available() -> bool:
    try:
        import gurobipy as gp
        with gp.Env(params={"OutputFlag": 0}):
            return True
    except Exception:
        return False
