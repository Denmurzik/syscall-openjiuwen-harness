"""Round 10, task 59: revenue written as (units * price) is read in phase 1."""
from pathlib import Path

import pytest

from harness.solvers import data_solver

EX = Path(__file__).parent / "fixtures" / "examples"


@pytest.mark.parametrize("phrase", ["(units * price)", "(units x price)", "(units times price)", "(price * units)",
                                    "(units multiplied by price)"])
def test_revenue_definition_variants(phrase):
    ws = EX / "data_analysis_0006" / "workspace"
    head = "The file sales.csv has columns order_id,region,product,month,units,price (300 rows). "
    plain = data_solver.parse(head + "What is the total revenue of product 'gizmo' in months 3 through 4 inclusive?", ws)
    spec = data_solver.parse(head + f"What is the total revenue {phrase} of product 'gizmo' in months 3 through 4 inclusive?", ws)
    assert isinstance(spec, dict) and spec == plain
