"""Degradation classification. Runs WITHOUT Isaac Sim.

Pins the rule that a physically impossible grid cell cannot make a policy look
"catastrophic": the first Carter eval had a 0.99 success drop at a 20 deg slope
the motors cannot climb with the nominal payload.
"""

from __future__ import annotations

import pytest

from eval.metrics import CellMetrics, classify_degradation


def cell(value: float, success: float, regime: str, feasible: bool = True) -> CellMetrics:
    return CellMetrics(
        axis="slope_angle_deg", value=value, regime=regime, num_episodes=100,
        success_rate=success, collision_rate=0.0, timeout_rate=1.0 - success,
        path_efficiency=1.0, mean_final_distance_m=0.0, feasible=feasible,
    )


GRID = [(0, 0.95, "train"), (5, 0.94, "train"), (10, 0.93, "train"),
        (12, 0.92, "ood"), (15, 0.90, "ood")]


def test_infeasible_cliff_does_not_make_a_policy_catastrophic():
    cells = [cell(v, s, r) for v, s, r in GRID] + [cell(20, 0.0, "ood", feasible=False)]
    profile = classify_degradation(cells, cliff_drop_threshold=0.25)
    assert profile.classification == "robust"
    assert profile.infeasible_values == [20]
    assert profile.max_drop == pytest.approx(0.02)
    # every cell is still reported (for plots), including the excluded one
    assert profile.values == [0, 5, 10, 12, 15, 20]


def test_same_cliff_at_a_feasible_cell_is_catastrophic():
    cells = [cell(v, s, r) for v, s, r in GRID] + [cell(20, 0.0, "ood", feasible=True)]
    profile = classify_degradation(cells, cliff_drop_threshold=0.25)
    assert profile.classification == "catastrophic"
    assert profile.max_drop_at == 20
    assert profile.infeasible_values == []


def test_all_infeasible_is_undefined_not_a_crash():
    cells = [cell(v, 0.0, "ood", feasible=False) for v in (18, 20)]
    assert classify_degradation(cells).classification == "undefined"
