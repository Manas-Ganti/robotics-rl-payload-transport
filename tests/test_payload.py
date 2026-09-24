"""Payload spec + inertia math. Runs WITHOUT Isaac Sim.

The payload is the study's dynamics axis, so the pure pieces that feed the
simulator -- which attach mode is configured, and the box inertia written
alongside every per-env mass -- are pinned here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from env.config import load_train_config
from env.payload import ATTACH_MODES, PayloadSpec, box_inertia_diagonal

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CONFIG = REPO_ROOT / "configs" / "train.yaml"


def test_real_config_builds_a_deck_link_spec():
    spec = PayloadSpec.from_config(load_train_config(TRAIN_CONFIG), mass_kg=25.0)
    assert spec.attach_mode == "deck_link"
    assert spec.mass_kg == 25.0
    assert len(spec.size_m) == 3 and min(spec.size_m) > 0


def test_unknown_attach_mode_is_rejected():
    with pytest.raises(ValueError, match="attach_mode"):
        PayloadSpec(mass_kg=1.0, size_m=(0.4, 0.3, 0.2), offset_m=(0, 0, 0), attach_mode="glue")
    assert "deck_link" in ATTACH_MODES


def test_box_inertia_matches_closed_form():
    # 12 kg, 1 x 2 x 3 m: I = m/12 * (sum of squares of the other two sides)
    ixx, iyy, izz = box_inertia_diagonal(12.0, (1.0, 2.0, 3.0))
    assert (ixx, iyy, izz) == pytest.approx((13.0, 10.0, 5.0))


def test_box_inertia_scales_linearly_with_mass():
    """set_link_payload_mass relies on this: inertia is rewritten per mass."""
    light = box_inertia_diagonal(10.0, (0.4, 0.3, 0.2))
    heavy = box_inertia_diagonal(90.0, (0.4, 0.3, 0.2))
    assert [h / l for h, l in zip(heavy, light)] == pytest.approx([9.0, 9.0, 9.0])


# ---------------------------------------------------------------------------
# Climb feasibility (motor torque cap) -- impossible cells are not "failures"
# ---------------------------------------------------------------------------
def _robot():
    from env.config import load_train_config

    return load_train_config(TRAIN_CONFIG).to_dict()["robot"]


def test_climb_torque_matches_the_documented_derivation():
    """robot.yaml's comment: 90 kg on 20 deg needs 76.5 N*m; 45 kg on 10 deg 28.3."""
    from env.payload import climb_torque_per_wheel_nm

    r = _robot()["wheel_radius_m"]
    assert climb_torque_per_wheel_nm(76.2 + 90.0, 20.0, r) == pytest.approx(76.5, abs=0.1)
    assert climb_torque_per_wheel_nm(76.2 + 45.0, 10.0, r) == pytest.approx(28.3, abs=0.1)
    assert climb_torque_per_wheel_nm(100.0, 0.0, r) == 0.0


def test_slope_axis_20deg_at_nominal_payload_is_infeasible():
    """The slope sweep pins payload at the eval nominal (25 kg): 15 deg is
    climbable, 20 deg needs 46.6 N*m > the 42.5 cap -- impossible for ANY policy."""
    from env.payload import climb_feasible

    robot = _robot()
    assert climb_feasible(robot, 25.0, 15.0)
    assert not climb_feasible(robot, 25.0, 20.0)
    assert climb_feasible(robot, 90.0, 0.0)   # the payload axis runs on flat ground


def test_unlimited_cap_is_always_feasible():
    from env.payload import climb_feasible

    robot = {**_robot(), "actuator": {"effort_limit_sim": None}}
    assert climb_feasible(robot, 500.0, 45.0)
