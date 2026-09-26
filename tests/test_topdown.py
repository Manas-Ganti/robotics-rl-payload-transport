"""Top-down clip frames (eval/topdown.py). Runs WITHOUT Isaac Sim."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from eval.topdown import FrameState, _robot_polygon, render_frame  # noqa: E402


def state(**kw) -> FrameState:
    base = dict(robot_xy=(0.0, 0.0), yaw=0.0, goal_xy=(3.0, 2.0), goal_tolerance=0.5,
                obstacles=[(1.5, 0.0, 0.4)], patch_half=5.0,
                lidar_angles=[-0.5, 0.0, 0.5], lidar_ranges=[2.0, 1.1, 3.0],
                trail=[(-1.0, 0.0), (0.0, 0.0)], title="t", readout="r")
    base.update(kw)
    return FrameState(**base)


def test_frame_is_rgb_uint8_of_requested_size():
    frame = render_frame(state(), size_px=320)
    assert frame.shape == (320, 320, 3) and frame.dtype == np.uint8
    assert frame.std() > 0  # something was drawn


def test_footprint_follows_heading():
    """Facing +y, the robot's front edge (x=+0.28 in its frame) lies at +y."""
    poly = _robot_polygon(state(yaw=math.pi / 2))
    assert poly[:, 1].max() == pytest.approx(0.28, abs=1e-9)
    assert poly[:, 1].min() == pytest.approx(-0.43, abs=1e-9)
