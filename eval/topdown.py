"""Top-down video frames drawn from simulator STATE -- no RTX renderer needed.

PURE: numpy + matplotlib (Agg). Isaac Sim's RTX viewport segfaulted at startup
on ARC's L40S nodes (headless, driver 595.x), so clips are drawn from the
quantities the simulator already exposes: robot pose, the episode's obstacles
and goal, and the lidar ranges the policy actually saw. For diagnosis this is
more informative than a 3D view: the overlay shows speed, height above the
expected ground and tilt, so a robot being thrown or stuck is visible at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class FrameState:
    """Everything one frame shows, in the env-local (patch) frame, metres."""

    robot_xy: Tuple[float, float]
    yaw: float
    goal_xy: Tuple[float, float]
    goal_tolerance: float
    obstacles: Sequence[Tuple[float, float, float]]          # (x, y, radius)
    patch_half: float
    lidar_angles: Sequence[float] = ()
    lidar_ranges: Sequence[float] = ()
    trail: List[Tuple[float, float]] = field(default_factory=list)
    footprint_x: Tuple[float, float] = (-0.43, 0.28)         # rear, front in the robot frame
    footprint_half_width: float = 0.314
    title: str = ""
    readout: str = ""


def _robot_polygon(state: FrameState) -> np.ndarray:
    (rear, front), hw = state.footprint_x, state.footprint_half_width
    local = np.array([[front, hw], [front, -hw], [rear, -hw], [rear, hw]])
    c, s = math.cos(state.yaw), math.sin(state.yaw)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.asarray(state.robot_xy)


def render_frame(state: FrameState, size_px: int = 720) -> np.ndarray:
    """Draw one top-down frame; returns an (H, W, 3) uint8 RGB array."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Polygon, Rectangle

    dpi = 100
    fig = Figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.9])
    h = state.patch_half
    ax.set_xlim(-h - 0.3, h + 0.3)
    ax.set_ylim(-h - 0.3, h + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Rectangle((-h, -h), 2 * h, 2 * h, facecolor="#eceae4", edgecolor="#555", lw=1.5))
    for x, y, r in state.obstacles:
        ax.add_patch(Circle((x, y), r, facecolor="#b5483a", edgecolor="#7a2e25", lw=0.8))
    ax.add_patch(Circle(state.goal_xy, state.goal_tolerance, facecolor="#3aa55d", alpha=0.35, lw=0))
    ax.plot(*state.goal_xy, marker="*", markersize=16, color="#1f7a3f")

    rx, ry = state.robot_xy
    for a, r in zip(state.lidar_angles, state.lidar_ranges):
        ang = state.yaw + a
        ax.plot([rx, rx + r * math.cos(ang)], [ry, ry + r * math.sin(ang)], color="#4a7fd4", lw=0.4, alpha=0.45)
    if len(state.trail) > 1:
        t = np.asarray(state.trail)
        ax.plot(t[:, 0], t[:, 1], color="#2c3e66", lw=1.4, alpha=0.8)
    ax.add_patch(Polygon(_robot_polygon(state), closed=True, facecolor="#2c3e66", edgecolor="black", lw=1.0))
    ax.arrow(rx, ry, 0.45 * math.cos(state.yaw), 0.45 * math.sin(state.yaw),
             width=0.03, head_width=0.16, color="#f2c14e", length_includes_head=True)

    fig.text(0.02, 0.955, state.title, fontsize=11, weight="bold", family="monospace")
    fig.text(0.02, 0.915, state.readout, fontsize=9, family="monospace")

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[:, :, :3].copy()
