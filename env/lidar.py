"""Analytic 2D lidar: exact ray casting against the episode's TerrainSpec.

PURE LOGIC -- no Isaac Sim import, and no hard torch dependency. The single
implementation, :func:`cast_rays`, is written against an array module ``xp``
(``numpy`` or ``torch``) using only operations whose signatures the two share,
so the code the unit tests exercise on a laptop is literally the code that runs
on the GPU. There is no "GPU mirror" to keep in sync.

WHY THIS EXISTS (instead of Isaac Lab's RayCaster)
--------------------------------------------------
Isaac Lab's ``RayCaster`` warps against STATIC meshes. The obstacles here are
kinematic bodies re-posed every episode, so the RayCaster was likely blind to
them: the policy would train without seeing what it collides with, and the
obstacle-density axis would measure luck. The smoke test's "depth saturated"
check would not catch it either -- rays still hit the floor.

Every obstacle in this project is a vertical cylinder whose centre and radius
are already known exactly (``TerrainSpec.obstacles``), so ray-circle
intersection gives the true range in closed form: exact, a few batched tensor
ops, and no rendering.

GEOMETRY / MODELLING CHOICES
----------------------------
* Rays are cast in the horizontal (x, y) projection, in the env-local frame
  (the same frame as ``TerrainSpec``, the occupancy grid, and A*). On a slope
  this reports HORIZONTAL range; the along-surface range differs by at most
  1/cos(slope) (~6.4% at the 20 deg OOD extreme, only along the gradient). A
  world-horizontal 3D ray would be worse: on a 20 deg slope it strikes the
  uphill ground ~0.3 m away and reports a phantom wall.
* Obstacles are taller (``terrain.obstacle_height_m``) than the sensor mount,
  so every obstacle is visible; no vertical test is needed.
* Optionally the patch boundary acts as a wall (``boundary_as_obstacle``). The
  patch edge is a drop-off the robot must not cross, and Nav2's costmap is
  bounded to the same patch -- exposing it keeps the comparison even.
* A ray whose origin is inside an obstacle (or outside the patch) returns 0:
  the robot is already in contact, which the collision sensor terminates.

CAVEAT -- this sensor reads the SPEC, not the simulator. If the sim fails to
move an obstacle (a Tier 1 silent failure), this lidar still "sees" it where
the spec says it is. The smoke test therefore cross-checks sim obstacle poses
against the spec (``training/train.py::run_smoke_test``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np

# Returned for "no hit" before clipping; any value > max_range works.
_FAR = 1.0e6
# Direction components smaller than this are treated as axis-parallel.
_EPS = 1.0e-9


@dataclass(frozen=True)
class LidarSpec:
    """Ray layout + range limits, resolved from ``sensors.raycaster``."""

    num_rays: int
    horizontal_fov_deg: float
    max_range_m: float
    boundary_as_obstacle: bool
    half_size_m: float  # the patch spans [-half, +half] on both axes

    @classmethod
    def from_config(cls, cfg: Any) -> "LidarSpec":
        data = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
        rays = data["sensors"]["raycaster"]  # ray geometry is shared by both modalities
        lidar = data["sensors"]["analytic_lidar"]
        spec = cls(
            num_rays=int(rays["num_rays"]),
            horizontal_fov_deg=float(rays["horizontal_fov_deg"]),
            max_range_m=float(rays["max_range_m"]),
            boundary_as_obstacle=bool(lidar["boundary_as_obstacle"]),
            half_size_m=float(data["env"]["terrain_size_m"]) / 2.0,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.num_rays < 1:
            raise ValueError(f"num_rays must be >= 1, got {self.num_rays}")
        if not 0.0 < self.horizontal_fov_deg <= 360.0:
            raise ValueError(f"horizontal_fov_deg must be in (0, 360], got {self.horizontal_fov_deg}")
        if self.max_range_m <= 0.0:
            raise ValueError(f"max_range_m must be > 0, got {self.max_range_m}")
        if self.half_size_m <= 0.0:
            raise ValueError(f"terrain half size must be > 0, got {self.half_size_m}")

    def ray_angles(self) -> np.ndarray:
        """Ray bearings (rad) relative to the robot heading, left-to-right order
        matching ``LidarPatternCfg``: evenly spaced over [-fov/2, +fov/2].

        A full 360 deg sweep drops the duplicate endpoint (-pi and +pi are the
        same ray).
        """
        half = math.radians(self.horizontal_fov_deg) / 2.0
        if self.num_rays == 1:
            return np.zeros(1)
        if self.horizontal_fov_deg >= 360.0:
            return np.linspace(-half, half, self.num_rays, endpoint=False)
        return np.linspace(-half, half, self.num_rays)


def obstacles_to_arrays(
    obstacle_lists: Sequence[Sequence[Any]], max_obstacles: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack per-env obstacle lists into fixed-shape arrays for batched casting.

    Returns ``(xyr, valid)`` with shapes ``(N, K, 3)`` and ``(N, K)``. Obstacles
    carrying a pool ``slot`` are written at that index, so row ``k`` describes
    exactly the sim prim ``Obstacle_k``; unused rows are ``valid=False``.
    """
    n = len(obstacle_lists)
    xyr = np.zeros((n, max_obstacles, 3), dtype=np.float32)
    valid = np.zeros((n, max_obstacles), dtype=bool)
    for i, obstacles in enumerate(obstacle_lists):
        for j, obs in enumerate(obstacles):
            k = obs.slot if getattr(obs, "slot", -1) >= 0 else j
            if k >= max_obstacles:
                raise ValueError(f"obstacle slot {k} exceeds the pool size {max_obstacles}")
            if valid[i, k]:
                raise ValueError(f"env {i}: two obstacles claim pool slot {k}")
            xyr[i, k] = (obs.x, obs.y, obs.radius)
            valid[i, k] = True
    return xyr, valid


def cast_rays(
    xp: Any,
    origin_xy: Any,
    yaw: Any,
    ray_angles: Any,
    obstacle_xyr: Any,
    obstacle_valid: Any,
    *,
    max_range_m: float,
    half_size_m: float,
    boundary_as_obstacle: bool,
) -> Any:
    """Batched exact ray-circle (+ optional box-boundary) casting.

    Parameters (``xp`` arrays -- all numpy, or all torch on one device)
    ----------
    xp             : ``numpy`` or ``torch``
    origin_xy      : (N, 2)    sensor position, env-local frame
    yaw            : (N,)      robot heading (rad)
    ray_angles     : (R,)      bearings relative to heading (rad)
    obstacle_xyr   : (N, K, 3) obstacle centre x, y and radius
    obstacle_valid : (N, K)    bool; False rows are ignored (K may be 0)

    Returns
    -------
    (N, R) ranges in metres, clipped to ``[0, max_range_m]``.
    """
    # Ray directions, (N, R, 2). Unit length -> the quadratic's leading coeff is 1.
    theta = yaw[:, None] + ray_angles[None, :]
    dx, dy = xp.cos(theta), xp.sin(theta)

    ranges = xp.full_like(dx, _FAR)

    # ---- obstacles: |o + t d - c|^2 = r^2  ->  t^2 + 2 b t + cc = 0 ----------
    if obstacle_xyr.shape[1] > 0:
        fx = origin_xy[:, None, 0:1] - obstacle_xyr[:, None, :, 0]  # (N, 1, K)
        fy = origin_xy[:, None, 1:2] - obstacle_xyr[:, None, :, 1]
        r = obstacle_xyr[:, None, :, 2]
        b = fx * dx[..., None] + fy * dy[..., None]                   # (N, R, K)
        cc = fx * fx + fy * fy - r * r                                # (N, 1, K)
        disc = b * b - cc
        hit = disc >= 0.0
        t_near = -b - xp.sqrt(xp.where(hit, disc, xp.zeros_like(disc)))
        # cc <= 0: origin inside (or on) the disc -> range 0.
        # cc > 0 and t_near < 0: both roots negative (product = cc > 0) -> behind us.
        t = xp.where(cc <= 0.0, xp.zeros_like(t_near), t_near)
        ok = hit & (t >= 0.0) & obstacle_valid[:, None, :]
        t = xp.where(ok, t, xp.full_like(t, _FAR))
        ranges = xp.minimum(ranges, xp.amin(t, -1))

    # ---- patch boundary: exit distance from the square [-h, h]^2 -------------
    if boundary_as_obstacle:
        h = half_size_m
        ox, oy = origin_xy[:, 0:1], origin_xy[:, 1:2]                  # (N, 1)
        tx = _axis_exit(xp, ox, dx, h)
        ty = _axis_exit(xp, oy, dy, h)
        t_box = xp.minimum(tx, ty)
        inside = (xp.abs(ox) <= h) & (xp.abs(oy) <= h)
        t_box = xp.where(inside, t_box, xp.zeros_like(t_box))
        ranges = xp.minimum(ranges, t_box)

    return xp.clip(ranges, 0.0, max_range_m)


def _axis_exit(xp: Any, o: Any, d: Any, h: float) -> Any:
    """Distance along each ray until coordinate ``o + t d`` leaves [-h, h]."""
    safe_d = xp.where(xp.abs(d) < _EPS, xp.full_like(d, _EPS), d)
    t_pos = (h - o) / safe_d
    t_neg = (-h - o) / safe_d
    far = xp.full_like(d, _FAR)
    return xp.where(d > _EPS, t_pos, xp.where(d < -_EPS, t_neg, far))


def cast_rays_single(
    origin_xy: Tuple[float, float],
    yaw: float,
    obstacles: Sequence[Any],
    spec: LidarSpec,
) -> np.ndarray:
    """Convenience: one robot, a list of ``Obstacle``s -> (R,) ranges (NumPy).

    For tests and CPU-side consumers (e.g. a baseline that wants the same
    sensor as the policy). Uses the exact same :func:`cast_rays`.
    """
    xyr, valid = obstacles_to_arrays([list(obstacles)], max(len(obstacles), 1))
    out = cast_rays(
        np,
        np.asarray([origin_xy], dtype=np.float64),
        np.asarray([yaw], dtype=np.float64),
        spec.ray_angles(),
        xyr.astype(np.float64),
        valid,
        max_range_m=spec.max_range_m,
        half_size_m=spec.half_size_m,
        boundary_as_obstacle=spec.boundary_as_obstacle,
    )
    return out[0]


__all__: List[str] = [
    "LidarSpec",
    "cast_rays",
    "cast_rays_single",
    "obstacles_to_arrays",
]
