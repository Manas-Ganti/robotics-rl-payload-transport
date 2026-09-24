"""Procedural terrain generation: slope, scattered obstacles, friction.

PURE LOGIC -- no Isaac Sim import. This module produces a :class:`TerrainSpec`
(plain NumPy + dataclasses) describing what the scene should contain;
``env/nav_env.py`` is the only place that turns a spec into USD prims. That
split is what lets terrain generation and the solvability guarantee be unit
tested on a laptop with no GPU.

CLAUDE.md principle 1: parameterized, not hand-designed. There are no hand-built
levels here -- every layout is a deterministic function of (EpisodeParams, RNG).

CLAUDE.md principle 3: every spec returned by :meth:`TerrainFactory.generate`
has already passed the A* solvability check with robot-radius clearance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from env.randomization import EpisodeParams
from env.solvability import (
    Cell,
    SolvabilityResult,
    check_solvable,
    generate_until_solvable,
    grid_dim,
    grid_to_world,
    world_to_grid,
)


@dataclass(frozen=True)
class Obstacle:
    """One cylindrical obstacle, in world metres relative to the env origin.

    ``slot`` is the index of the simulator prim (``Obstacle_<slot>``) that
    realises this obstacle. Its radius is that slot's FIXED spawned radius (see
    :func:`pool_slot_radii`), so the sim, the occupancy grid / A* check, and the
    analytic lidar all agree on the obstacle's size. -1 = not bound to a slot
    (hand-built test fixtures only).
    """

    x: float
    y: float
    radius: float
    height: float
    slot: int = -1


def pool_slot_radii(radius_range_m: Tuple[float, float], num_slots: int) -> np.ndarray:
    """The fixed radius of each obstacle-pool prim: evenly spaced over the range.

    Why fixed per slot: Isaac Lab clones one prim graph across envs and PhysX
    cannot rescale a collision shape per env at runtime, so a prim's radius is
    set once at spawn. Rather than pretend otherwise (spawning one mean radius
    while the planner assumed sampled ones), terrain generation draws obstacles
    FROM the slots. Picking slots uniformly at random then gives radii uniform
    over this grid -- a discretised uniform over the configured range.

    ``env/nav_env.py`` spawns slot ``k`` with ``pool_slot_radii(...)[k]``; both
    sides call this one function, so they cannot disagree.
    """
    lo, hi = float(radius_range_m[0]), float(radius_range_m[1])
    if num_slots < 1:
        raise ValueError(f"num_slots must be >= 1, got {num_slots}")
    if not 0.0 < lo <= hi:
        raise ValueError(f"obstacle_radius_range_m must satisfy 0 < lo <= hi, got {radius_range_m}")
    if num_slots == 1:
        return np.array([(lo + hi) / 2.0])
    return np.linspace(lo, hi, num_slots)


@dataclass
class TerrainSpec:
    """Everything needed to instantiate one episode's scene.

    ``optimal_path_length_m`` comes from the A* check and is the denominator of
    the path-efficiency metric -- it is ground truth, computed mechanically, not
    an estimate (CLAUDE.md principle 4).
    """

    occupancy: np.ndarray            # (n, n) bool; True = occupied
    obstacles: List[Obstacle]
    start_xy: Tuple[float, float]
    goal_xy: Tuple[float, float]
    start_cell: Cell
    goal_cell: Cell
    params: EpisodeParams
    optimal_path_length_m: float
    optimal_path_cells: List[Cell] = field(default_factory=list)
    resolution_m: float = 0.25
    size_m: float = 10.0
    generation_attempts: int = 1

    # -- derived -------------------------------------------------------------
    @property
    def slope_angle_deg(self) -> float:
        return self.params.slope_angle_deg

    @property
    def friction_coeff(self) -> float:
        return self.params.friction_coeff

    @property
    def payload_mass_kg(self) -> float:
        return self.params.payload_mass_kg

    @property
    def straight_line_distance_m(self) -> float:
        return math.dist(self.start_xy, self.goal_xy)

    def height_at(self, x: float, y: float) -> float:
        """Ground height (m) at a world point, from the slope tilt.

        The patch is tilted about the y-axis, so +x is uphill. Travel along +x
        is a climb; this is why start/goal are placed across the gradient (see
        :meth:`TerrainFactory._sample_start_goal`).
        """
        return float(x * math.tan(math.radians(self.params.slope_angle_deg)))

    def to_log_dict(self) -> Dict[str, Any]:
        """Flat summary for CSV rows / W&B logging."""
        return {
            **self.params.as_dict(),
            "optimal_path_length_m": self.optimal_path_length_m,
            "straight_line_distance_m": self.straight_line_distance_m,
            "num_obstacles": len(self.obstacles),
            "occupied_fraction": float(self.occupancy.mean()),
            "generation_attempts": self.generation_attempts,
        }


class TerrainFactory:
    """Generates solvable, parameterized terrain patches from config.

    All tunables come from the ``terrain``, ``env``, and ``solvability`` config
    blocks -- there are no literals in the generation logic.
    """

    def __init__(self, cfg: Any) -> None:
        """
        Parameters
        ----------
        cfg
            A loaded train config (``env.config.Config``) or an equivalent
            mapping exposing the ``env``, ``terrain``, and ``solvability`` blocks.
        """
        data = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
        env_cfg = data["env"]
        terrain_cfg = data["terrain"]
        solv_cfg = data["solvability"]

        self.size_m: float = float(env_cfg["terrain_size_m"])
        self.resolution_m: float = float(env_cfg["grid_resolution_m"])
        self.min_start_goal_distance_m: float = float(env_cfg["min_start_goal_distance_m"])

        self.obstacle_radius_range: Tuple[float, float] = tuple(
            float(v) for v in terrain_cfg["obstacle_radius_range_m"]
        )  # type: ignore[assignment]
        self.obstacle_height_m: float = float(terrain_cfg["obstacle_height_m"])
        self.border_margin_m: float = float(terrain_cfg["border_margin_m"])
        self.start_goal_clearance_m: float = float(terrain_cfg["start_goal_clearance_m"])
        self.max_generation_attempts: int = int(terrain_cfg["max_generation_attempts"])
        self.max_obstacle_area_fraction: float = float(terrain_cfg["max_obstacle_area_fraction"])
        # Hard cap matching the simulator's fixed per-env obstacle pool. Enforced
        # HERE so the occupancy grid can never claim obstacles that nav_env.py
        # has no prim to spawn -- a mismatch would break the solvability
        # guarantee in the one direction that matters (sim easier than planned).
        self.max_obstacles_per_env: int = int(terrain_cfg["max_obstacles_per_env"])
        self.slot_radii: np.ndarray = pool_slot_radii(
            self.obstacle_radius_range, self.max_obstacles_per_env
        )

        self.robot_radius_m: float = float(solv_cfg["robot_radius_m"])
        self.connectivity: int = int(solv_cfg["connectivity"])
        self.require_clearance: bool = bool(solv_cfg["require_clearance"])

        self.n_cells: int = grid_dim(self.size_m, self.resolution_m)

        if self.min_start_goal_distance_m >= self.size_m * math.sqrt(2.0):
            raise ValueError(
                f"env.min_start_goal_distance_m ({self.min_start_goal_distance_m}) exceeds the "
                f"terrain diagonal ({self.size_m * math.sqrt(2.0):.2f} m) -- no valid start/goal "
                "pair can exist. Lower the distance or raise env.terrain_size_m."
            )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def generate(self, params: EpisodeParams, rng: np.random.Generator) -> TerrainSpec:
        """Generate a SOLVABLE terrain spec for the given episode parameters.

        Retries generation until the A* check passes (or the attempt budget is
        exhausted, which raises ``UnsolvableLayoutError``). Each attempt draws
        fresh values from ``rng``, so retries genuinely differ.
        """

        def build(attempt: int) -> Optional[TerrainSpec]:
            return self._build_candidate(params, rng)

        def check(candidate: Optional[TerrainSpec]) -> SolvabilityResult:
            if candidate is None:
                return SolvabilityResult(False, reason="start_goal_sampling_failed")
            return check_solvable(
                candidate.occupancy,
                candidate.start_cell,
                candidate.goal_cell,
                robot_radius_m=self.robot_radius_m,
                resolution_m=self.resolution_m,
                connectivity=self.connectivity,
                require_clearance=self.require_clearance,
            )

        spec, result, attempts = generate_until_solvable(
            build, check, max_attempts=self.max_generation_attempts
        )
        assert spec is not None  # guaranteed: a None candidate can never pass `check`

        spec.optimal_path_length_m = result.path_length_m
        spec.optimal_path_cells = result.path or []
        spec.generation_attempts = attempts
        return spec

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------
    def _build_candidate(
        self, params: EpisodeParams, rng: np.random.Generator
    ) -> Optional[TerrainSpec]:
        """One unverified candidate layout. May return None if sampling fails."""
        obstacles = self._sample_obstacles(params.obstacle_density, rng)
        occupancy = self._rasterize(obstacles)

        sampled = self._sample_start_goal(occupancy, params, rng)
        if sampled is None:
            return None
        start_cell, goal_cell = sampled

        # Carve clearance discs around start and goal so the robot can physically
        # spawn and so the goal is actually enterable. Done AFTER rasterizing so
        # the carve is authoritative over obstacle placement.
        occupancy = self._carve_clearance(occupancy, [start_cell, goal_cell])
        obstacles = self._drop_carved_obstacles(obstacles, [start_cell, goal_cell])

        return TerrainSpec(
            occupancy=occupancy,
            obstacles=obstacles,
            start_xy=grid_to_world(start_cell, terrain_size_m=self.size_m, resolution_m=self.resolution_m),
            goal_xy=grid_to_world(goal_cell, terrain_size_m=self.size_m, resolution_m=self.resolution_m),
            start_cell=start_cell,
            goal_cell=goal_cell,
            params=params,
            optimal_path_length_m=0.0,  # filled in by generate() from the A* result
            resolution_m=self.resolution_m,
            size_m=self.size_m,
        )

    def _sample_obstacles(self, density: float, rng: np.random.Generator) -> List[Obstacle]:
        """Scatter cylindrical obstacles to hit a target area fraction.

        ``obstacle_density`` in [0, 1] is a NORMALIZED knob, not a literal
        occupied-area fraction: it scales linearly up to
        ``terrain.max_obstacle_area_fraction``. A literal reading would make
        density=0.9 mean 90% of the floor covered, which is trivially
        unsolvable and would make the top of the OOD sweep measure nothing but
        the regeneration budget.
        """
        usable_half = self.size_m / 2.0 - self.border_margin_m
        if usable_half <= 0:
            raise ValueError(
                f"terrain.border_margin_m ({self.border_margin_m}) is too large for "
                f"env.terrain_size_m ({self.size_m}) -- no usable area remains."
            )

        target_area = (
            float(np.clip(density, 0.0, 1.0))
            * self.max_obstacle_area_fraction
            * (2.0 * usable_half) ** 2
        )

        obstacles: List[Obstacle] = []
        placed_area = 0.0

        # Draw pool slots in random order; each obstacle takes its slot's fixed
        # radius (see pool_slot_radii). Bounded by the pool size, so generation
        # can neither spin forever nor request a prim nav_env.py does not have.
        for slot in rng.permutation(self.max_obstacles_per_env):
            if placed_area >= target_area:
                break
            radius = float(self.slot_radii[slot])
            x = float(rng.uniform(-usable_half, usable_half))
            y = float(rng.uniform(-usable_half, usable_half))
            obstacles.append(
                Obstacle(x=x, y=y, radius=radius, height=self.obstacle_height_m, slot=int(slot))
            )
            placed_area += math.pi * radius**2

        if len(obstacles) >= self.max_obstacles_per_env and placed_area < target_area * 0.95:
            # The pool cap bound before the density target was met. High-density
            # OOD points would then be indistinguishable from lower ones, and the
            # obstacle_density curve would flatten at the top for a scene-budget
            # reason rather than a policy one. Warn loudly -- this is a silent
            # invalidator of the geometry axis.
            self._warn_pool_cap_bound(placed_area, target_area, density)

        return obstacles

    _pool_cap_warned = False

    def _warn_pool_cap_bound(self, placed_area: float, target_area: float, density: float) -> None:
        """Warn once per process that the obstacle pool cap is limiting density."""
        if TerrainFactory._pool_cap_warned:
            return
        TerrainFactory._pool_cap_warned = True

        import warnings

        warnings.warn(
            f"terrain.max_obstacles_per_env ({self.max_obstacles_per_env}) bound before the "
            f"obstacle_density={density:.2f} target was reached "
            f"({placed_area:.1f} of {target_area:.1f} m^2 placed). High-density points in the "
            "OOD sweep will be less dense than configured, flattening the geometry "
            "robustness curve for a scene-budget reason rather than a policy one. "
            "Raise max_obstacles_per_env or obstacle_radius_range_m.",
            stacklevel=3,
        )

    def _rasterize(self, obstacles: List[Obstacle]) -> np.ndarray:
        """Burn obstacles into a boolean occupancy grid.

        A cell is occupied if its CENTRE falls inside an obstacle disc. The
        robot-radius inflation in the solvability check supplies the safety
        margin, so no extra padding is applied here.
        """
        occupancy = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        if not obstacles:
            return occupancy

        half = self.size_m / 2.0
        centers = (np.arange(self.n_cells) + 0.5) * self.resolution_m - half
        xx, yy = np.meshgrid(centers, centers)  # xx varies along cols, yy along rows

        for obs in obstacles:
            occupancy |= ((xx - obs.x) ** 2 + (yy - obs.y) ** 2) <= obs.radius**2

        return occupancy

    def _sample_start_goal(
        self,
        occupancy: np.ndarray,
        params: EpisodeParams,
        rng: np.random.Generator,
        max_tries: int = 200,
    ) -> Optional[Tuple[Cell, Cell]]:
        """Pick free start/goal cells at least ``min_start_goal_distance_m`` apart.

        On sloped terrain the pair is oriented along the gradient (start downhill,
        goal uphill) so the episode actually exercises the slope axis. A pair
        placed across the contour would leave slope nearly irrelevant, and the
        Phase 2 curve would flatten for the wrong reason.
        """
        free = np.argwhere(~occupancy)
        if len(free) < 2:
            return None

        uphill = params.slope_angle_deg > 0.0
        min_dist_cells = self.min_start_goal_distance_m / self.resolution_m

        for _ in range(max_tries):
            a = free[rng.integers(len(free))]
            b = free[rng.integers(len(free))]
            if math.dist(a, b) < min_dist_cells:
                continue

            start, goal = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
            if uphill and goal[1] < start[1]:
                # Column index tracks +x, which is uphill. Swap so goal is uphill.
                start, goal = goal, start
            return start, goal

        return None

    def _carve_clearance(self, occupancy: np.ndarray, cells: List[Cell]) -> np.ndarray:
        """Clear a disc of radius ``start_goal_clearance_m`` around given cells."""
        out = occupancy.copy()
        radius_cells = int(math.ceil(self.start_goal_clearance_m / self.resolution_m))
        if radius_cells <= 0:
            return out

        for row, col in cells:
            r0, r1 = max(0, row - radius_cells), min(self.n_cells, row + radius_cells + 1)
            c0, c1 = max(0, col - radius_cells), min(self.n_cells, col + radius_cells + 1)
            rr, cc = np.mgrid[r0:r1, c0:c1]
            out[r0:r1, c0:c1] &= ((rr - row) ** 2 + (cc - col) ** 2) > radius_cells**2

        return out

    def _drop_carved_obstacles(self, obstacles: List[Obstacle], cells: List[Cell]) -> List[Obstacle]:
        """Remove obstacles overlapping a carved clearance disc.

        Keeps the obstacle LIST consistent with the occupancy GRID. If they
        diverge, the sim spawns a physical obstacle the planner believes is not
        there -- producing collisions the solvability check swore were
        impossible, and a wholly misleading collision rate.
        """
        keep: List[Obstacle] = []
        for obs in obstacles:
            overlaps = False
            for cell in cells:
                cx, cy = grid_to_world(cell, terrain_size_m=self.size_m, resolution_m=self.resolution_m)
                if math.hypot(obs.x - cx, obs.y - cy) <= (obs.radius + self.start_goal_clearance_m):
                    overlaps = True
                    break
            if not overlaps:
                keep.append(obs)
        return keep

    # -- convenience ---------------------------------------------------------
    def world_to_grid(self, xy: Tuple[float, float]) -> Cell:
        return world_to_grid(xy, terrain_size_m=self.size_m, resolution_m=self.resolution_m)

    def grid_to_world(self, cell: Cell) -> Tuple[float, float]:
        return grid_to_world(cell, terrain_size_m=self.size_m, resolution_m=self.resolution_m)
