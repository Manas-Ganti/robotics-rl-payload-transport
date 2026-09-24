"""Solvability guarantee: A* path-existence check + regenerate-on-fail.

PURE LOGIC -- no Isaac Sim import. Locally testable (tests/test_solvability.py).

CLAUDE.md principle 3: *every* generated episode must be verified to have a
valid path from start to goal before the robot spawns. Unsolvable layouts are
rejected and regenerated. Without this, "failure" conflates two totally
different things -- the policy being bad, and the level being impossible -- and
the OOD degradation curve becomes uninterpretable. This matters most at high
obstacle_density, which is exactly where the OOD sweep lives.

Grid convention
---------------
Occupancy grids are ``np.ndarray`` of dtype bool, shape ``(rows, cols)``, where
``True`` means OCCUPIED. Indices are ``(row, col)``. World coordinates are
metres with the terrain patch centred on the origin.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

Cell = Tuple[int, int]
T = TypeVar("T")

# Neighbour offsets. 8-connected includes diagonals; the diagonal cost is
# sqrt(2) so path lengths stay geometrically meaningful for path-efficiency.
_NEIGHBORS_4: Tuple[Tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
)

_SQRT2 = math.sqrt(2.0)

_NEIGHBORS_8: Tuple[Tuple[int, int, float], ...] = _NEIGHBORS_4 + (
    (-1, -1, _SQRT2),
    (-1, 1, _SQRT2),
    (1, -1, _SQRT2),
    (1, 1, _SQRT2),
)


class UnsolvableLayoutError(RuntimeError):
    """Raised when no solvable layout could be generated within the budget."""


@dataclass
class SolvabilityResult:
    """Outcome of a solvability check on one candidate layout."""

    solvable: bool
    path: Optional[List[Cell]] = None
    path_length_cells: float = 0.0
    path_length_m: float = 0.0
    reason: str = ""

    def __bool__(self) -> bool:
        return self.solvable


# ---------------------------------------------------------------------------
# World <-> grid conversion
# ---------------------------------------------------------------------------
def world_to_grid(
    xy: Sequence[float],
    *,
    terrain_size_m: float,
    resolution_m: float,
) -> Cell:
    """Convert a world (x, y) in metres to a ``(row, col)`` grid cell.

    The terrain patch is centred on the origin and spans
    ``[-terrain_size_m/2, +terrain_size_m/2]`` on both axes. Row indexes y,
    column indexes x. Results are clamped into bounds so a pose exactly on the
    boundary does not produce an out-of-range index.
    """
    n = grid_dim(terrain_size_m, resolution_m)
    half = terrain_size_m / 2.0
    col = int((float(xy[0]) + half) / resolution_m)
    row = int((float(xy[1]) + half) / resolution_m)
    return (int(np.clip(row, 0, n - 1)), int(np.clip(col, 0, n - 1)))


def grid_to_world(
    cell: Cell,
    *,
    terrain_size_m: float,
    resolution_m: float,
) -> Tuple[float, float]:
    """Convert a ``(row, col)`` cell to the world (x, y) of the CELL CENTRE."""
    half = terrain_size_m / 2.0
    x = (cell[1] + 0.5) * resolution_m - half
    y = (cell[0] + 0.5) * resolution_m - half
    return (x, y)


def grid_dim(terrain_size_m: float, resolution_m: float) -> int:
    """Number of cells per side for a square terrain patch."""
    if terrain_size_m <= 0 or resolution_m <= 0:
        raise ValueError(
            f"terrain_size_m and resolution_m must be > 0 "
            f"(got {terrain_size_m}, {resolution_m})"
        )
    return max(1, int(round(terrain_size_m / resolution_m)))


# ---------------------------------------------------------------------------
# Obstacle inflation (robot radius / clearance)
# ---------------------------------------------------------------------------
def inflate_occupancy(grid: np.ndarray, radius_cells: int) -> np.ndarray:
    """Dilate occupied cells by ``radius_cells`` using a circular structuring element.

    Planning on the inflated grid treats the robot as a point, which guarantees
    the returned path keeps at least ``radius_cells`` clearance from obstacles.
    A path found without inflation would be geometrically valid but physically
    impossible for a robot with real width.
    """
    if radius_cells <= 0:
        return grid.astype(bool, copy=True)

    r = int(radius_cells)
    size = 2 * r + 1
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = (yy**2 + xx**2) <= (r**2 + 1e-9)

    occupied = np.asarray(grid, dtype=bool)
    inflated = np.zeros_like(occupied)
    rows, cols = occupied.shape

    # Scatter each occupied cell's disk. Loops over obstacle cells only, which
    # is sparse for realistic densities -- fine at 40x40 grids, called once per
    # candidate layout rather than per step.
    for row, col in zip(*np.nonzero(occupied)):
        r0, r1 = max(0, row - r), min(rows, row + r + 1)
        c0, c1 = max(0, col - r), min(cols, col + r + 1)
        d_r0, d_c0 = r0 - (row - r), c0 - (col - r)
        sub = disk[d_r0 : d_r0 + (r1 - r0), d_c0 : d_c0 + (c1 - c0)]
        inflated[r0:r1, c0:c1] |= sub

    return inflated


def edge_mask(n_cells: int, resolution_m: float, clearance_m: float) -> np.ndarray:
    """Cells whose CENTRE is closer than ``clearance_m`` to the patch boundary.

    Obstacle inflation cannot see the patch edge -- it lies outside the grid --
    so without this a start, goal or path cell may sit right at the edge with
    half the robot hanging off the patch. With clearance = robot radius, the
    whole robot footprint stays on the ground. (ARC smoke test: Carter's rear
    caster, 0.43 m behind the axle, overhung the edge on ~11% of starts and
    the robot tipped tail-down.)
    """
    if clearance_m <= 0:
        return np.zeros((n_cells, n_cells), dtype=bool)
    idx = np.arange(n_cells)
    # distance from each cell centre to the nearest edge along each axis
    edge_dist = np.minimum(idx + 0.5, n_cells - idx - 0.5) * resolution_m
    return (edge_dist[:, None] < clearance_m) | (edge_dist[None, :] < clearance_m)


def radius_to_cells(radius_m: float, resolution_m: float) -> int:
    """Convert a physical clearance radius to a (ceiled) cell count.

    Ceiling rather than rounding: under-inflating produces paths the robot
    cannot physically follow, which is a worse failure than a slightly
    conservative plan.
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be > 0, got {resolution_m}")
    return int(math.ceil(max(0.0, radius_m) / resolution_m))


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------
def _neighbors(connectivity: int) -> Tuple[Tuple[int, int, float], ...]:
    if connectivity == 4:
        return _NEIGHBORS_4
    if connectivity == 8:
        return _NEIGHBORS_8
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


def _heuristic(a: Cell, b: Cell, connectivity: int) -> float:
    """Admissible heuristic matched to the connectivity.

    4-connected -> Manhattan. 8-connected -> octile distance. Using Euclidean
    for the 8-connected case would still be admissible but is looser, so octile
    expands fewer nodes.
    """
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    if connectivity == 4:
        return float(dr + dc)
    return float(max(dr, dc) + (_SQRT2 - 1.0) * min(dr, dc))


def astar(
    grid: np.ndarray,
    start: Cell,
    goal: Cell,
    *,
    connectivity: int = 8,
) -> Optional[List[Cell]]:
    """Shortest path from ``start`` to ``goal`` over free cells, or None.

    ``grid`` must ALREADY be inflated if clearance is required -- this function
    treats the robot as a point.

    Returns the full cell path including both endpoints.
    """
    occupied = np.asarray(grid, dtype=bool)
    rows, cols = occupied.shape

    def in_bounds(cell: Cell) -> bool:
        return 0 <= cell[0] < rows and 0 <= cell[1] < cols

    if not in_bounds(start) or not in_bounds(goal):
        return None
    if occupied[start] or occupied[goal]:
        return None
    if start == goal:
        return [start]

    neighbors = _neighbors(connectivity)
    open_heap: List[Tuple[float, float, Cell]] = [(_heuristic(start, goal, connectivity), 0.0, start)]
    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, float] = {start: 0.0}
    closed: set[Cell] = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct(came_from, current)
        closed.add(current)

        for d_row, d_col, cost in neighbors:
            neighbor = (current[0] + d_row, current[1] + d_col)
            if not in_bounds(neighbor) or occupied[neighbor] or neighbor in closed:
                continue

            # Forbid diagonal moves that cut a corner between two obstacles --
            # geometrically the robot would clip both.
            if d_row != 0 and d_col != 0:
                if occupied[current[0] + d_row, current[1]] and occupied[current[0], current[1] + d_col]:
                    continue

            tentative = g + cost
            if tentative < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                f = tentative + _heuristic(neighbor, goal, connectivity)
                heapq.heappush(open_heap, (f, tentative, neighbor))

    return None


def _reconstruct(came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_length_cells(path: Sequence[Cell]) -> float:
    """Geometric length of a cell path, in cell units (diagonals cost sqrt(2))."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def path_exists(
    grid: np.ndarray,
    start: Cell,
    goal: Cell,
    *,
    connectivity: int = 8,
) -> bool:
    """Cheap boolean path-existence check (BFS/flood fill, no cost ordering).

    Faster than A* when only reachability matters. :func:`check_solvable` uses
    A* instead because it needs the optimal path length for the path-efficiency
    metric.
    """
    occupied = np.asarray(grid, dtype=bool)
    rows, cols = occupied.shape

    def in_bounds(cell: Cell) -> bool:
        return 0 <= cell[0] < rows and 0 <= cell[1] < cols

    if not in_bounds(start) or not in_bounds(goal):
        return False
    if occupied[start] or occupied[goal]:
        return False
    if start == goal:
        return True

    neighbors = _neighbors(connectivity)
    seen = np.zeros_like(occupied)
    seen[start] = True
    frontier: List[Cell] = [start]

    while frontier:
        next_frontier: List[Cell] = []
        for cell in frontier:
            for d_row, d_col, _ in neighbors:
                neighbor = (cell[0] + d_row, cell[1] + d_col)
                if not in_bounds(neighbor) or occupied[neighbor] or seen[neighbor]:
                    continue
                if neighbor == goal:
                    return True
                seen[neighbor] = True
                next_frontier.append(neighbor)
        frontier = next_frontier

    return False


# ---------------------------------------------------------------------------
# The guarantee
# ---------------------------------------------------------------------------
def check_solvable(
    grid: np.ndarray,
    start: Cell,
    goal: Cell,
    *,
    robot_radius_m: float,
    resolution_m: float,
    connectivity: int = 8,
    require_clearance: bool = True,
    edge_clearance_m: float = 0.0,
) -> SolvabilityResult:
    """Verify a start->goal path exists with robot-radius clearance.

    ``edge_clearance_m`` > 0 also blocks the band along the patch boundary
    (see :func:`edge_mask`), which obstacle inflation cannot see.

    This is the function every generated episode must pass before spawning.
    The returned ``path_length_m`` is the OPTIMAL path length, which becomes the
    denominator of the path-efficiency metric in ``eval/metrics.py``.
    """
    occupied = np.asarray(grid, dtype=bool)

    planning_grid = occupied
    if require_clearance:
        radius_cells = radius_to_cells(robot_radius_m, resolution_m)
        planning_grid = inflate_occupancy(occupied, radius_cells)
    if edge_clearance_m > 0:
        planning_grid = planning_grid | edge_mask(occupied.shape[0], resolution_m, edge_clearance_m)

    # Inflation (or the edge band) can bury the endpoints themselves. That means
    # the start or goal is too close to an obstacle or the patch edge for the
    # robot to physically occupy, so the layout is genuinely invalid -- report
    # it distinctly rather than as a generic "no path", because the fix is
    # different (move the endpoints, not thin the obstacles).
    if require_clearance or edge_clearance_m > 0:
        if planning_grid[start]:
            return SolvabilityResult(False, reason="start_blocked_after_inflation")
        if planning_grid[goal]:
            return SolvabilityResult(False, reason="goal_blocked_after_inflation")

    path = astar(planning_grid, start, goal, connectivity=connectivity)
    if path is None:
        return SolvabilityResult(False, reason="no_path")

    length_cells = path_length_cells(path)
    return SolvabilityResult(
        solvable=True,
        path=path,
        path_length_cells=length_cells,
        path_length_m=length_cells * resolution_m,
        reason="ok",
    )


def generate_until_solvable(
    generator: Callable[[int], T],
    checker: Callable[[T], SolvabilityResult],
    *,
    max_attempts: int = 50,
) -> Tuple[T, SolvabilityResult, int]:
    """Regenerate layouts until one passes the solvability check.

    Parameters
    ----------
    generator
        ``attempt_index -> candidate layout``. Must produce a DIFFERENT layout
        per call (drive it from a seeded RNG that advances), otherwise this
        loop retries the same failing layout ``max_attempts`` times.
    checker
        ``candidate -> SolvabilityResult``.
    max_attempts
        Budget before giving up.

    Returns
    -------
    (layout, result, attempts_used)

    Raises
    ------
    UnsolvableLayoutError
        If the budget is exhausted. At high obstacle_density this is a real
        possibility, and it must be loud: silently falling back to the last
        (unsolvable) layout would poison the OOD results exactly where the
        study is most interesting.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    reasons: List[str] = []
    for attempt in range(max_attempts):
        candidate = generator(attempt)
        result = checker(candidate)
        if result.solvable:
            return candidate, result, attempt + 1
        reasons.append(result.reason)

    from collections import Counter

    raise UnsolvableLayoutError(
        f"No solvable layout after {max_attempts} attempts. "
        f"Failure reasons: {dict(Counter(reasons))}. "
        "At high obstacle_density this is expected -- either lower the density, "
        "raise terrain.max_generation_attempts, or shrink solvability.robot_radius_m."
    )


def iter_free_cells(grid: np.ndarray) -> Iterator[Cell]:
    """Yield every free ``(row, col)`` cell -- used for start/goal sampling."""
    for row, col in zip(*np.nonzero(~np.asarray(grid, dtype=bool))):
        yield (int(row), int(col))
