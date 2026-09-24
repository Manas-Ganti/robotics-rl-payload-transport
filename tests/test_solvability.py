"""Path-existence checks on synthetic grids. Runs WITHOUT Isaac Sim.

The solvability guarantee (CLAUDE.md principle 3) is what separates "the policy
failed" from "the level was impossible". If these tests are wrong, every OOD
number in the study is suspect -- especially at high obstacle_density, where
unsolvable layouts are most likely and the OOD sweep spends most of its time.
"""

from __future__ import annotations

import numpy as np
import pytest

from env.solvability import (
    edge_mask,
    SolvabilityResult,
    UnsolvableLayoutError,
    astar,
    check_solvable,
    generate_until_solvable,
    grid_dim,
    grid_to_world,
    inflate_occupancy,
    path_exists,
    path_length_cells,
    radius_to_cells,
    world_to_grid,
)


def empty_grid(n: int = 10) -> np.ndarray:
    return np.zeros((n, n), dtype=bool)


def wall_grid(n: int = 10, gap: int | None = None) -> np.ndarray:
    """A grid with a full vertical wall at the middle column, optionally gapped."""
    grid = empty_grid(n)
    grid[:, n // 2] = True
    if gap is not None:
        grid[gap, n // 2] = False
    return grid


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------
class TestAStar:
    def test_finds_straight_path_on_empty_grid(self):
        path = astar(empty_grid(), (0, 0), (0, 5))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (0, 5)

    def test_returns_none_when_fully_walled_off(self):
        assert astar(wall_grid(), (0, 0), (0, 9)) is None

    def test_finds_path_through_a_gap(self):
        path = astar(wall_grid(gap=3), (0, 0), (0, 9))
        assert path is not None
        assert (3, 5) in path  # must route through the gap

    def test_rejects_blocked_start(self):
        grid = empty_grid()
        grid[0, 0] = True
        assert astar(grid, (0, 0), (5, 5)) is None

    def test_rejects_blocked_goal(self):
        grid = empty_grid()
        grid[5, 5] = True
        assert astar(grid, (0, 0), (5, 5)) is None

    def test_rejects_out_of_bounds(self):
        assert astar(empty_grid(), (0, 0), (99, 99)) is None

    def test_start_equals_goal(self):
        assert astar(empty_grid(), (3, 3), (3, 3)) == [(3, 3)]

    def test_diagonal_is_shorter_than_manhattan(self):
        """8-connectivity must actually use diagonals, or path efficiency
        denominators come out systematically too long and every policy looks
        more efficient than it is."""
        path_8 = astar(empty_grid(), (0, 0), (5, 5), connectivity=8)
        path_4 = astar(empty_grid(), (0, 0), (5, 5), connectivity=4)
        assert path_8 is not None and path_4 is not None
        assert len(path_8) < len(path_4)

    def test_does_not_cut_diagonal_corners(self):
        """A diagonal squeeze between two obstacles is not physically traversable
        by a robot with width; allowing it would certify impossible layouts."""
        grid = empty_grid(5)
        grid[1, 2] = True
        grid[2, 1] = True
        path = astar(grid, (1, 1), (2, 2), connectivity=8)
        # The direct diagonal is blocked, so either no path or a longer detour.
        assert path is None or len(path) > 2

    def test_invalid_connectivity_rejected(self):
        with pytest.raises(ValueError, match="connectivity"):
            astar(empty_grid(), (0, 0), (1, 1), connectivity=6)


# ---------------------------------------------------------------------------
# path_exists (BFS)
# ---------------------------------------------------------------------------
class TestPathExists:
    def test_agrees_with_astar_on_open_grid(self):
        assert path_exists(empty_grid(), (0, 0), (9, 9)) is True

    def test_agrees_with_astar_on_walled_grid(self):
        assert path_exists(wall_grid(), (0, 0), (0, 9)) is False

    def test_finds_gap(self):
        assert path_exists(wall_grid(gap=3), (0, 0), (0, 9)) is True


# ---------------------------------------------------------------------------
# Inflation
# ---------------------------------------------------------------------------
class TestInflation:
    def test_zero_radius_is_identity(self):
        grid = wall_grid()
        assert np.array_equal(inflate_occupancy(grid, 0), grid)

    def test_inflation_grows_obstacles(self):
        grid = empty_grid()
        grid[5, 5] = True
        inflated = inflate_occupancy(grid, 1)
        assert inflated[5, 5] and inflated[4, 5] and inflated[6, 5]
        assert inflated[5, 4] and inflated[5, 6]
        assert inflated.sum() > grid.sum()

    def test_inflation_is_circular_not_square(self):
        grid = empty_grid()
        grid[5, 5] = True
        inflated = inflate_occupancy(grid, 2)
        # A corner at Euclidean distance sqrt(8) > 2 must remain free.
        assert not inflated[3, 3]

    def test_inflation_clips_at_grid_edges(self):
        grid = empty_grid()
        grid[0, 0] = True
        inflate_occupancy(grid, 3)  # must not raise

    def test_radius_to_cells_ceils(self):
        """Ceiling, not rounding: under-inflating certifies paths the robot
        physically cannot follow."""
        assert radius_to_cells(0.35, 0.25) == 2
        assert radius_to_cells(0.5, 0.25) == 2
        assert radius_to_cells(0.51, 0.25) == 3

    def test_radius_to_cells_rejects_bad_resolution(self):
        with pytest.raises(ValueError):
            radius_to_cells(0.35, 0.0)


# ---------------------------------------------------------------------------
# check_solvable
# ---------------------------------------------------------------------------
class TestCheckSolvable:
    def test_open_grid_is_solvable(self):
        result = check_solvable(
            empty_grid(20), (0, 0), (19, 19), robot_radius_m=0.35, resolution_m=0.25
        )
        assert result.solvable
        assert result.path_length_m > 0
        assert result.reason == "ok"

    def test_walled_grid_is_unsolvable(self):
        result = check_solvable(
            wall_grid(20), (0, 0), (0, 19), robot_radius_m=0.35, resolution_m=0.25
        )
        assert not result.solvable
        assert result.reason == "no_path"

    def test_narrow_gap_blocked_by_robot_radius(self):
        """A 1-cell gap is passable by a point but not by a robot with radius --
        the whole reason inflation exists. Without it the harness would certify
        layouts the robot cannot actually traverse and score the resulting
        collisions against the policy."""
        grid = wall_grid(20, gap=10)
        as_point = check_solvable(
            grid, (0, 0), (0, 19), robot_radius_m=0.35, resolution_m=0.25,
            require_clearance=False,
        )
        with_radius = check_solvable(
            grid, (0, 0), (0, 19), robot_radius_m=0.35, resolution_m=0.25,
            require_clearance=True,
        )
        assert as_point.solvable
        assert not with_radius.solvable

    def test_reports_blocked_start_distinctly(self):
        """Start buried by inflation is a different failure than 'no path', and
        needs a different fix (move the endpoint, not thin the obstacles)."""
        grid = empty_grid(20)
        grid[1, 1] = True
        result = check_solvable(
            grid, (0, 0), (19, 19), robot_radius_m=0.5, resolution_m=0.25
        )
        assert not result.solvable
        assert result.reason == "start_blocked_after_inflation"

    def test_result_is_truthy(self):
        result = check_solvable(
            empty_grid(20), (0, 0), (19, 19), robot_radius_m=0.35, resolution_m=0.25
        )
        assert bool(result) is True
        assert bool(SolvabilityResult(False)) is False

    def test_path_length_scales_with_resolution(self):
        coarse = check_solvable(
            empty_grid(20), (0, 0), (0, 10), robot_radius_m=0.1, resolution_m=0.5
        )
        fine = check_solvable(
            empty_grid(20), (0, 0), (0, 10), robot_radius_m=0.1, resolution_m=0.25
        )
        assert coarse.path_length_m == pytest.approx(2 * fine.path_length_m)


# ---------------------------------------------------------------------------
# Regeneration loop
# ---------------------------------------------------------------------------
class TestGenerateUntilSolvable:
    def test_returns_first_solvable_candidate(self):
        def generator(attempt: int) -> int:
            return attempt

        def checker(candidate: int) -> SolvabilityResult:
            return SolvabilityResult(solvable=candidate >= 2, reason="no_path")

        candidate, result, attempts = generate_until_solvable(generator, checker, max_attempts=10)
        assert candidate == 2
        assert result.solvable
        assert attempts == 3

    def test_raises_when_budget_exhausted(self):
        """Must raise, never silently return the last unsolvable layout -- that
        would poison OOD results exactly where the study is most interesting."""
        def generator(attempt: int) -> int:
            return attempt

        def checker(candidate: int) -> SolvabilityResult:
            return SolvabilityResult(solvable=False, reason="no_path")

        with pytest.raises(UnsolvableLayoutError, match="No solvable layout"):
            generate_until_solvable(generator, checker, max_attempts=5)

    def test_error_message_reports_failure_reasons(self):
        def generator(attempt: int) -> int:
            return attempt

        def checker(candidate: int) -> SolvabilityResult:
            return SolvabilityResult(solvable=False, reason="start_blocked_after_inflation")

        with pytest.raises(UnsolvableLayoutError, match="start_blocked_after_inflation"):
            generate_until_solvable(generator, checker, max_attempts=3)

    def test_rejects_zero_budget(self):
        with pytest.raises(ValueError):
            generate_until_solvable(lambda i: i, lambda c: SolvabilityResult(True), max_attempts=0)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------
class TestCoordinateConversion:
    def test_round_trip_is_stable(self):
        """Cell -> world -> cell must be identity, or start/goal poses drift away
        from the cells the solvability check actually verified."""
        for cell in [(0, 0), (5, 7), (19, 19), (10, 3)]:
            xy = grid_to_world(cell, terrain_size_m=10.0, resolution_m=0.5)
            assert world_to_grid(xy, terrain_size_m=10.0, resolution_m=0.5) == cell

    def test_origin_maps_to_grid_center(self):
        cell = world_to_grid((0.0, 0.0), terrain_size_m=10.0, resolution_m=0.5)
        assert cell == (10, 10)

    def test_out_of_bounds_is_clamped(self):
        cell = world_to_grid((999.0, 999.0), terrain_size_m=10.0, resolution_m=0.5)
        assert cell == (19, 19)
        cell = world_to_grid((-999.0, -999.0), terrain_size_m=10.0, resolution_m=0.5)
        assert cell == (0, 0)

    def test_grid_dim(self):
        assert grid_dim(10.0, 0.25) == 40
        assert grid_dim(10.0, 0.5) == 20

    def test_grid_dim_rejects_bad_input(self):
        with pytest.raises(ValueError):
            grid_dim(0.0, 0.25)


class TestPathLength:
    def test_empty_and_single_point_paths_are_zero(self):
        assert path_length_cells([]) == 0.0
        assert path_length_cells([(0, 0)]) == 0.0

    def test_straight_line(self):
        assert path_length_cells([(0, 0), (0, 1), (0, 2)]) == pytest.approx(2.0)

    def test_diagonal_costs_sqrt_two(self):
        assert path_length_cells([(0, 0), (1, 1)]) == pytest.approx(np.sqrt(2))


# ---------------------------------------------------------------------------
# Patch edge: the whole robot must stay on the ground
# ---------------------------------------------------------------------------
class TestEdgeClearance:
    def test_edge_mask_band_width(self):
        """0.25 m cells, 0.55 m clearance: centres at 0.125 and 0.375 m from the
        edge are inside the band, 0.625 m is not -- 2 cells per side."""
        mask = edge_mask(40, 0.25, 0.55)
        assert mask[0, 20] and mask[1, 20] and not mask[2, 20]
        assert mask[20, 39] and mask[20, 38] and not mask[20, 37]
        assert not mask[20, 20]

    def test_zero_clearance_blocks_nothing(self):
        assert not edge_mask(40, 0.25, 0.0).any()

    def test_start_on_the_edge_is_rejected(self):
        result = check_solvable(
            empty_grid(40), (0, 20), (39, 20), robot_radius_m=0.1, resolution_m=0.25,
            edge_clearance_m=0.55,
        )
        assert not result.solvable and result.reason == "start_blocked_after_inflation"

    def test_default_keeps_old_behaviour(self):
        result = check_solvable(empty_grid(40), (0, 20), (39, 20), robot_radius_m=0.1, resolution_m=0.25)
        assert result.solvable

    def test_generated_episodes_keep_the_robot_on_the_patch(self):
        """End-to-end over the real config: every start, goal AND A* path cell is
        at least one robot radius from the patch edge (the Carter tail-down bug)."""
        from pathlib import Path

        from env.config import load_train_config
        from env.randomization import EpisodeParams
        from env.terrain_factory import TerrainFactory

        cfg = load_train_config(Path(__file__).resolve().parent.parent / "configs" / "train.yaml")
        factory = TerrainFactory(cfg)
        half = factory.size_m / 2.0
        rng = np.random.default_rng(1)
        for density in (0.2, 0.6, 0.9):
            params = EpisodeParams(
                obstacle_density=density, slope_angle_deg=0.0, friction_coeff=0.7,
                payload_mass_kg=25.0, depth_dropout_prob=0.0, depth_noise_std=0.0,
            )
            for _ in range(10):
                spec = factory.generate(params, rng)
                for x, y in (spec.start_xy, spec.goal_xy):
                    assert half - max(abs(x), abs(y)) >= factory.robot_radius_m - 1e-9
                for cell in spec.optimal_path_cells:
                    assert not factory.edge_band[cell], f"path cell {cell} inside the edge band"
