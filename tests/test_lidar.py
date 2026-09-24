"""Analytic lidar + obstacle-slot consistency. Runs WITHOUT Isaac Sim.

The lidar is the policy's only view of obstacles, so a geometry bug here would
not crash anything -- it would quietly train a policy on a wrong picture of the
world. These tests pin the geometry against closed-form cases AND against an
independent brute-force ray-march oracle, then check the torch path matches the
NumPy path bit-for-bit-ish (it is the same function, run on another backend).

The slot tests guard the other half of the contract: the radius the lidar and
the A* check use for an obstacle is the radius the simulator actually spawned.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from env.config import load_train_config
from env.lidar import LidarSpec, cast_rays, cast_rays_single, obstacles_to_arrays
from env.randomization import EpisodeParams
from env.spaces import build_observation_spec
from env.terrain_factory import Obstacle, TerrainFactory, pool_slot_radii

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CONFIG = REPO_ROOT / "configs" / "train.yaml"

MAX_RANGE = 8.0
HALF = 5.0


def spec(num_rays: int = 3, fov: float = 90.0, boundary: bool = False) -> LidarSpec:
    return LidarSpec(
        num_rays=num_rays,
        horizontal_fov_deg=fov,
        max_range_m=MAX_RANGE,
        boundary_as_obstacle=boundary,
        half_size_m=HALF,
    )


def obs(x: float, y: float, r: float, slot: int = -1) -> Obstacle:
    return Obstacle(x=x, y=y, radius=r, height=0.6, slot=slot)


CENTRE = 1  # index of the straight-ahead ray for spec(num_rays=3)


# ---------------------------------------------------------------------------
# Closed-form geometry
# ---------------------------------------------------------------------------
class TestSingleObstacle:
    def test_straight_ahead_hits_near_surface(self):
        r = cast_rays_single((0.0, 0.0), 0.0, [obs(3.0, 0.0, 0.5)], spec())
        assert r[CENTRE] == pytest.approx(2.5)

    def test_heading_rotates_the_scan(self):
        """Obstacle on +y is straight ahead once the robot faces +y."""
        r = cast_rays_single((0.0, 0.0), math.pi / 2, [obs(0.0, 3.0, 0.5)], spec())
        assert r[CENTRE] == pytest.approx(2.5)

    def test_off_centre_ray_hits_at_its_bearing(self):
        """+45 deg ray toward a disc centred on that bearing."""
        d = 3.0
        o = obs(d * math.cos(math.pi / 4), d * math.sin(math.pi / 4), 0.5)
        r = cast_rays_single((0.0, 0.0), 0.0, [o], spec())
        assert r[2] == pytest.approx(2.5)  # angles are [-45, 0, +45]
        assert r[CENTRE] == pytest.approx(MAX_RANGE)

    def test_obstacle_behind_is_invisible(self):
        r = cast_rays_single((0.0, 0.0), 0.0, [obs(-3.0, 0.0, 0.5)], spec(fov=90.0))
        assert np.all(r == pytest.approx(MAX_RANGE))

    def test_lateral_miss(self):
        r = cast_rays_single((0.0, 0.0), 0.0, [obs(3.0, 0.6, 0.5)], spec())
        assert r[CENTRE] == pytest.approx(MAX_RANGE)

    def test_tangent_ray_counts_as_hit(self):
        r = cast_rays_single((0.0, 0.0), 0.0, [obs(3.0, 0.5, 0.5)], spec())
        assert r[CENTRE] == pytest.approx(3.0, abs=1e-6)

    def test_origin_inside_obstacle_reads_zero(self):
        """Robot overlapping an obstacle is already in contact."""
        r = cast_rays_single((0.0, 0.0), 0.0, [obs(0.1, 0.0, 0.5)], spec())
        assert np.all(r == 0.0)

    def test_nearest_of_two_wins(self):
        r = cast_rays_single(
            (0.0, 0.0), 0.0, [obs(6.0, 0.0, 0.5), obs(3.0, 0.0, 0.5)], spec()
        )
        assert r[CENTRE] == pytest.approx(2.5)

    def test_beyond_max_range_clips(self):
        r = cast_rays_single((0.0, 0.0), 0.0, [obs(20.0, 0.0, 0.5)], spec())
        assert r[CENTRE] == pytest.approx(MAX_RANGE)

    def test_no_obstacles(self):
        r = cast_rays_single((0.0, 0.0), 0.0, [], spec())
        assert np.all(r == pytest.approx(MAX_RANGE))


class TestBoundary:
    def test_straight_to_wall(self):
        r = cast_rays_single((0.0, 0.0), 0.0, [], spec(boundary=True))
        assert r[CENTRE] == pytest.approx(HALF)

    def test_diagonal_to_corner(self):
        r = cast_rays_single((0.0, 0.0), 0.0, [], spec(boundary=True))
        assert r[2] == pytest.approx(HALF * math.sqrt(2.0))

    def test_obstacle_nearer_than_wall(self):
        r = cast_rays_single((0.0, 0.0), 0.0, [obs(2.0, 0.0, 0.5)], spec(boundary=True))
        assert r[CENTRE] == pytest.approx(1.5)

    def test_outside_patch_reads_zero(self):
        r = cast_rays_single((6.0, 0.0), 0.0, [], spec(boundary=True))
        assert np.all(r == 0.0)

    def test_disabled_boundary_is_open(self):
        r = cast_rays_single((0.0, 0.0), 0.0, [], spec(boundary=False))
        assert np.all(r == pytest.approx(MAX_RANGE))


# ---------------------------------------------------------------------------
# Independent oracle: brute-force ray marching
# ---------------------------------------------------------------------------
def _march(origin, yaw, angles, obstacles, *, boundary, step=0.002):
    """First distance along each ray that is inside a disc or outside the box."""
    t = np.arange(0.0, MAX_RANGE + step, step)
    out = np.full(len(angles), MAX_RANGE)
    for i, a in enumerate(angles):
        px = origin[0] + t * math.cos(yaw + a)
        py = origin[1] + t * math.sin(yaw + a)
        blocked = np.zeros_like(t, dtype=bool)
        for o in obstacles:
            blocked |= (px - o.x) ** 2 + (py - o.y) ** 2 <= o.radius**2
        if boundary:
            blocked |= (np.abs(px) > HALF) | (np.abs(py) > HALF)
        idx = np.flatnonzero(blocked)
        if idx.size:
            out[i] = t[idx[0]]
    return out


@pytest.mark.parametrize("seed", range(6))
def test_matches_brute_force_oracle(seed):
    rng = np.random.default_rng(seed)
    obstacles = [
        obs(float(rng.uniform(-4.5, 4.5)), float(rng.uniform(-4.5, 4.5)), float(rng.uniform(0.3, 0.7)))
        for _ in range(12)
    ]
    # Start from a point outside every disc (inside-disc is the special case above).
    while True:
        origin = (float(rng.uniform(-4.0, 4.0)), float(rng.uniform(-4.0, 4.0)))
        if all(math.hypot(origin[0] - o.x, origin[1] - o.y) > o.radius + 0.05 for o in obstacles):
            break
    yaw = float(rng.uniform(-math.pi, math.pi))
    sp = spec(num_rays=64, fov=180.0, boundary=True)

    got = cast_rays_single(origin, yaw, obstacles, sp)
    want = _march(origin, yaw, sp.ray_angles(), obstacles, boundary=True)
    # March resolution is 2 mm; allow one step of slack.
    np.testing.assert_allclose(got, want, atol=0.005)


def test_batch_equals_individual_calls():
    rng = np.random.default_rng(7)
    sp = spec(num_rays=16, fov=180.0, boundary=True)
    scenes = [
        [obs(float(rng.uniform(-4, 4)), float(rng.uniform(-4, 4)), 0.5, slot=k) for k in range(5)]
        for _ in range(4)
    ]
    origins = np.array([[0.0, 0.0], [1.0, -1.0], [-2.0, 2.0], [3.0, 0.5]])
    origins = np.array(
        [o if all(math.hypot(o[0] - b.x, o[1] - b.y) > 0.6 for b in sc) else [4.8, 4.8]
         for o, sc in zip(origins, scenes)]
    )
    yaws = np.array([0.0, 1.0, -2.0, 3.0])

    xyr, valid = obstacles_to_arrays(scenes, 8)
    batch = cast_rays(
        np, origins, yaws, sp.ray_angles(), xyr.astype(np.float64), valid,
        max_range_m=MAX_RANGE, half_size_m=HALF, boundary_as_obstacle=True,
    )
    for i in range(4):
        np.testing.assert_allclose(
            batch[i], cast_rays_single(tuple(origins[i]), yaws[i], scenes[i], sp), atol=1e-9
        )


def test_invalid_rows_are_ignored():
    xyr = np.array([[[3.0, 0.0, 0.5]]])
    valid = np.array([[False]])
    r = cast_rays(
        np, np.zeros((1, 2)), np.zeros(1), spec().ray_angles(), xyr, valid,
        max_range_m=MAX_RANGE, half_size_m=HALF, boundary_as_obstacle=False,
    )
    assert np.all(r == pytest.approx(MAX_RANGE))


def test_torch_backend_matches_numpy():
    """Same function, other backend: this is the code path that runs on the GPU."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    n, k = 32, 10
    sp = spec(num_rays=64, fov=180.0, boundary=True)
    xyr = np.stack(
        [rng.uniform(-4.5, 4.5, (n, k)), rng.uniform(-4.5, 4.5, (n, k)), rng.uniform(0.3, 0.7, (n, k))],
        axis=-1,
    ).astype(np.float32)
    valid = rng.random((n, k)) < 0.7
    origins = rng.uniform(-4.0, 4.0, (n, 2)).astype(np.float32)
    yaws = rng.uniform(-math.pi, math.pi, n).astype(np.float32)
    angles = sp.ray_angles().astype(np.float32)
    kw = dict(max_range_m=MAX_RANGE, half_size_m=HALF, boundary_as_obstacle=True)

    ref = cast_rays(np, origins, yaws, angles, xyr, valid, **kw)
    got = cast_rays(
        torch, torch.from_numpy(origins), torch.from_numpy(yaws), torch.from_numpy(angles),
        torch.from_numpy(xyr), torch.from_numpy(valid), **kw,
    ).numpy()
    np.testing.assert_allclose(got, ref, atol=1e-4)


# ---------------------------------------------------------------------------
# Spec / config
# ---------------------------------------------------------------------------
class TestSpec:
    def test_angles_span_fov_symmetrically(self):
        a = spec(num_rays=64, fov=180.0).ray_angles()
        assert len(a) == 64
        assert a[0] == pytest.approx(-math.pi / 2) and a[-1] == pytest.approx(math.pi / 2)
        np.testing.assert_allclose(a, -a[::-1])

    def test_full_circle_has_no_duplicate_ray(self):
        a = spec(num_rays=8, fov=360.0).ray_angles()
        assert len(a) == 8
        assert not math.isclose((a[-1] - a[0]) % (2 * math.pi), 0.0, abs_tol=1e-9)

    @pytest.mark.parametrize(
        "kwargs", [dict(num_rays=0), dict(horizontal_fov_deg=0.0), dict(max_range_m=0.0)]
    )
    def test_validate_rejects_bad_values(self, kwargs):
        base = dict(num_rays=3, horizontal_fov_deg=90.0, max_range_m=8.0,
                    boundary_as_obstacle=False, half_size_m=5.0)
        with pytest.raises(ValueError):
            LidarSpec(**{**base, **kwargs}).validate()

    def test_real_config_is_analytic_and_matches_obs_layout(self):
        cfg = load_train_config(TRAIN_CONFIG)
        assert cfg.to_dict()["sensors"]["modality"] == "analytic_lidar"
        sp = LidarSpec.from_config(cfg)
        obs_spec = build_observation_spec(cfg)
        depth = obs_spec.slice_of("depth")
        assert depth.stop - depth.start == sp.num_rays
        assert sp.half_size_m == cfg.to_dict()["env"]["terrain_size_m"] / 2.0


# ---------------------------------------------------------------------------
# Obstacle slots: sim radius == planner radius == lidar radius
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def factory_and_specs():
    cfg = load_train_config(TRAIN_CONFIG)
    factory = TerrainFactory(cfg)
    rng = np.random.default_rng(0)
    specs = []
    for density in (0.2, 0.6, 0.9):  # includes the OOD top end
        params = EpisodeParams(
            obstacle_density=density, slope_angle_deg=0.0, friction_coeff=0.7,
            payload_mass_kg=3.0, depth_dropout_prob=0.0, depth_noise_std=0.0,
        )
        specs += [factory.generate(params, rng) for _ in range(5)]
    return factory, specs


class TestObstacleSlots:
    def test_pool_radii_span_the_range(self):
        r = pool_slot_radii((0.3, 0.7), 48)
        assert len(r) == 48
        assert r[0] == pytest.approx(0.3) and r[-1] == pytest.approx(0.7)
        assert np.all(np.diff(r) > 0)

    @pytest.mark.parametrize("bad", [(0.0, 0.5), (0.7, 0.3), (-0.1, 0.5)])
    def test_pool_radii_reject_bad_range(self, bad):
        with pytest.raises(ValueError):
            pool_slot_radii(bad, 10)

    def test_arrays_place_obstacles_at_their_slot(self):
        xyr, valid = obstacles_to_arrays([[obs(1.0, 2.0, 0.4, slot=5)]], 8)
        assert valid[0].tolist() == [False] * 5 + [True] + [False] * 2
        np.testing.assert_allclose(xyr[0, 5], [1.0, 2.0, 0.4])

    def test_arrays_reject_duplicate_or_out_of_pool_slots(self):
        with pytest.raises(ValueError, match="two obstacles"):
            obstacles_to_arrays([[obs(0, 0, 0.4, slot=1), obs(1, 1, 0.4, slot=1)]], 4)
        with pytest.raises(ValueError, match="exceeds"):
            obstacles_to_arrays([[obs(0, 0, 0.4, slot=4)]], 4)

    def test_generated_obstacles_use_their_slot_radius(self, factory_and_specs):
        factory, specs = factory_and_specs
        for s in specs:
            slots = [o.slot for o in s.obstacles]
            assert all(0 <= k < factory.max_obstacles_per_env for k in slots)
            assert len(set(slots)) == len(slots), "two obstacles bound to one prim"
            for o in s.obstacles:
                assert o.radius == pytest.approx(factory.slot_radii[o.slot])

    def test_start_is_clear_to_the_lidar(self, factory_and_specs):
        """Factory + lidar agree: the carved start clearance is visible as free
        space in every direction (boundary off -- the start may be near an edge)."""
        factory, specs = factory_and_specs
        sp = spec(num_rays=64, fov=360.0, boundary=False)
        for s in specs:
            ranges = cast_rays_single(s.start_xy, 0.0, s.obstacles, sp)
            assert ranges.min() >= factory.start_goal_clearance_m - 1e-6
