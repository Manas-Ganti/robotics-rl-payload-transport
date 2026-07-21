"""Domain-randomization sampler -- PURE LOGIC, no Isaac Sim import.

Reads the INNER (training) ranges from config and samples per-episode parameter
sets. The same :class:`EpisodeParams` structure is produced for evaluation, but
by :func:`params_from_grid_point` instead of by random sampling, so training and
evaluation drive the environment through exactly one code path.

NOTE: gravity is never randomized here (CLAUDE.md principle 7). Payload mass is
the dynamics axis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from env.config import STUDY_AXES, ConfigError, Range


@dataclass(frozen=True)
class EpisodeParams:
    """The sampled physical parameters for a single episode.

    One instance fully determines the episode's terrain, physics, and payload,
    which is what makes a run reproducible from (config, seed) alone.
    """

    obstacle_density: float
    slope_angle_deg: float
    friction_coeff: float
    payload_mass_kg: float
    depth_dropout_prob: float
    depth_noise_std: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def with_overrides(self, **overrides: float) -> "EpisodeParams":
        """Return a copy with specific axes replaced (used by the OOD sweep)."""
        unknown = set(overrides) - set(STUDY_AXES)
        if unknown:
            raise ConfigError(f"Unknown study axes in override: {sorted(unknown)}")
        return replace(self, **overrides)


class DomainSampler:
    """Seeded uniform sampler over the training (inner) ranges.

    Phase flags gate individual axes: when ``enable_slope`` is False the slope
    axis is pinned to 0 regardless of its configured range, so enabling Phase 2
    is a one-line config change rather than a range edit that would then need
    reverting.
    """

    def __init__(
        self,
        ranges: Mapping[str, Range],
        *,
        seed: int,
        enable_slope: bool = False,
        enable_sensor_noise: bool = False,
    ) -> None:
        missing = sorted(set(STUDY_AXES) - set(ranges))
        if missing:
            raise ConfigError(f"DomainSampler is missing ranges for axes: {missing}")

        self._ranges = dict(ranges)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._enable_slope = enable_slope
        self._enable_sensor_noise = enable_sensor_noise

    # -- construction --------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Any, *, seed: Optional[int] = None) -> "DomainSampler":
        """Build from a loaded train config (``env.config.Config``)."""
        from env.config import get_train_ranges

        data = cfg.to_dict()
        phases = data.get("phases", {})
        return cls(
            get_train_ranges(cfg),
            seed=int(seed if seed is not None else data["seed"]),
            enable_slope=bool(phases.get("enable_slope", False)),
            enable_sensor_noise=bool(phases.get("enable_sensor_noise", False)),
        )

    # -- sampling ------------------------------------------------------------
    def _uniform(self, axis: str) -> float:
        rng_range = self._ranges[axis]
        return float(self._rng.uniform(rng_range.low, rng_range.high))

    def sample(self) -> EpisodeParams:
        """Sample one episode's parameters from the training ranges."""
        return EpisodeParams(
            obstacle_density=self._uniform("obstacle_density"),
            slope_angle_deg=self._uniform("slope_angle_deg") if self._enable_slope else 0.0,
            friction_coeff=self._uniform("friction_coeff"),
            payload_mass_kg=self._uniform("payload_mass_kg"),
            depth_dropout_prob=self._uniform("depth_dropout_prob") if self._enable_sensor_noise else 0.0,
            depth_noise_std=self._uniform("depth_noise_std") if self._enable_sensor_noise else 0.0,
        )

    def sample_batch(self, n: int) -> list[EpisodeParams]:
        """Sample ``n`` independent parameter sets (one per parallel env)."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        return [self.sample() for _ in range(n)]

    def spawn_child(self, offset: int) -> "DomainSampler":
        """Derive an independent sampler with a deterministic offset seed.

        Used to give each parallel environment its own reproducible stream
        without them sharing (and therefore correlating) a single RNG.
        """
        return DomainSampler(
            self._ranges,
            seed=self._seed + offset,
            enable_slope=self._enable_slope,
            enable_sensor_noise=self._enable_sensor_noise,
        )

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def rng(self) -> np.random.Generator:
        """The underlying generator -- shared with terrain generation so a whole
        episode (params + layout) derives from one reproducible stream."""
        return self._rng


# ---------------------------------------------------------------------------
# Evaluation-side construction (no randomness on the swept axis)
# ---------------------------------------------------------------------------
def params_from_nominal(nominal: Mapping[str, float]) -> EpisodeParams:
    """Build an :class:`EpisodeParams` from the eval config's nominal values."""
    missing = sorted(set(STUDY_AXES) - set(nominal))
    if missing:
        raise ConfigError(f"nominal block is missing axes: {missing}")
    return EpisodeParams(**{axis: float(nominal[axis]) for axis in STUDY_AXES})


def params_from_grid_point(
    nominal: Mapping[str, float],
    *,
    axis: str,
    value: float,
) -> EpisodeParams:
    """Pin every axis to nominal, then set ``axis`` to the swept ``value``.

    This is what makes single-axis attribution valid: exactly one parameter
    departs from the in-distribution nominal, so any measured degradation is
    caused by that axis and not by a confounding off-axis shift.
    """
    if axis not in STUDY_AXES:
        raise ConfigError(f"Unknown study axis '{axis}'. Valid: {list(STUDY_AXES)}")
    return params_from_nominal(nominal).with_overrides(**{axis: float(value)})


# ---------------------------------------------------------------------------
# Sensor degradation (applied to observations, not to the physics scene)
# ---------------------------------------------------------------------------
def apply_depth_degradation(
    depth: np.ndarray,
    *,
    dropout_prob: float,
    noise_std: float,
    max_range_m: float,
    rng: np.random.Generator,
    fill_value: str = "max_range",
) -> np.ndarray:
    """Apply dropout + additive Gaussian noise to a depth/range reading.

    Pure and NumPy-only so the degradation model is unit-testable without a
    simulator. ``env/nav_env.py`` mirrors this on GPU tensors -- see
    ``_apply_depth_degradation_torch`` there, and keep the two in sync.

    Parameters
    ----------
    depth
        Range readings in metres, any shape.
    dropout_prob
        Per-element probability the reading is lost entirely.
    noise_std
        Std-dev of additive Gaussian noise, in metres.
    fill_value
        What a dropped reading becomes: ``"max_range"`` (sensor saw nothing),
        ``"zero"`` (sensor reports contact -- pessimistic), or ``"last_valid"``
        (hold previous; approximated here as max_range since this function is
        stateless).
    """
    out = np.array(depth, dtype=np.float64, copy=True)

    if noise_std > 0:
        out = out + rng.normal(0.0, noise_std, size=out.shape)

    if dropout_prob > 0:
        mask = rng.random(out.shape) < dropout_prob
        if fill_value == "zero":
            out[mask] = 0.0
        else:  # "max_range" and the stateless "last_valid" approximation
            out[mask] = max_range_m

    return np.clip(out, 0.0, max_range_m)


def resolve_physics_material(params: EpisodeParams) -> Dict[str, float]:
    """Map sampled params to PhysX material properties.

    Static friction is set slightly above dynamic to avoid the unphysical
    stick-slip chatter PhysX produces when the two are exactly equal -- a real
    source of spurious "the robot vibrates in place" episodes.

    VERIFY ON A100: confirm restitution 0.0 is appropriate for the chosen robot;
    a bouncy chassis will corrupt collision-based terminations.
    """
    return {
        "static_friction": float(params.friction_coeff),
        "dynamic_friction": float(params.friction_coeff * 0.9),
        "restitution": 0.0,
    }


def summarize_params(params: Sequence[EpisodeParams]) -> Dict[str, float]:
    """Mean of each axis across a batch -- for W&B logging of the actual
    realized distribution (which is worth checking against the configured one)."""
    if not params:
        return {}
    stacked = {axis: np.array([getattr(p, axis) for p in params]) for axis in STUDY_AXES}
    return {f"domain/{axis}_mean": float(values.mean()) for axis, values in stacked.items()}
