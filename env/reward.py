"""Reward computation -- PURE LOGIC, no Isaac Sim import.

Every term is individually toggleable and individually weighted from YAML
(``reward.terms.<name>.enabled`` / ``.weight``) so reward ablations on the A100
never require a code edit.

Backend-agnostic: all functions operate elementwise on scalars, NumPy arrays, or
torch tensors. Isaac Lab hands us torch tensors of shape ``(num_envs,)``; the
local tests pass plain floats and NumPy arrays. The tiny dispatch helpers at the
top are the only place that difference is handled.

SIGN CONVENTION (important)
---------------------------
Config weights are always NON-NEGATIVE magnitudes; validation in
``env/config.py`` rejects negative weights. The sign of a penalty is applied
HERE, in code. This means a typo in YAML can change a penalty's magnitude but
can never silently flip it into a bonus -- which is the failure mode that
quietly teaches a policy to seek collisions.


=============================================================================
REWARD-DESIGN FAILURE HISTORY
=============================================================================
Fill this in as failure modes are observed on the A100. Each entry should
record: what the policy actually did, which term caused it, and the fix. This
history is the most valuable artifact of reward iteration -- future-you will
otherwise re-introduce a fix that was already tried and reverted.

Template:

  ### [DATE] <short name of the pathology>
  - **Observed:**   what the policy did (with the metric that revealed it)
  - **Diagnosis:**  which term/weight produced the incentive
  - **Fix:**        config change made (term, old weight -> new weight)
  - **Outcome:**    did it work; any new pathology introduced

Known pathologies to watch for in THIS task specifically:

  - **Weak terminal signal.** If `progress_shaping.weight` is large relative to
    `goal_reached`, the journey pays more than the arrival and the policy learns
    to approach the goal without committing to entering it. Watch for episodes
    with high return but low success rate and a small final distance-to-goal.
    Invariant checked by `warn_on_degenerate_weights`:
        goal_reached.weight >= 0.5 * progress_shaping.weight * map_diagonal_m
    (Note: oscillating to farm shaping is NOT a risk here, because the term is
    symmetric -- retreating costs exactly what approaching pays. The
    `clip_per_step` guard exists for teleport/reset spikes, not for farming.)

  - **Collision-as-escape.** If `step_cost` accumulates faster than the
    `collision` penalty hurts, deliberately crashing early becomes cheaper than
    a long episode. Invariant to preserve:
        collision.weight  >  step_cost.weight * (episode_length_s / policy_dt)
    Checked at construction time by `RewardConfig.warn_on_degenerate_weights`.

  - **Stall-in-place.** With `step_cost` too small, timing out costs little and
    the policy learns to freeze near obstacles rather than risk a collision.
    Shows up as timeout_rate climbing while collision_rate falls.

  - **Payload-blind smoothness (v2).** Over-weighting `jerk`/`action_rate` makes
    the policy so conservative it times out on sloped terrain. v2 terms should
    stay an order of magnitude below the v1 terms unless ablation says otherwise.

  (no entries yet -- populate from A100 runs)
=============================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, TypeVar, Union

import numpy as np

Array = TypeVar("Array", bound=Union[float, np.ndarray])

# ---------------------------------------------------------------------------
# Backend dispatch: works for float / np.ndarray / torch.Tensor without
# importing torch (which is absent on the local GPU-less dev machine).
# ---------------------------------------------------------------------------


def _clip(x: Any, lo: float, hi: float) -> Any:
    """Elementwise clip for floats, NumPy arrays, and torch tensors."""
    if hasattr(x, "clamp"):  # torch.Tensor
        return x.clamp(lo, hi)
    return np.clip(x, lo, hi)


def _abs(x: Any) -> Any:
    """Elementwise absolute value."""
    if hasattr(x, "abs"):  # torch.Tensor
        return x.abs()
    return np.abs(x)


def _sum_last(x: Any) -> Any:
    """Sum over the final (action-dim) axis, collapsing per-action values to per-env.

    The last axis of an action is ALWAYS the action dimension -- ``(A,)`` for a
    single env, ``(N, A)`` batched -- so it is always reduced. (A previous
    ``ndim > 1`` guard left a single env's ``(A,)`` unsummed, which then
    broadcast the whole reward to shape ``(A,)``.) Scalars pass through.
    """
    if np.isscalar(x):
        return x
    if hasattr(x, "sum") and hasattr(x, "dim"):  # torch.Tensor
        return x.sum(dim=-1) if x.dim() >= 1 else x
    arr = np.asarray(x)
    return arr.sum(axis=-1) if arr.ndim >= 1 else arr


def _as_float(x: Any) -> Any:
    """Cast booleans/ints to float, preserving the backend type."""
    if hasattr(x, "float"):  # torch.Tensor
        return x.float()
    if isinstance(x, (bool, np.bool_)):
        return float(x)
    return np.asarray(x, dtype=np.float64) if isinstance(x, np.ndarray) else float(x)


def _zeros_like(x: Any) -> Any:
    """A zero of the same shape/backend as ``x``."""
    if hasattr(x, "zeros_like"):
        return x.zeros_like()
    if isinstance(x, np.ndarray):
        return np.zeros_like(x, dtype=np.float64)
    return 0.0


# ---------------------------------------------------------------------------
# Term specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TermSpec:
    """One reward term's toggle + weight, straight from YAML."""

    enabled: bool
    weight: float

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(
                f"Reward weights must be non-negative magnitudes (got {self.weight}). "
                "Penalty signs are applied in env/reward.py, not in config."
            )

    @property
    def active_weight(self) -> float:
        """Weight if enabled, else exactly 0.0 -- lets terms be summed blindly."""
        return self.weight if self.enabled else 0.0


@dataclass
class RewardConfig:
    """Fully-resolved reward configuration.

    Built from the ``reward`` block of ``configs/train.yaml`` via
    :meth:`from_config`, which also applies the ``enable_reward_v2`` phase gate.
    """

    goal_reached: TermSpec
    collision: TermSpec
    step_cost: TermSpec
    progress_shaping: TermSpec
    action_rate: TermSpec = field(default_factory=lambda: TermSpec(False, 0.0))
    jerk: TermSpec = field(default_factory=lambda: TermSpec(False, 0.0))
    energy_under_load: TermSpec = field(default_factory=lambda: TermSpec(False, 0.0))

    progress_clip_per_step: float = 1.0
    payload_mass_norm_kg: float = 5.0
    version: str = "v1"

    # -- construction --------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        reward_cfg: Mapping[str, Any],
        *,
        enable_reward_v2: bool = False,
    ) -> "RewardConfig":
        """Build from the YAML ``reward`` block.

        ``enable_reward_v2`` is the phase flag from ``phases.enable_reward_v2``.
        When it is False, the v2 terms are force-disabled regardless of their
        individual ``enabled`` flags -- so a single phase flag cleanly controls
        the reward tier without hand-editing three separate toggles.
        """
        terms = reward_cfg["terms"]

        def spec(name: str, *, v2: bool = False) -> TermSpec:
            if name not in terms:
                return TermSpec(False, 0.0)
            raw = terms[name]
            enabled = bool(raw["enabled"]) and (enable_reward_v2 or not v2)
            return TermSpec(enabled=enabled, weight=float(raw["weight"]))

        version = str(reward_cfg.get("version", "v1"))
        if version == "v2" and not enable_reward_v2:
            warnings.warn(
                "reward.version is 'v2' but phases.enable_reward_v2 is False; "
                "v2 terms stay disabled. Set the phase flag to activate them.",
                stacklevel=2,
            )

        progress_clip = float(terms.get("progress_shaping", {}).get("clip_per_step", 1.0))

        cfg = cls(
            goal_reached=spec("goal_reached"),
            collision=spec("collision"),
            step_cost=spec("step_cost"),
            progress_shaping=spec("progress_shaping"),
            action_rate=spec("action_rate", v2=True),
            jerk=spec("jerk", v2=True),
            energy_under_load=spec("energy_under_load", v2=True),
            progress_clip_per_step=progress_clip,
            payload_mass_norm_kg=float(reward_cfg.get("payload_mass_norm_kg", 5.0)),
            version=version,
        )
        return cfg

    # -- sanity checks -------------------------------------------------------
    def warn_on_degenerate_weights(
        self,
        *,
        episode_length_s: float,
        policy_dt: float,
        max_goal_distance_m: float | None = None,
    ) -> None:
        """Warn about weight combinations with known pathological incentives.

        Deliberately a warning, not an error: an ablation may intentionally
        explore a degenerate corner. The point is that it is never accidental.
        """
        if policy_dt <= 0:
            raise ValueError(f"policy_dt must be > 0, got {policy_dt}")
        max_steps = episode_length_s / policy_dt

        if self.collision.enabled and self.step_cost.enabled:
            worst_case_step_cost = self.step_cost.weight * max_steps
            if self.collision.weight <= worst_case_step_cost:
                warnings.warn(
                    "COLLISION-AS-ESCAPE risk: collision penalty "
                    f"({self.collision.weight}) does not exceed the worst-case accumulated "
                    f"step cost ({worst_case_step_cost:.1f} = {self.step_cost.weight} x "
                    f"{max_steps:.0f} steps). Crashing early may be cheaper than timing out. "
                    "See the failure history in env/reward.py.",
                    stacklevel=2,
                )

        if (
            self.goal_reached.enabled
            and self.progress_shaping.enabled
            and max_goal_distance_m is not None
        ):
            # Total shaping available for traversing the whole map once. Note
            # this is NOT an oscillation-farming bound: shaping is symmetric
            # (retreating is penalized exactly as approaching is rewarded), so
            # oscillating nets ~zero. The real risk is a WEAK TERMINAL SIGNAL --
            # if the journey pays far more than the arrival, the policy optimizes
            # for travelling toward the goal rather than for actually reaching it,
            # and hovering just outside the goal tolerance becomes acceptable.
            traversal_shaping = self.progress_shaping.weight * max_goal_distance_m
            if self.goal_reached.weight < 0.5 * traversal_shaping:
                warnings.warn(
                    "WEAK TERMINAL SIGNAL: goal bonus "
                    f"({self.goal_reached.weight}) is small relative to the shaping reward "
                    f"for crossing the map once ({traversal_shaping:.1f}). Arrival should "
                    "dominate the journey. Raise reward.terms.goal_reached.weight or lower "
                    "progress_shaping.weight. See the failure history in env/reward.py.",
                    stacklevel=2,
                )


# ---------------------------------------------------------------------------
# Step state
# ---------------------------------------------------------------------------
@dataclass
class RewardState:
    """Everything the reward needs about one policy step.

    All array fields are per-env with shape ``(num_envs,)``, except ``action``
    and ``prev_action`` which are ``(num_envs, action_dim)``. Scalars are
    accepted throughout for local testing.
    """

    # --- v1 requirements ---
    prev_distance: Any          # distance to goal BEFORE this step (m)
    curr_distance: Any          # distance to goal AFTER this step (m)
    reached_goal: Any           # bool/0-1: goal reached this step
    collided: Any               # bool/0-1: collision detected this step

    # --- v2 requirements (optional; only read when the term is enabled) ---
    action: Optional[Any] = None        # current action, (num_envs, action_dim)
    prev_action: Optional[Any] = None   # previous action, same shape
    lin_vel: Optional[Any] = None       # body linear velocity (m/s)
    prev_lin_vel: Optional[Any] = None  # previous linear velocity (m/s)
    payload_mass: Optional[Any] = None  # per-env payload mass (kg)
    dt: float = 0.02                    # policy timestep (s)


# ---------------------------------------------------------------------------
# Individual terms. Each returns the SIGNED contribution (penalties negative).
# ---------------------------------------------------------------------------
def term_goal_reached(state: RewardState, spec: TermSpec) -> Any:
    """Terminal bonus for reaching the goal. Positive."""
    return spec.active_weight * _as_float(state.reached_goal)


def term_collision(state: RewardState, spec: TermSpec) -> Any:
    """Terminal penalty for collision. NEGATIVE."""
    return -spec.active_weight * _as_float(state.collided)


def term_step_cost(state: RewardState, spec: TermSpec) -> Any:
    """Constant per-step cost (efficiency / anti-stall). NEGATIVE.

    Shaped like the other terms so it broadcasts to ``(num_envs,)``: derived
    from ``curr_distance`` rather than returning a bare scalar.
    """
    ones = _as_float(state.curr_distance) * 0.0 + 1.0
    return -spec.active_weight * ones


def term_progress_shaping(state: RewardState, spec: TermSpec, clip_per_step: float) -> Any:
    """Dense shaping on distance-to-goal reduction. Signed.

    Positive when closing distance, negative when retreating. Clipped to
    ``+/-clip_per_step`` metres so a reset or a physics teleport cannot inject a
    huge spurious reward spike -- the single most common source of a policy that
    trains to a great return curve and a terrible success rate.
    """
    delta = _as_float(state.prev_distance) - _as_float(state.curr_distance)
    return spec.active_weight * _clip(delta, -clip_per_step, clip_per_step)


def term_action_rate(state: RewardState, spec: TermSpec) -> Any:
    """[v2] Penalize action churn -- jerky commands destabilize the payload."""
    if state.action is None or state.prev_action is None:
        return 0.0
    delta = _abs(_as_float(state.action) - _as_float(state.prev_action))
    return -spec.active_weight * _sum_last(delta)


def term_jerk(state: RewardState, spec: TermSpec) -> Any:
    """[v2] Penalize linear acceleration magnitude (payload-tipping proxy)."""
    if state.lin_vel is None or state.prev_lin_vel is None:
        return 0.0
    accel = _abs(_as_float(state.lin_vel) - _as_float(state.prev_lin_vel)) / max(state.dt, 1e-8)
    return -spec.active_weight * accel


def term_energy_under_load(state: RewardState, spec: TermSpec, mass_norm_kg: float) -> Any:
    """[v2] Penalize work done while loaded: mass-scaled speed x time.

    Normalized by ``payload_mass_norm_kg`` so the term's scale stays comparable
    across the payload range instead of growing with heavier payloads.
    """
    if state.lin_vel is None or state.payload_mass is None:
        return 0.0
    mass_factor = _as_float(state.payload_mass) / max(mass_norm_kg, 1e-8)
    return -spec.active_weight * mass_factor * _abs(_as_float(state.lin_vel)) * state.dt


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def compute_reward(state: RewardState, cfg: RewardConfig) -> tuple[Any, Dict[str, Any]]:
    """Compute the total reward and its per-term breakdown.

    Returns
    -------
    total : same backend/shape as the state arrays
        Sum of all enabled, signed terms.
    components : dict[str, Any]
        Per-term signed contributions, for W&B logging and reward debugging.
        Disabled terms are present with value 0.0, so logged keys stay stable
        across ablations (a term vanishing from a chart would otherwise look
        like a logging bug).
    """
    components: Dict[str, Any] = {
        "goal_reached": term_goal_reached(state, cfg.goal_reached),
        "collision": term_collision(state, cfg.collision),
        "step_cost": term_step_cost(state, cfg.step_cost),
        "progress_shaping": term_progress_shaping(
            state, cfg.progress_shaping, cfg.progress_clip_per_step
        ),
        "action_rate": term_action_rate(state, cfg.action_rate),
        "jerk": term_jerk(state, cfg.jerk),
        "energy_under_load": term_energy_under_load(
            state, cfg.energy_under_load, cfg.payload_mass_norm_kg
        ),
    }

    total = components["goal_reached"]
    for name, value in components.items():
        if name != "goal_reached":
            total = total + value

    return total, components
