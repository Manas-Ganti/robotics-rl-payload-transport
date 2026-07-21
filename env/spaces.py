"""Observation / action space definitions -- the SINGLE SOURCE OF TRUTH.

PURE LOGIC -- no Isaac Sim import.

Everything that needs to know the observation layout reads it from here:
``nav_env.py`` (assembling obs on GPU), ``nav2_runner.py`` (feeding the baseline
through the same harness), ``run_eval.py`` (loading a checkpoint), and the tests.
If the layout lived in two places it would eventually disagree in one of them,
and a checkpoint would be silently fed scrambled observations -- a failure that
looks like "the policy generalizes badly" rather than like a bug.

Layout (fixed order; disabled components are omitted, never zero-padded):

    [ depth (num_rays) | goal_pose (3) | proprioception (2) | payload_mass (1) ]

    goal_pose      = (cos(bearing), sin(bearing), normalized_distance)
    proprioception = (normalized_lin_vel, normalized_ang_vel)
    payload_mass   = (normalized_mass,)

Bearing is encoded as (cos, sin) rather than a raw angle so there is no
discontinuity at +/-pi -- a wrap-around jump there produces exactly the kind of
rare, hard-to-diagnose action spike that shows up as a mystery collision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Component names in their canonical order. Do not reorder: existing checkpoints
# are interpreted against this sequence.
COMPONENT_ORDER: Tuple[str, ...] = ("depth", "goal_pose", "proprioception", "payload_mass")

GOAL_POSE_DIM = 3
PROPRIOCEPTION_DIM = 2
PAYLOAD_MASS_DIM = 1
ACTION_DIM = 2  # (linear velocity, angular velocity)


@dataclass(frozen=True)
class ObservationSpec:
    """Resolved observation layout for one experiment configuration."""

    components: Tuple[str, ...]
    sizes: Tuple[int, ...]
    offsets: Tuple[int, ...]
    total_dim: int

    # normalization constants (all traced to config)
    max_range_m: float
    max_distance_m: float
    max_lin_vel: float
    max_ang_vel: float
    max_payload_mass_kg: float
    clip_range: Tuple[float, float]
    normalize: bool

    def slice_of(self, component: str) -> slice:
        """Index slice for one component -- for debugging and obs ablations."""
        if component not in self.components:
            raise KeyError(
                f"Component '{component}' is disabled in this config. "
                f"Active components: {self.components}"
            )
        idx = self.components.index(component)
        return slice(self.offsets[idx], self.offsets[idx] + self.sizes[idx])

    def has(self, component: str) -> bool:
        return component in self.components

    def describe(self) -> str:
        """Human-readable layout summary -- logged at startup on the A100."""
        lines = [f"ObservationSpec(total_dim={self.total_dim})"]
        for name, size, offset in zip(self.components, self.sizes, self.offsets):
            lines.append(f"  [{offset:4d}:{offset + size:4d}] {name} ({size})")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "components": list(self.components),
            "sizes": list(self.sizes),
            "total_dim": self.total_dim,
        }


def build_observation_spec(cfg: Any) -> ObservationSpec:
    """Resolve the observation layout from a loaded config.

    Honours both the ``observations.*`` toggles and the
    ``phases.enable_inferred_payload`` flag -- the Phase 5 experiment
    (told-vs-inferred payload) is exactly this one flag, per CLAUDE.md.
    """
    data = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    obs_cfg = data["observations"]
    phases = data.get("phases", {})
    sensors = data.get("sensors", {})
    action_cfg = data["action"]

    # Sensor dimensionality depends on the chosen modality.
    modality = sensors.get("modality", "raycaster")
    if modality == "raycaster":
        depth_dim = int(sensors["raycaster"]["num_rays"])
        max_range = float(sensors["raycaster"]["max_range_m"])
    elif modality == "camera":
        cam = sensors["camera"]
        depth_dim = int(cam["width"]) * int(cam["height"])
        max_range = float(cam["max_range_m"])
    else:
        raise ValueError(f"sensors.modality must be 'raycaster' or 'camera', got {modality!r}")

    # Phase 5: inferred payload -- drop payload_mass from the observation.
    use_payload_mass = bool(obs_cfg["use_payload_mass"]) and not bool(
        phases.get("enable_inferred_payload", False)
    )

    enabled: Dict[str, int] = {
        "depth": depth_dim if obs_cfg["use_depth"] else 0,
        "goal_pose": GOAL_POSE_DIM if obs_cfg["use_goal_pose"] else 0,
        "proprioception": PROPRIOCEPTION_DIM if obs_cfg["use_proprioception"] else 0,
        "payload_mass": PAYLOAD_MASS_DIM if use_payload_mass else 0,
    }

    components: List[str] = []
    sizes: List[int] = []
    offsets: List[int] = []
    cursor = 0
    for name in COMPONENT_ORDER:
        size = enabled[name]
        if size <= 0:
            continue
        components.append(name)
        sizes.append(size)
        offsets.append(cursor)
        cursor += size

    if cursor == 0:
        raise ValueError("All observation components are disabled -- the policy would be blind.")

    # Max goal distance: the terrain diagonal bounds any in-patch separation.
    terrain_size = float(data["env"]["terrain_size_m"])
    max_distance = terrain_size * math.sqrt(2.0)

    # Payload normalizer must exceed the OOD ceiling, not just the training
    # ceiling: if it saturated, 6 kg and 9 kg would be indistinguishable to the
    # policy and the payload OOD curve would flatten for a purely
    # representational reason rather than a dynamics one.
    max_payload = float(obs_cfg["payload_mass_norm_kg"])

    return ObservationSpec(
        components=tuple(components),
        sizes=tuple(sizes),
        offsets=tuple(offsets),
        total_dim=cursor,
        max_range_m=max_range,
        max_distance_m=max_distance,
        max_lin_vel=float(max(abs(v) for v in action_cfg["lin_vel_range"])),
        max_ang_vel=float(max(abs(v) for v in action_cfg["ang_vel_range"])),
        max_payload_mass_kg=max_payload,
        clip_range=tuple(float(v) for v in obs_cfg["clip_range"]),  # type: ignore[arg-type]
        normalize=bool(obs_cfg["normalize"]),
    )


# ---------------------------------------------------------------------------
# Goal-relative pose
# ---------------------------------------------------------------------------
def goal_relative_pose(
    robot_xy: Sequence[float],
    robot_yaw: float,
    goal_xy: Sequence[float],
) -> Tuple[float, float, float]:
    """Return ``(cos_bearing, sin_bearing, distance_m)`` in the ROBOT frame.

    The bearing is the goal direction relative to the robot's heading, so the
    observation is heading-invariant -- the policy learns "turn toward the goal"
    rather than memorizing world-frame directions.
    """
    dx = float(goal_xy[0]) - float(robot_xy[0])
    dy = float(goal_xy[1]) - float(robot_xy[1])
    distance = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx) - float(robot_yaw)
    return (math.cos(bearing), math.sin(bearing), distance)


def build_observation(
    spec: ObservationSpec,
    *,
    depth: Optional[np.ndarray] = None,
    robot_xy: Optional[Sequence[float]] = None,
    robot_yaw: Optional[float] = None,
    goal_xy: Optional[Sequence[float]] = None,
    lin_vel: Optional[float] = None,
    ang_vel: Optional[float] = None,
    payload_mass_kg: Optional[float] = None,
) -> np.ndarray:
    """Assemble a single observation vector (NumPy, single agent).

    Used by the Nav2 baseline runner and by tests. ``nav_env.py`` builds the
    batched GPU equivalent and MUST produce an identical layout -- that is the
    contract this module exists to hold.
    """
    parts: List[np.ndarray] = []

    for name in spec.components:
        if name == "depth":
            if depth is None:
                raise ValueError("Observation spec includes 'depth' but none was provided")
            values = np.asarray(depth, dtype=np.float32).reshape(-1)
            if values.size != spec.sizes[spec.components.index("depth")]:
                raise ValueError(
                    f"depth has {values.size} elements, spec expects "
                    f"{spec.sizes[spec.components.index('depth')]}"
                )
            parts.append(values / spec.max_range_m if spec.normalize else values)

        elif name == "goal_pose":
            if robot_xy is None or robot_yaw is None or goal_xy is None:
                raise ValueError("Observation spec includes 'goal_pose' but pose inputs are missing")
            cos_b, sin_b, distance = goal_relative_pose(robot_xy, robot_yaw, goal_xy)
            norm_d = distance / spec.max_distance_m if spec.normalize else distance
            parts.append(np.array([cos_b, sin_b, norm_d], dtype=np.float32))

        elif name == "proprioception":
            if lin_vel is None or ang_vel is None:
                raise ValueError("Observation spec includes 'proprioception' but velocities are missing")
            if spec.normalize:
                values = [lin_vel / spec.max_lin_vel, ang_vel / spec.max_ang_vel]
            else:
                values = [lin_vel, ang_vel]
            parts.append(np.array(values, dtype=np.float32))

        elif name == "payload_mass":
            if payload_mass_kg is None:
                raise ValueError("Observation spec includes 'payload_mass' but no mass was provided")
            value = (
                payload_mass_kg / spec.max_payload_mass_kg if spec.normalize else payload_mass_kg
            )
            parts.append(np.array([value], dtype=np.float32))

    obs = np.concatenate(parts).astype(np.float32)
    return np.clip(obs, spec.clip_range[0], spec.clip_range[1])


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionSpec:
    """Continuous action space: (linear velocity, angular velocity)."""

    lin_vel_range: Tuple[float, float]
    ang_vel_range: Tuple[float, float]
    scale: float = 1.0
    dim: int = ACTION_DIM

    @classmethod
    def from_config(cls, cfg: Any) -> "ActionSpec":
        data = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
        action_cfg = data["action"]
        return cls(
            lin_vel_range=tuple(float(v) for v in action_cfg["lin_vel_range"]),  # type: ignore[arg-type]
            ang_vel_range=tuple(float(v) for v in action_cfg["ang_vel_range"]),  # type: ignore[arg-type]
            scale=float(action_cfg.get("scale", 1.0)),
        )


def scale_action(raw_action: np.ndarray, spec: ActionSpec) -> np.ndarray:
    """Map a policy output in ``[-1, 1]^2`` to physical velocity commands.

    Asymmetric ranges (e.g. limited reverse) are handled by mapping through the
    midpoint rather than by naive multiplication, so a raw 0.0 corresponds to
    the range's centre and the full commanded range remains reachable.
    """
    raw = np.clip(np.asarray(raw_action, dtype=np.float64).reshape(-1, ACTION_DIM), -1.0, 1.0)
    raw = raw * spec.scale

    out = np.empty_like(raw)
    for i, (lo, hi) in enumerate((spec.lin_vel_range, spec.ang_vel_range)):
        mid, half = (hi + lo) / 2.0, (hi - lo) / 2.0
        out[:, i] = np.clip(mid + raw[:, i] * half, lo, hi)

    return out.squeeze() if out.shape[0] == 1 else out


def differential_drive_wheel_speeds(
    lin_vel: float,
    ang_vel: float,
    *,
    wheel_radius_m: float,
    wheel_base_m: float,
    max_wheel_speed_rad_s: float,
) -> Tuple[float, float]:
    """Convert a (v, omega) command into (left, right) wheel angular velocities.

    Standard differential-drive inverse kinematics. When a command exceeds the
    wheel speed limit, BOTH wheels are scaled by the same factor so the turn
    radius is preserved -- clipping them independently would silently alter the
    commanded heading, which the policy has no way to observe or correct for.

    VERIFY ON A100: confirm the sign convention matches the robot USD (a flipped
    wheel axis makes the robot drive backwards on a positive command).
    """
    if wheel_radius_m <= 0 or wheel_base_m <= 0:
        raise ValueError("wheel_radius_m and wheel_base_m must be > 0")

    left = (lin_vel - ang_vel * wheel_base_m / 2.0) / wheel_radius_m
    right = (lin_vel + ang_vel * wheel_base_m / 2.0) / wheel_radius_m

    peak = max(abs(left), abs(right))
    if peak > max_wheel_speed_rad_s:
        factor = max_wheel_speed_rad_s / peak
        left, right = left * factor, right * factor

    return (left, right)
