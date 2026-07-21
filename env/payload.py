"""Variable-mass payload: spec, inertia math, and Isaac attachment.

Payload mass is THE dynamics axis of this study -- it replaces gravity
randomization entirely (CLAUDE.md principle 7), because a robot carrying an
unknown load is a real deployment condition while a robot on a planet with
different gravity is not.

Structure follows the project's split:
  * The top of this file is PURE math (inertia, CoM shift, tip-over margin) and
    is unit-testable locally with no simulator.
  * The bottom holds the Isaac-dependent attachment calls, each isolated in a
    small function with a ``# VERIFY ON A100:`` note. Isaac is imported lazily
    INSIDE those functions so this module remains importable on a GPU-less
    machine -- the pure math and the tests do not pay for the simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Pure spec + math
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PayloadSpec:
    """Resolved payload configuration for one episode."""

    mass_kg: float
    size_m: Tuple[float, float, float]
    offset_m: Tuple[float, float, float]
    attach_mode: str
    shape: str = "box"
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.mass_kg < 0:
            raise ValueError(f"payload mass must be >= 0, got {self.mass_kg}")
        if self.attach_mode not in ("mass_modifier", "rigid_body_with_joint"):
            raise ValueError(
                f"attach_mode must be 'mass_modifier' or 'rigid_body_with_joint', "
                f"got {self.attach_mode!r}"
            )

    @classmethod
    def from_config(cls, cfg: Any, *, mass_kg: float) -> "PayloadSpec":
        """Build from the ``payload`` block of robot.yaml plus a sampled mass."""
        data = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
        payload_cfg = data["payload"]
        return cls(
            mass_kg=float(mass_kg),
            size_m=tuple(float(v) for v in payload_cfg["size_m"]),  # type: ignore[arg-type]
            offset_m=tuple(float(v) for v in payload_cfg["offset_m"]),  # type: ignore[arg-type]
            attach_mode=str(payload_cfg["attach_mode"]),
            shape=str(payload_cfg.get("shape", "box")),
            enabled=bool(payload_cfg.get("enabled", True)),
        )


def box_inertia_diagonal(mass_kg: float, size_m: Sequence[float]) -> Tuple[float, float, float]:
    """Diagonal inertia tensor of a uniform-density box about its own centre.

    I_xx = m/12 * (h^2 + d^2), and cyclic permutations.
    """
    if mass_kg < 0:
        raise ValueError(f"mass must be >= 0, got {mass_kg}")
    w, d, h = (float(v) for v in size_m)
    if min(w, d, h) <= 0:
        raise ValueError(f"box dimensions must be > 0, got {size_m}")

    factor = mass_kg / 12.0
    return (
        factor * (d**2 + h**2),
        factor * (w**2 + h**2),
        factor * (w**2 + d**2),
    )


def combined_center_of_mass(
    base_mass_kg: float,
    payload_mass_kg: float,
    payload_offset_m: Sequence[float],
) -> Tuple[float, float, float]:
    """Centre of mass of (chassis + payload), in the chassis frame.

    The chassis CoM is taken as the frame origin. Returns the weighted CoM,
    which for a top-mounted payload sits ABOVE the chassis -- the higher it
    rides, the more readily the robot tips on a slope. That coupling is exactly
    what the payload x slope corner of the OOD grid is probing.
    """
    total = base_mass_kg + payload_mass_kg
    if total <= 0:
        raise ValueError("combined mass must be > 0")
    factor = payload_mass_kg / total
    return tuple(float(v) * factor for v in payload_offset_m)  # type: ignore[return-value]


def tip_over_margin_deg(
    base_mass_kg: float,
    payload_mass_kg: float,
    payload_offset_m: Sequence[float],
    *,
    wheel_base_m: float,
) -> float:
    """Static tip-over angle (degrees) for the loaded robot on a slope.

    Quasi-static estimate: the robot tips when the combined CoM passes outside
    the wheel contact patch, i.e. ``atan((wheel_base/2) / com_height)``.

    This is a PLANNING aid, not ground truth -- it ignores dynamics, suspension,
    and friction limits, so the simulator will tip earlier than this predicts
    under acceleration. Its value is in sanity-checking the OOD grid before a
    long run: if this returns an angle BELOW the OOD slope range, the heavy
    payload x steep slope cells are physically impossible rather than merely
    hard, and any policy will fail them. That distinction matters enormously
    when interpreting a degradation cliff -- a cliff at an impossible cell says
    nothing about generalization.
    """
    com = combined_center_of_mass(base_mass_kg, payload_mass_kg, payload_offset_m)
    com_height = abs(com[2])
    if com_height <= 1e-9:
        return 90.0
    return float(math.degrees(math.atan((wheel_base_m / 2.0) / com_height)))


def payload_mass_ratio(base_mass_kg: float, payload_mass_kg: float) -> float:
    """Payload mass as a fraction of the chassis mass.

    Worth logging: at ratio >> 1 the payload dominates the dynamics and the
    robot is effectively a different machine. If the OOD payload range crosses
    that threshold while the training range does not, an observed failure is
    arguably a task change rather than a generalization gap -- and the write-up
    should say so.
    """
    if base_mass_kg <= 0:
        raise ValueError(f"base_mass_kg must be > 0, got {base_mass_kg}")
    return float(payload_mass_kg / base_mass_kg)


# ---------------------------------------------------------------------------
# Isaac-dependent attachment. Isaac imports are LAZY (inside functions).
# ---------------------------------------------------------------------------
def build_payload_cfg(spec: PayloadSpec, prim_path: str) -> Any:
    """Build the Isaac Lab RigidObjectCfg for a rigid-body payload.

    VERIFY ON A100: import paths and cfg field names for the installed version.
      Isaac Lab 1.x : isaaclab.assets.RigidObjectCfg, isaaclab.sim.CuboidCfg
      Orbit (older) : omni.isaac.orbit.assets.RigidObjectCfg
    Specifically confirm that ``MassPropertiesCfg`` accepts ``mass`` (older
    versions used ``density`` only) -- this whole study varies mass directly.
    """
    import isaaclab.sim as sim_utils  # noqa: PLC0415  (lazy: A100-only import)
    from isaaclab.assets import RigidObjectCfg  # noqa: PLC0415

    inertia = box_inertia_diagonal(spec.mass_kg, spec.size_m)

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=spec.size_m,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=spec.offset_m),
    )


def attach_payload_joint(
    stage: Any,
    chassis_prim_path: str,
    payload_prim_path: str,
    offset_m: Sequence[float],
) -> None:
    """Create a FixedJoint welding the payload to the chassis.

    VERIFY ON A100: this uses raw USD physics schemas rather than an Isaac Lab
    helper, because Isaac Lab has no first-class "attach rigid body to
    articulation link" API as of writing. Check that:
      1. ``UsdPhysics.FixedJoint`` is the right schema (vs. PhysxSchema variants)
      2. body0/body1 ordering matches the convention (parent first)
      3. the joint survives ``sim.reset()`` -- some joint types are re-created
         on stage reset and would silently detach, dropping the payload

    If the payload visibly falls through or lags the chassis in the viewport,
    this function is the first place to look.
    """
    from pxr import Gf, UsdPhysics  # noqa: PLC0415  (lazy: A100-only import)

    joint_path = f"{payload_prim_path}/FixedJoint"
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([chassis_prim_path])
    joint.CreateBody1Rel().SetTargets([payload_prim_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in offset_m)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))


def set_payload_mass_batched(
    payload_object: Any,
    masses_kg: Any,
    env_ids: Optional[Any] = None,
) -> None:
    """Set per-environment payload masses on the GPU.

    This is the hot path: it runs on every reset for every env, so it must be a
    batched tensor write, never a Python loop over environments.

    VERIFY ON A100: the exact API for writing masses to a RigidObject.
      Expected (Isaac Lab 1.x):
          payload_object.root_physx_view.set_masses(masses, env_ids)
      Note ``set_masses`` typically expects a CPU tensor even in GPU pipelines,
      and shape ``(num_envs, num_bodies)``. If masses appear not to take effect,
      check BOTH the device and the shape before anything else -- a silently
      ignored write here means the entire payload axis is inert and every
      payload OOD result is meaningless while looking perfectly plausible.

    ALSO VERIFY: whether inertia must be updated alongside mass. PhysX does not
    rescale inertia automatically; if it does not, use ``set_inertias`` with
    :func:`box_inertia_diagonal` or heavy payloads will rotate unrealistically
    easily.
    """
    import torch  # noqa: PLC0415  (lazy: A100-only import)

    masses = torch.as_tensor(masses_kg, dtype=torch.float32)
    if masses.dim() == 1:
        masses = masses.unsqueeze(-1)  # (num_envs,) -> (num_envs, 1 body)

    view = payload_object.root_physx_view
    indices = env_ids if env_ids is not None else torch.arange(masses.shape[0])
    view.set_masses(masses.cpu(), indices.cpu() if hasattr(indices, "cpu") else indices)


def apply_mass_modifier(
    robot: Any,
    base_mass_kg: float,
    payload_masses_kg: Any,
    body_index: int,
    env_ids: Optional[Any] = None,
) -> None:
    """Cheap payload mode: fold payload mass into the chassis body mass.

    Trades physical fidelity for speed -- there is no separate payload body, so
    no load-shift or tip-over dynamics, only added inertia. Adequate for Phase 1
    (where payload is "static known") and much faster at num_envs=2048.

    Switch ``payload.attach_mode`` to ``rigid_body_with_joint`` before drawing
    conclusions from the transport-aware v2 reward: penalizing jerk to protect a
    payload that has no independent dynamics is not measuring what it claims to.

    VERIFY ON A100: same set_masses signature caveats as
    :func:`set_payload_mass_batched`.
    """
    import torch  # noqa: PLC0415  (lazy: A100-only import)

    payload = torch.as_tensor(payload_masses_kg, dtype=torch.float32).reshape(-1)
    total = payload + float(base_mass_kg)

    view = robot.root_physx_view
    masses = view.get_masses().clone()
    indices = env_ids if env_ids is not None else torch.arange(masses.shape[0])
    masses[indices, body_index] = total.to(masses.device)
    view.set_masses(masses.cpu(), indices.cpu() if hasattr(indices, "cpu") else indices)


def summarize_payload(
    spec: PayloadSpec,
    *,
    base_mass_kg: float,
    wheel_base_m: float,
) -> dict:
    """Diagnostic summary -- logged once per run to make the physics inspectable."""
    return {
        "payload/mass_kg": spec.mass_kg,
        "payload/mass_ratio": payload_mass_ratio(base_mass_kg, spec.mass_kg),
        "payload/com_height_m": combined_center_of_mass(
            base_mass_kg, spec.mass_kg, spec.offset_m
        )[2],
        "payload/tip_over_margin_deg": tip_over_margin_deg(
            base_mass_kg, spec.mass_kg, spec.offset_m, wheel_base_m=wheel_base_m
        ),
    }
