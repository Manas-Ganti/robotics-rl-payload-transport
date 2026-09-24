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


# deck_link (default): a payload rigid body fixed-jointed to the chassis, INSIDE
#   the articulation -- the same way Carter's own 20 kg ballast (com_offset) is
#   modelled. PhysX composes mass, centre of mass and inertia exactly.
# mass_modifier: adds mass to the chassis body only. Kept for comparison, but
#   it leaves the chassis COM and inertia unchanged -- on Carter (two front
#   wheels + one rear caster, no front support) that tipped robots up to 52 deg.
# rigid_body_with_joint: a free rigid body welded after cloning. NOT wired.
ATTACH_MODES = ("deck_link", "mass_modifier", "rigid_body_with_joint")


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
        if self.attach_mode not in ATTACH_MODES:
            raise ValueError(f"attach_mode must be one of {ATTACH_MODES}, got {self.attach_mode!r}")

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


def author_payload_link(
    stage: Any,
    robot_path: str,
    chassis_name: str,
    offset_root_m: Sequence[float],
    size_m: Sequence[float],
    nominal_mass_kg: float,
) -> str:
    """Author a payload body inside env_0's robot, fixed-jointed to the chassis.

    Call BEFORE ``clone_environments`` so every env gets it. The body becomes an
    articulation link (as Carter's ``com_offset`` ballast is), so PhysX composes
    the robot's total mass, centre of mass and inertia itself -- no hand-derived
    combination to get wrong. It has NO collision shape: it is load, not a
    bumper (the collision sensor watches the chassis only).

    ``offset_root_m`` is the payload's centre in the ROBOT ROOT frame; the
    joint's chassis-side frame is derived from the stage, so a chassis_link that
    is offset or rotated relative to the root is handled.

    VERIFY ON ARC: the smoke test lists "payload" among robot.body_names, and
    its PhysX mass reads back across the sampled range.
    """
    from pxr import Gf, UsdGeom, UsdPhysics  # noqa: PLC0415  (lazy: Isaac-only)

    robot = stage.GetPrimAtPath(robot_path)
    chassis = stage.GetPrimAtPath(f"{robot_path}/{chassis_name}")
    if not robot or not chassis:
        raise RuntimeError(f"cannot author payload: missing {robot_path} or its '{chassis_name}' body")

    body_path = f"{robot_path}/payload"
    body = UsdGeom.Xform.Define(stage, body_path)
    body.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in offset_root_m)))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass_api.CreateMassAttr(float(nominal_mass_kg))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*box_inertia_diagonal(nominal_mass_kg, size_m)))

    # Joint frame on the chassis side = the payload's pose expressed in the
    # chassis frame (USD row-vector convention: local * parent_to_world).
    xf = UsdGeom.XformCache()
    payload_to_world = xf.GetLocalToWorldTransform(body.GetPrim())
    rel = payload_to_world * xf.GetLocalToWorldTransform(chassis).GetInverse()
    joint = UsdPhysics.FixedJoint.Define(stage, f"{robot_path}/payload_joint")
    joint.CreateBody0Rel().SetTargets([chassis.GetPath()])
    joint.CreateBody1Rel().SetTargets([body.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(rel.ExtractTranslation()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(rel.ExtractRotationQuat()))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return body_path


def set_link_payload_mass(
    robot: Any,
    body_index: int,
    masses_kg: Any,
    size_m: Sequence[float],
    env_ids: Optional[Any] = None,
) -> None:
    """Per-env payload mass on the deck link, with its box inertia to match.

    PhysX does not rescale inertia when mass changes, so the inertia is written
    alongside: a uniform box of this mass (diagonal, about the link origin,
    which is the payload centre). Full-tensor read-modify-write with CPU
    indices, as in Isaac Lab's randomize_rigid_body_mass.
    """
    import torch  # noqa: PLC0415  (lazy: Isaac-only)

    view = robot.root_physx_view
    idx = (
        torch.arange(view.count) if env_ids is None else torch.as_tensor(env_ids)
    ).long().cpu()
    masses = view.get_masses().clone()
    inertias = view.get_inertias().clone()  # (N, B, 9), row-major 3x3
    m = torch.as_tensor(masses_kg, dtype=masses.dtype).reshape(-1).cpu()
    w, d, h = (float(v) for v in size_m)
    masses[idx, body_index] = m
    inertias[idx, body_index] = 0.0
    inertias[idx, body_index, 0] = m / 12.0 * (d * d + h * h)
    inertias[idx, body_index, 4] = m / 12.0 * (w * w + h * h)
    inertias[idx, body_index, 8] = m / 12.0 * (w * w + d * d)
    view.set_masses(masses, idx)
    view.set_inertias(inertias, idx)


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

    # Read-modify-write the FULL (num_envs, num_bodies) CPU tensor, then write
    # back with indices (Isaac Lab's randomize_rigid_body_mass pattern).
    view = payload_object.root_physx_view
    all_masses = view.get_masses().clone()
    idx = (
        torch.arange(all_masses.shape[0]) if env_ids is None else torch.as_tensor(env_ids)
    ).long().cpu()
    values = torch.as_tensor(masses_kg, dtype=all_masses.dtype).reshape(-1).cpu()
    all_masses[idx, 0] = values
    view.set_masses(all_masses, idx)


def apply_mass_modifier(
    robot: Any,
    payload_masses_kg: Any,
    body_index: int,
    default_masses: Any,
    env_ids: Optional[Any] = None,
) -> None:
    """Cheap payload mode: fold payload mass into the chassis body mass.

    Trades physical fidelity for speed -- there is no separate payload body, so
    no load-shift or tip-over dynamics, only added inertia. Adequate for Phase 1
    (where payload is "static known") and much faster at num_envs=2048.

    Switch ``payload.attach_mode`` to ``rigid_body_with_joint`` before drawing
    conclusions from the transport-aware v2 reward: penalizing jerk to protect a
    payload that has no independent dynamics is not measuring what it claims to.

    chassis mass = the USD's own chassis mass (``default_masses``, snapshotted
    before the first write) + this episode's payload. Using the real USD mass
    rather than a config constant means the payload-to-robot ratio reported in
    the write-up is the one the simulator actually ran.

    PhysX view tensors are CPU-side; indices must be CPU too. Full-tensor
    read-modify-write, as in Isaac Lab's randomize_rigid_body_mass.
    VERIFY ON ARC: the smoke test's chassis-mass spread matches the payload
    range (read back from the view, not from our own buffers).
    """
    import torch  # noqa: PLC0415  (lazy: A100-only import)

    view = robot.root_physx_view
    masses = view.get_masses().clone()
    idx = (
        torch.arange(masses.shape[0]) if env_ids is None else torch.as_tensor(env_ids)
    ).long().cpu()
    payload = torch.as_tensor(payload_masses_kg, dtype=masses.dtype).reshape(-1).cpu()
    masses[idx, body_index] = default_masses[idx, body_index] + payload
    view.set_masses(masses, idx)


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
