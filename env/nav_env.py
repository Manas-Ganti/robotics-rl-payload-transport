"""Isaac Lab environment: reset / step / obs / reward / termination.

THIS IS THE ONLY MODULE THAT IMPORTS ISAAC SIM. It is written but NOT unit
tested locally (CLAUDE.md) -- everything testable was deliberately pushed into
``reward.py``, ``solvability.py``, ``terrain_factory.py``, ``spaces.py``, and
``config.py``, all of which are pure and covered by ``tests/``.

Every uncertain Isaac Lab API call is isolated in a small function marked
``# VERIFY ON A100:``. Those are collected as a checklist in ``setup_notes.md``.
Work through that checklist before the first long training run -- several of the
uncertain calls fail SILENTLY (a mass write that does not land, a friction
material that does not bind), and a silent failure here produces a full set of
plausible-looking results for an experiment that never actually varied.

Design notes
------------
* **Obstacle pooling.** Isaac Lab clones a single env prim across thousands of
  instances, so per-env obstacle COUNTS cannot vary structurally. Instead each
  env gets a fixed pool of ``terrain.max_obstacles_per_env`` obstacle prims and
  unused ones are parked far below the floor. This keeps the scene graph static
  (which is what makes 2048 envs viable) while the LAYOUT stays fully procedural.
* **Terrain lives on the CPU, physics on the GPU.** TerrainSpec generation
  (NumPy, with the A* solvability check) runs on CPU at reset; only the
  resulting poses are pushed to GPU tensors. At high ``num_envs`` this is the
  main reset cost -- see ``setup_notes.md`` for the mitigation if it dominates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from env.config import Config, get_train_ranges
from env.payload import PayloadSpec, apply_mass_modifier, set_payload_mass_batched
from env.randomization import DomainSampler, EpisodeParams, resolve_physics_material
from env.reward import RewardConfig, RewardState, compute_reward
from env.spaces import ActionSpec, ObservationSpec, build_observation_spec
from env.lidar import LidarSpec, cast_rays, obstacles_to_arrays
from env.terrain_factory import TerrainFactory, TerrainSpec, pool_slot_radii

# ---------------------------------------------------------------------------
# Isaac Lab imports.
# VERIFY ON A100: module paths changed across Isaac Gym -> Orbit -> Isaac Lab.
#   Isaac Lab 2.x : isaaclab.*            (pinned: v2.1.0, see arc/setup_env.sh)
#   Isaac Lab 1.x : omni.isaac.lab.*
#   Orbit (older) : omni.isaac.orbit.*
# ---------------------------------------------------------------------------
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns  # noqa: E402
from isaaclab.sim import SimulationCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402


# ---------------------------------------------------------------------------
# Terminal outcome codes (shared with eval/metrics.py -- keep in sync)
# ---------------------------------------------------------------------------
OUTCOME_RUNNING = 0
OUTCOME_SUCCESS = 1
OUTCOME_COLLISION = 2
OUTCOME_TIMEOUT = 3

OUTCOME_NAMES: Dict[int, str] = {
    OUTCOME_RUNNING: "running",
    OUTCOME_SUCCESS: "success",
    OUTCOME_COLLISION: "collision",
    OUTCOME_TIMEOUT: "timeout",
}


@dataclass
class EpisodeRecord:
    """One completed episode, as consumed by ``eval/metrics.py``.

    Mirrors the CSV schema written by the OOD harness. Every field is measured
    mechanically against ground truth -- no subjective scoring (principle 4).
    """

    outcome: str
    params: Dict[str, float]
    optimal_path_length_m: float
    actual_path_length_m: float
    episode_length_steps: int
    final_distance_to_goal_m: float
    collided: bool
    reached_goal: bool
    timed_out: bool


# ---------------------------------------------------------------------------
# Isaac Lab env configuration
# ---------------------------------------------------------------------------
def resolve_robot_cfg(import_path: str) -> ArticulationCfg:
    """Dynamically import the built-in robot ArticulationCfg named in robot.yaml.

    Isolated so swapping robots is a config edit, never a code edit
    (CLAUDE.md: "Isolate the robot choice in configs/robot.yaml").

    VERIFY ON A100: confirm the import path resolves on the installed version.
        default : isaaclab_assets.robots.jetbot.JETBOT_CFG
        older   : omni.isaac.orbit_assets.jetbot.JETBOT_CFG
    """
    import importlib

    module_path, _, attr = import_path.rpartition(".")
    if not module_path:
        raise ValueError(f"robot.cfg_import_path must be fully qualified, got {import_path!r}")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Could not import '{module_path}' (from robot.cfg_import_path={import_path!r}). "
            "Isaac Lab asset module paths differ between versions -- see setup_notes.md."
        ) from exc

    if not hasattr(module, attr):
        raise AttributeError(f"'{module_path}' has no attribute '{attr}'")
    return getattr(module, attr)


@configclass
class TransportNavEnvCfg(DirectRLEnvCfg):
    """Isaac Lab env config, populated from YAML by :func:`build_env_cfg`.

    Isaac Lab requires a ``@configclass``, so YAML values are transferred onto
    this object rather than read from a dict at runtime. :func:`build_env_cfg`
    is the single translation point between the two representations.
    """

    decimation: int = 4
    episode_length_s: float = 30.0
    action_space: int = 2
    observation_space: int = 1  # overwritten from the resolved ObservationSpec
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=12.0)

    robot: ArticulationCfg = None  # type: ignore[assignment]
    contact_sensor: ContactSensorCfg = None  # type: ignore[assignment]
    ray_caster: RayCasterCfg = None  # type: ignore[assignment]

    # Raw YAML, carried through so the env can read study-specific settings that
    # have no place in Isaac Lab's own cfg schema.
    raw: Dict[str, Any] = None  # type: ignore[assignment]


def build_env_cfg(cfg: Config) -> TransportNavEnvCfg:
    """Translate a loaded YAML config into an Isaac Lab env config."""
    data = cfg.to_dict()
    sim_cfg, env_cfg, robot_cfg, sensors_cfg = data["sim"], data["env"], data["robot"], data["sensors"]

    obs_spec = build_observation_spec(cfg)

    out = TransportNavEnvCfg()
    out.decimation = int(sim_cfg["decimation"])
    out.episode_length_s = float(env_cfg["episode_length_s"])
    out.action_space = 2
    out.observation_space = obs_spec.total_dim
    out.raw = data

    # -- simulation --------------------------------------------------------
    # VERIFY ON A100: PhysxCfg field names; GPU buffer keys have been renamed
    # between releases and an unknown key may be silently dropped.
    out.sim = SimulationCfg(
        dt=float(sim_cfg["dt"]),
        render_interval=int(sim_cfg["decimation"]),
        device=str(sim_cfg["device"]),
        use_fabric=bool(sim_cfg["use_fabric"]),
        physx=sim_utils.PhysxCfg(
            solver_position_iteration_count=int(sim_cfg["physx"]["solver_position_iteration_count"]),
            solver_velocity_iteration_count=int(sim_cfg["physx"]["solver_velocity_iteration_count"]),
            gpu_max_rigid_contact_count=int(sim_cfg["physx"]["gpu_max_rigid_contact_count"]),
            gpu_found_lost_pairs_capacity=int(sim_cfg["physx"]["gpu_found_lost_pairs_capacity"]),
        ),
    )

    # -- scene -------------------------------------------------------------
    out.scene = InteractiveSceneCfg(
        num_envs=int(env_cfg["num_envs"]),
        env_spacing=float(env_cfg["env_spacing"]),
        replicate_physics=True,
    )

    # -- robot -------------------------------------------------------------
    base_robot_cfg = resolve_robot_cfg(str(robot_cfg["cfg_import_path"]))
    out.robot = base_robot_cfg.replace(prim_path="/World/envs/env_.*/Robot")

    # -- contact sensor ----------------------------------------------------
    # VERIFY ON A100: prim path must match the robot's chassis body name.
    out.contact_sensor = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{robot_cfg['base_body_name']}",
        history_length=int(sensors_cfg["contact"]["history_length"]),
        track_air_time=bool(sensors_cfg["contact"]["track_air_time"]),
    )

    # -- exteroceptive sensor ---------------------------------------------
    if sensors_cfg["modality"] == "analytic_lidar":
        # No sensor prim: ranges are computed in torch from the episode's
        # TerrainSpec (env/lidar.py). Validate the geometry block up front so a
        # bad config fails here, not on the first observation.
        LidarSpec.from_config(cfg)
        out.ray_caster = None
    elif sensors_cfg["modality"] == "raycaster":
        rc = sensors_cfg["raycaster"]
        # VERIFY ON A100: LidarPatternCfg field names (`horizontal_fov_range` vs
        # `horizontal_fov`) differ across versions, and `mesh_prim_paths` must
        # list the meshes the rays actually hit. If depth returns all-max-range,
        # a missing mesh path is the usual cause.
        out.ray_caster = RayCasterCfg(
            prim_path=f"/World/envs/env_.*/Robot/{robot_cfg['base_body_name']}",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, float(rc["height_offset_m"]))),
            attach_yaw_only=True,
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1,
                vertical_fov_range=(0.0, 0.0),
                horizontal_fov_range=(
                    -float(rc["horizontal_fov_deg"]) / 2.0,
                    float(rc["horizontal_fov_deg"]) / 2.0,
                ),
                horizontal_res=float(rc["horizontal_fov_deg"]) / max(1, int(rc["num_rays"]) - 1),
            ),
            max_distance=float(rc["max_range_m"]),
            update_period=float(rc["update_period_s"]),
            mesh_prim_paths=["/World/ground", "/World/envs/env_.*/Obstacles"],
        )
    else:
        raise NotImplementedError(
            "sensors.modality='camera' is defined in robot.yaml but not wired here. "
            "depth-only via raycaster is the v1 path (CLAUDE.md marks RGB/camera "
            "lowest priority). To enable it, add a TiledCameraCfg here -- plain "
            "CameraCfg will not scale past a few hundred envs."
        )

    return out


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------
class TransportNavEnv(DirectRLEnv):
    """Payload-transport navigation over parameterized terrain.

    Mission: carry a sampled-mass payload from start to goal without collision.
    Termination: goal reached (success) | collision (fail) | timeout (fail).
    """

    cfg: TransportNavEnvCfg

    def __init__(self, cfg: TransportNavEnvCfg, render_mode: Optional[str] = None, **kwargs: Any):
        self._raw = cfg.raw
        self._cfg_obj = Config(cfg.raw)

        # -- resolved specs (pure, no Isaac) --------------------------------
        self.obs_spec: ObservationSpec = build_observation_spec(self._cfg_obj)
        self.action_spec: ActionSpec = ActionSpec.from_config(self._cfg_obj)
        self.reward_cfg: RewardConfig = RewardConfig.from_config(
            self._raw["reward"],
            enable_reward_v2=bool(self._raw["phases"].get("enable_reward_v2", False)),
        )
        self.terrain_factory = TerrainFactory(self._cfg_obj)
        self.sampler = DomainSampler.from_config(self._cfg_obj)

        policy_dt = float(self._raw["sim"]["dt"]) * int(self._raw["sim"]["decimation"])
        self.reward_cfg.warn_on_degenerate_weights(
            episode_length_s=float(self._raw["env"]["episode_length_s"]),
            policy_dt=policy_dt,
            max_goal_distance_m=self.obs_spec.max_distance_m,
        )
        self.policy_dt = policy_dt

        # Eval override: when set, every reset uses these params instead of
        # sampling. This is how the OOD harness pins a grid point, and it means
        # training and evaluation share ONE reset code path.
        self._forced_params: Optional[EpisodeParams] = None

        super().__init__(cfg, render_mode, **kwargs)

        # Buffers are allocated AFTER super().__init__ because they need
        # num_envs and device, which the base class resolves.
        #
        # VERIFY ON A100: some Isaac Lab versions call `_reset_idx` (and thus
        # `_get_observations`) from inside `DirectRLEnv.__init__`, which would
        # run before these buffers exist and raise AttributeError on
        # `self.goal_pos`. If that happens, split _init_buffers into a
        # `_preallocate()` called before super().__init__ using
        # `cfg.scene.num_envs` and `cfg.sim.device` directly.
        self._init_buffers()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        """Build robot, sensors, ground, obstacle pool, and payload."""
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        # Only the legacy "raycaster" modality has a sensor prim; the default
        # analytic lidar is pure tensor math over the TerrainSpec.
        self.ray_caster = None
        if self.cfg.ray_caster is not None:
            self.ray_caster = RayCaster(self.cfg.ray_caster)
            self.scene.sensors["ray_caster"] = self.ray_caster

        self._spawn_ground()
        self._spawn_obstacle_pool()

        # Clone per-env prims BEFORE spawning anything global.
        # VERIFY ON A100: `copy_from_source=False` is the fast path but requires
        # that per-env prims are never structurally edited afterwards -- which is
        # exactly why obstacles are a fixed-size pool that gets re-POSED rather
        # than re-created.
        self.scene.clone_environments(copy_from_source=False)

        self._spawn_payload()

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_ground(self) -> None:
        """Spawn the per-env ground patch.

        The patch is a thin kinematic cuboid rather than a shared ground plane so
        each env's SLOPE can be set independently on reset by writing its
        orientation. A single shared plane would force one slope for all envs,
        which would destroy per-env randomization of the terrain axis.

        VERIFY ON A100: confirm a kinematic RigidObject's pose can be written
        per-env at reset and that ray casts still hit it. If the raycaster
        reports nothing, check that '/World/envs/env_.*/Ground' appears in the
        RayCasterCfg.mesh_prim_paths list.
        """
        size = float(self._raw["env"]["terrain_size_m"])
        self._ground_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Ground",
            spawn=sim_utils.CuboidCfg(
                size=(size, size, 0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.7, dynamic_friction=0.63, restitution=0.0
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        )
        self.ground = RigidObject(self._ground_cfg)
        self.scene.rigid_objects["ground"] = self.ground

    def _spawn_obstacle_pool(self) -> None:
        """Spawn a FIXED pool of obstacle prims per env.

        Layout varies per episode by re-posing pool members; unused ones are
        parked below the floor. Isaac Lab's env cloning requires a structurally
        identical prim graph across envs, so a variable obstacle COUNT is not
        available -- but a variable POSE is, which is enough for procedural
        layouts.
        """
        self.max_obstacles = int(self._raw["terrain"]["max_obstacles_per_env"])
        # Slot k is spawned with the SAME radius TerrainFactory assigns to any
        # obstacle it binds to slot k -- one shared function, so sim geometry,
        # the A* occupancy grid, and the analytic lidar cannot disagree.
        slot_radii = pool_slot_radii(
            tuple(self._raw["terrain"]["obstacle_radius_range_m"]), self.max_obstacles
        )
        height = float(self._raw["terrain"]["obstacle_height_m"])

        self.obstacles: List[RigidObject] = []
        for i in range(self.max_obstacles):
            obstacle_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Obstacles/Obstacle_{i}",
                spawn=sim_utils.CylinderCfg(
                    radius=float(slot_radii[i]),
                    height=height,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.2, 0.2)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -50.0)),
            )
            obstacle = RigidObject(obstacle_cfg)
            self.scene.rigid_objects[f"obstacle_{i}"] = obstacle
            self.obstacles.append(obstacle)

        # NOTE: radii are fixed PER SLOT at spawn (no runtime rescaling -- PhysX
        # cannot rescale a cloned collision shape per env). TerrainFactory draws
        # obstacles from these slots (Obstacle.slot), which is what keeps the sim
        # consistent with the solvability guarantee.
        # VERIFY ON ARC: each env_.*/Obstacles/Obstacle_k prim has radius
        # slot_radii[k] (print a few from the stage after cloning).

    def _spawn_payload(self) -> None:
        """Spawn the payload according to ``payload.attach_mode``."""
        payload_cfg = self._raw["payload"]
        self.payload_mode = str(payload_cfg["attach_mode"])
        self.payload_enabled = bool(payload_cfg["enabled"])
        self.payload = None

        if not self.payload_enabled or self.payload_mode == "mass_modifier":
            return

        from env.payload import build_payload_cfg

        spec = PayloadSpec.from_config(
            self._cfg_obj, mass_kg=float(self._raw["domain"]["train"]["payload_mass_kg"][0])
        )
        self.payload = RigidObject(build_payload_cfg(spec, "/World/envs/env_.*/Payload"))
        self.scene.rigid_objects["payload"] = self.payload
        # VERIFY ON A100: the FixedJoint weld (env/payload.py::attach_payload_joint)
        # must be created on the stage after cloning. If the payload lags or
        # falls through the chassis, that joint is the first thing to check.

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------
    def _init_buffers(self) -> None:
        """Allocate per-env state. All persistent state is a GPU tensor."""
        n, device = self.num_envs, self.device

        self.goal_pos = torch.zeros(n, 2, device=device)
        self.start_pos = torch.zeros(n, 2, device=device)
        self.prev_distance = torch.zeros(n, device=device)
        self.curr_distance = torch.zeros(n, device=device)
        self.optimal_path_length = torch.ones(n, device=device)
        self.actual_path_length = torch.zeros(n, device=device)
        self.prev_xy = torch.zeros(n, 2, device=device)

        self.payload_mass = torch.ones(n, device=device)
        self.friction = torch.ones(n, device=device)
        self.slope_deg = torch.zeros(n, device=device)
        self.depth_dropout = torch.zeros(n, device=device)
        self.depth_noise = torch.zeros(n, device=device)

        self.prev_actions = torch.zeros(n, self.action_spec.dim, device=device)
        self.prev_lin_vel = torch.zeros(n, device=device)

        self.reached_goal = torch.zeros(n, dtype=torch.bool, device=device)
        self.collided = torch.zeros(n, dtype=torch.bool, device=device)
        self.outcome = torch.zeros(n, dtype=torch.long, device=device)

        self.goal_tolerance = float(self._raw["env"]["goal_tolerance_m"])
        self.collision_threshold = float(self._raw["env"]["collision_force_threshold"])
        self.max_range_m = float(self._raw["sensors"]["raycaster"]["max_range_m"])

        # Analytic lidar state: per-env obstacle table indexed by pool slot, so
        # row k describes exactly the sim prim Obstacle_k.
        self.sensor_modality = str(self._raw["sensors"]["modality"])
        self.obstacle_xyr = torch.zeros(n, self.max_obstacles, 3, device=device)
        self.obstacle_valid = torch.zeros(n, self.max_obstacles, dtype=torch.bool, device=device)
        if self.sensor_modality == "analytic_lidar":
            self.lidar_spec = LidarSpec.from_config(self._cfg_obj)
            self.lidar_angles = torch.tensor(
                self.lidar_spec.ray_angles(), dtype=torch.float32, device=device
            )

        # Per-env CPU-side terrain specs (NumPy layouts + ground-truth path lengths).
        self._terrain_specs: List[Optional[TerrainSpec]] = [None] * n

        # Completed episodes, drained by the eval harness.
        self._completed: List[EpisodeRecord] = []
        self._episode_steps = torch.zeros(n, dtype=torch.long, device=device)

        self._reward_components: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store and rescale the policy's action into physical velocity commands."""
        self.raw_actions = actions.clone()
        self.commands = self._scale_actions_torch(actions)

    def _scale_actions_torch(self, actions: torch.Tensor) -> torch.Tensor:
        """GPU mirror of ``env.spaces.scale_action`` -- keep the two in sync."""
        raw = torch.clamp(actions, -1.0, 1.0) * self.action_spec.scale
        out = torch.empty_like(raw)
        for i, (lo, hi) in enumerate((self.action_spec.lin_vel_range, self.action_spec.ang_vel_range)):
            mid, half = (hi + lo) / 2.0, (hi - lo) / 2.0
            out[:, i] = torch.clamp(mid + raw[:, i] * half, lo, hi)
        return out

    def _apply_action(self) -> None:
        """Convert (v, omega) to wheel speeds and write them to the articulation.

        VERIFY ON A100: joint indices, and whether the built-in robot expects a
        VELOCITY target (``set_joint_velocity_target``) or an effort command. A
        JetBot driven with the wrong actuator mode will simply not move -- check
        this first during Phase 0 bring-up.
        """
        v, omega = self.commands[:, 0], self.commands[:, 1]
        robot_cfg = self._raw["robot"]
        wheel_radius = float(robot_cfg["wheel_radius_m"])
        wheel_base = float(robot_cfg["wheel_base_m"])
        max_speed = float(robot_cfg["max_wheel_speed_rad_s"])

        left = (v - omega * wheel_base / 2.0) / wheel_radius
        right = (v + omega * wheel_base / 2.0) / wheel_radius

        # Scale both wheels together so the commanded turn radius is preserved
        # under saturation (see env.spaces.differential_drive_wheel_speeds).
        peak = torch.maximum(left.abs(), right.abs())
        factor = torch.where(peak > max_speed, max_speed / peak.clamp(min=1e-6), torch.ones_like(peak))
        wheel_targets = torch.stack([left * factor, right * factor], dim=-1)

        self.robot.set_joint_velocity_target(wheel_targets, joint_ids=self._wheel_joint_ids)

    @property
    def _wheel_joint_ids(self) -> List[int]:
        """Resolve wheel joint indices by name, once.

        VERIFY ON A100: joint names must match the USD exactly. Print
        ``self.robot.joint_names`` during Phase 0 bring-up and reconcile with
        ``robot.left_wheel_joint`` / ``right_wheel_joint`` in robot.yaml.
        """
        if not hasattr(self, "_cached_wheel_ids"):
            names = [self._raw["robot"]["left_wheel_joint"], self._raw["robot"]["right_wheel_joint"]]
            ids, _ = self.robot.find_joints(names)
            if len(ids) != 2:
                raise RuntimeError(
                    f"Expected 2 wheel joints {names}, found {ids}. "
                    f"Available joints: {self.robot.joint_names}"
                )
            self._cached_wheel_ids = ids
        return self._cached_wheel_ids

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Assemble the batched observation.

        MUST match ``env.spaces.build_observation`` component-for-component --
        that function is what the Nav2 baseline and the tests use, and a
        divergence here would feed a trained checkpoint scrambled inputs.
        """
        parts: List[torch.Tensor] = []
        spec = self.obs_spec

        for name in spec.components:
            if name == "depth":
                parts.append(self._get_depth_obs())
            elif name == "goal_pose":
                parts.append(self._get_goal_pose_obs())
            elif name == "proprioception":
                parts.append(self._get_proprioception_obs())
            elif name == "payload_mass":
                mass = self.payload_mass.unsqueeze(-1)
                parts.append(mass / spec.max_payload_mass_kg if spec.normalize else mass)

        obs = torch.cat(parts, dim=-1)
        obs = torch.clamp(obs, spec.clip_range[0], spec.clip_range[1])
        obs = torch.nan_to_num(obs, nan=0.0, posinf=spec.clip_range[1], neginf=spec.clip_range[0])
        return {"policy": obs}

    def _get_depth_obs(self) -> torch.Tensor:
        """Ray distances, with Phase 4 sensor degradation applied."""
        if self.sensor_modality == "analytic_lidar":
            distances = self._analytic_lidar_ranges()
        else:
            distances = self._raycaster_ranges()

        distances = self._apply_depth_degradation_torch(distances)

        return distances / self.max_range_m if self.obs_spec.normalize else distances

    def _analytic_lidar_ranges(self) -> torch.Tensor:
        """Exact ranges from the TerrainSpec (env/lidar.py -- the unit-tested code).

        Env-local frame, same as the spec: robot xy minus the env origin. The
        lidar is mounted at the chassis origin and turns with the robot's yaw.
        """
        robot_xy = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        return cast_rays(
            torch,
            robot_xy,
            self._robot_yaw(),
            self.lidar_angles,
            self.obstacle_xyr,
            self.obstacle_valid,
            max_range_m=self.lidar_spec.max_range_m,
            half_size_m=self.lidar_spec.half_size_m,
            boundary_as_obstacle=self.lidar_spec.boundary_as_obstacle,
        )

    def _raycaster_ranges(self) -> torch.Tensor:
        """Legacy Isaac Lab RayCaster ranges (``sensors.modality: raycaster``).

        VERIFY ON A100: ``ray_caster.data.ray_hits_w`` gives world-space HIT
        POINTS, not distances -- the norm below converts them. Rays that hit
        nothing come back as ``inf`` in some versions and as a large finite
        sentinel in others; the nan_to_num guards both.
        """
        hits = self.ray_caster.data.ray_hits_w  # (num_envs, num_rays, 3)
        origins = self.ray_caster.data.pos_w.unsqueeze(1)
        distances = torch.norm(hits - origins, dim=-1)
        distances = torch.nan_to_num(distances, nan=self.max_range_m, posinf=self.max_range_m)
        return torch.clamp(distances, 0.0, self.max_range_m)

    def _apply_depth_degradation_torch(self, distances: torch.Tensor) -> torch.Tensor:
        """GPU mirror of ``randomization.apply_depth_degradation``.

        Keep the two implementations in agreement: the NumPy one is what the
        tests and the Nav2 baseline exercise, so a divergence would mean the
        baseline is evaluated under different sensor conditions than the policy
        -- which would quietly invalidate the head-to-head comparison.
        """
        if not bool(self._raw["phases"].get("enable_sensor_noise", False)):
            return distances

        noise_std = self.depth_noise.unsqueeze(-1)
        if torch.any(noise_std > 0):
            distances = distances + torch.randn_like(distances) * noise_std

        dropout_p = self.depth_dropout.unsqueeze(-1)
        if torch.any(dropout_p > 0):
            mask = torch.rand_like(distances) < dropout_p
            fill = self._raw["sensors"]["degradation"]["dropout_fill_value"]
            fill_value = 0.0 if fill == "zero" else self.max_range_m
            distances = torch.where(mask, torch.full_like(distances, fill_value), distances)

        return torch.clamp(distances, 0.0, self.max_range_m)

    def _get_goal_pose_obs(self) -> torch.Tensor:
        """(cos bearing, sin bearing, normalized distance) in the robot frame."""
        robot_xy = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        delta = self.goal_pos - robot_xy
        distance = torch.norm(delta, dim=-1)

        yaw = self._robot_yaw()
        bearing = torch.atan2(delta[:, 1], delta[:, 0]) - yaw

        norm_d = distance / self.obs_spec.max_distance_m if self.obs_spec.normalize else distance
        return torch.stack([torch.cos(bearing), torch.sin(bearing), norm_d], dim=-1)

    def _robot_yaw(self) -> torch.Tensor:
        """Extract yaw from the root quaternion.

        VERIFY ON A100: Isaac Lab quaternions are (w, x, y, z). If headings look
        rotated by 90 degrees or mirrored, this ordering is the cause.
        """
        quat = self.robot.data.root_quat_w  # (num_envs, 4) as (w, x, y, z)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _get_proprioception_obs(self) -> torch.Tensor:
        """Body-frame forward velocity and yaw rate."""
        lin_vel = self.robot.data.root_lin_vel_b[:, 0]
        ang_vel = self.robot.data.root_ang_vel_b[:, 2]
        if self.obs_spec.normalize:
            lin_vel = lin_vel / self.obs_spec.max_lin_vel
            ang_vel = ang_vel / self.obs_spec.max_ang_vel
        return torch.stack([lin_vel, ang_vel], dim=-1)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        """Compute reward via the SHARED pure implementation in env/reward.py.

        No reward logic lives here -- this method only gathers state. That is
        what makes the reward locally unit-testable, and it means the tests
        exercise the same code path the A100 runs.
        """
        robot_xy = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        self.curr_distance = torch.norm(self.goal_pos - robot_xy, dim=-1)

        self.actual_path_length += torch.norm(robot_xy - self.prev_xy, dim=-1)
        self.prev_xy = robot_xy.clone()

        self.reached_goal = self.curr_distance < self.goal_tolerance
        self.collided = self._detect_collision()

        lin_vel = self.robot.data.root_lin_vel_b[:, 0]

        state = RewardState(
            prev_distance=self.prev_distance,
            curr_distance=self.curr_distance,
            reached_goal=self.reached_goal,
            collided=self.collided,
            action=self.raw_actions,
            prev_action=self.prev_actions,
            lin_vel=lin_vel,
            prev_lin_vel=self.prev_lin_vel,
            payload_mass=self.payload_mass,
            dt=self.policy_dt,
        )

        total, components = compute_reward(state, self.reward_cfg)
        self._reward_components = components

        self.prev_distance = self.curr_distance.clone()
        self.prev_actions = self.raw_actions.clone()
        self.prev_lin_vel = lin_vel.clone()
        self._episode_steps += 1

        return total

    def _detect_collision(self) -> torch.Tensor:
        """Contact-force based collision detection (ground truth for the metric).

        VERIFY ON A100: ``net_forces_w_history`` shape is
        ``(num_envs, history_length, num_bodies, 3)``. The max over history
        catches brief impacts that a single-frame read would miss between policy
        steps -- with decimation=4, a single-frame check misses most real hits.
        """
        forces = self.contact_sensor.data.net_forces_w_history
        magnitudes = torch.norm(forces, dim=-1)
        peak = magnitudes.max(dim=1)[0].max(dim=-1)[0]
        return peak > self.collision_threshold

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(terminated, truncated)``.

        Isaac Lab separates the two so bootstrapping is correct: a TIMEOUT is a
        truncation (the value function should still bootstrap from the final
        state), while success and collision are genuine terminations. Reporting
        a timeout as terminated would teach the critic that time running out is
        an absorbing state worth zero -- a subtle and quite damaging bug.
        """
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.reached_goal | self.collided

        self.outcome = torch.where(
            self.reached_goal,
            torch.full_like(self.outcome, OUTCOME_SUCCESS),
            torch.where(
                self.collided,
                torch.full_like(self.outcome, OUTCOME_COLLISION),
                torch.where(
                    timed_out,
                    torch.full_like(self.outcome, OUTCOME_TIMEOUT),
                    torch.full_like(self.outcome, OUTCOME_RUNNING),
                ),
            ),
        )

        return terminated, timed_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Optional[Sequence[int]]) -> None:
        """Reset the given envs: new params, new SOLVABLE terrain, new payload."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES  # type: ignore[attr-defined]

        self._record_completed_episodes(env_ids)
        super()._reset_idx(env_ids)

        env_ids_list = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)

        specs: List[TerrainSpec] = []
        for env_id in env_ids_list:
            params = self._forced_params or self.sampler.sample()
            # Terrain generation is CPU-side NumPy and includes the A* check --
            # every spec returned here is guaranteed solvable (principle 3).
            spec = self.terrain_factory.generate(params, self.sampler.rng)
            self._terrain_specs[env_id] = spec
            specs.append(spec)

        self._write_episode_params(env_ids_list, specs)
        self._apply_terrain_to_sim(env_ids_list, specs)
        self._reset_robot_state(env_ids, specs)
        self._apply_payload_mass(env_ids, specs)
        self._reset_episode_buffers(env_ids, specs)

    def _write_episode_params(self, env_ids: List[int], specs: List[TerrainSpec]) -> None:
        """Copy sampled scalars into the per-env GPU tensors."""
        device = self.device
        idx = torch.tensor(env_ids, dtype=torch.long, device=device)

        def to_tensor(values: List[float]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.float32, device=device)

        self.payload_mass[idx] = to_tensor([s.params.payload_mass_kg for s in specs])
        self.friction[idx] = to_tensor([s.params.friction_coeff for s in specs])
        self.slope_deg[idx] = to_tensor([s.params.slope_angle_deg for s in specs])
        self.depth_dropout[idx] = to_tensor([s.params.depth_dropout_prob for s in specs])
        self.depth_noise[idx] = to_tensor([s.params.depth_noise_std for s in specs])
        self.optimal_path_length[idx] = to_tensor(
            [max(s.optimal_path_length_m, 1e-3) for s in specs]
        )
        self.goal_pos[idx] = torch.tensor(
            [s.goal_xy for s in specs], dtype=torch.float32, device=device
        )
        self.start_pos[idx] = torch.tensor(
            [s.start_xy for s in specs], dtype=torch.float32, device=device
        )

        xyr, valid = obstacles_to_arrays([s.obstacles for s in specs], self.max_obstacles)
        self.obstacle_xyr[idx] = torch.as_tensor(xyr, device=device)
        self.obstacle_valid[idx] = torch.as_tensor(valid, device=device)

    def _apply_terrain_to_sim(self, env_ids: List[int], specs: List[TerrainSpec]) -> None:
        """Push obstacle poses, ground slope, and friction into the simulator.

        VERIFY ON A100 (three separate things, all silent-failure risks):
          1. ``write_root_pose_to_sim`` on a KINEMATIC rigid body actually moves
             it. If obstacles stay parked below the floor, the robot will drive
             through empty space and collision rate will be implausibly low.
          2. Per-instance friction. Isaac Lab binds physics materials at SPAWN
             time; changing friction per-env per-episode may require
             ``root_physx_view.set_material_properties``. If friction never
             changes, the friction OOD axis is inert -- and the resulting flat
             robustness curve looks like an exciting "policy is friction-robust"
             finding rather than the bug it is. Check this one first.
          3. Slot binding: prim Obstacle_k receives the spec obstacle with
             ``slot == k`` (its radius was fixed to match at spawn).
        """
        device = self.device
        # Obstacles are bound to pool slots by TerrainFactory (Obstacle.slot);
        # each prim must receive the obstacle whose radius it was spawned with.
        slot_maps = [{o.slot: o for o in spec.obstacles} for spec in specs]

        for slot, obstacle in enumerate(self.obstacles):
            poses: List[List[float]] = []
            active_ids: List[int] = []
            for env_id, spec, by_slot in zip(env_ids, specs, slot_maps):
                origin = self.scene.env_origins[env_id]
                obs = by_slot.get(slot)
                if obs is not None:
                    poses.append(
                        [
                            float(origin[0]) + obs.x,
                            float(origin[1]) + obs.y,
                            float(origin[2]) + spec.height_at(obs.x, obs.y) + obs.height / 2.0,
                            1.0, 0.0, 0.0, 0.0,
                        ]
                    )
                else:
                    # Park unused pool members far below the floor.
                    poses.append([float(origin[0]), float(origin[1]), -50.0, 1.0, 0.0, 0.0, 0.0])
                active_ids.append(env_id)

            if poses:
                pose_tensor = torch.tensor(poses, dtype=torch.float32, device=device)
                idx = torch.tensor(active_ids, dtype=torch.long, device=device)
                obstacle.write_root_pose_to_sim(pose_tensor, env_ids=idx)

        self._apply_ground_slope(env_ids, specs)
        self._apply_friction(env_ids, specs)

    def _apply_ground_slope(self, env_ids: List[int], specs: List[TerrainSpec]) -> None:
        """Tilt each env's ground patch about the y-axis by its slope angle.

        VERIFY ON A100: quaternion convention is (w, x, y, z) and a rotation
        about +y makes +x uphill, matching ``TerrainSpec.height_at``. If the
        robot slides the wrong way, the sign here and in ``height_at`` disagree.
        """
        poses: List[List[float]] = []
        for env_id, spec in zip(env_ids, specs):
            origin = self.scene.env_origins[env_id]
            half_angle = math.radians(spec.params.slope_angle_deg) / 2.0
            poses.append(
                [
                    float(origin[0]), float(origin[1]), float(origin[2]) - 0.05,
                    math.cos(half_angle), 0.0, math.sin(half_angle), 0.0,
                ]
            )

        pose_tensor = torch.tensor(poses, dtype=torch.float32, device=self.device)
        idx = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        self.ground.write_root_pose_to_sim(pose_tensor, env_ids=idx)

    def _apply_friction(self, env_ids: List[int], specs: List[TerrainSpec]) -> None:
        """Write per-env ground friction.

        VERIFY ON A100: ``set_material_properties`` signature and whether the
        material index axis is per-shape. Expected shape is
        ``(num_envs, num_shapes, 3)`` as (static, dynamic, restitution).

        THIS IS THE HIGHEST-RISK SILENT FAILURE IN THE PROJECT. If friction does
        not actually change, the friction OOD sweep produces a perfectly flat
        curve that reads as a genuine robustness result. Validate it explicitly
        during bring-up: set friction to 0.05, drive at full speed, and confirm
        the robot visibly slips. Do not trust the absence of an error.
        """
        materials = [resolve_physics_material(spec.params) for spec in specs]
        props = torch.tensor(
            [[m["static_friction"], m["dynamic_friction"], m["restitution"]] for m in materials],
            dtype=torch.float32,
        ).unsqueeze(1)  # (num_envs, 1 shape, 3)

        idx = torch.tensor(env_ids, dtype=torch.long)
        try:
            self.ground.root_physx_view.set_material_properties(props.cpu(), idx.cpu())
        except (AttributeError, RuntimeError) as exc:  # pragma: no cover - A100 only
            raise RuntimeError(
                "Failed to set per-env friction. The friction study axis would be "
                "INERT and the OOD results silently meaningless. Fix this before "
                f"training. Underlying error: {exc}"
            ) from exc

    def _reset_robot_state(self, env_ids: Any, specs: List[TerrainSpec]) -> None:
        """Place the robot at each episode's verified start pose, facing the goal."""
        device = self.device
        idx = env_ids if torch.is_tensor(env_ids) else torch.tensor(env_ids, device=device)

        root_state = self.robot.data.default_root_state[idx].clone()
        origins = self.scene.env_origins[idx]

        starts = torch.tensor([s.start_xy for s in specs], dtype=torch.float32, device=device)
        goals = torch.tensor([s.goal_xy for s in specs], dtype=torch.float32, device=device)
        spawn_height = float(self._raw["robot"]["spawn_height_m"])
        heights = torch.tensor(
            [s.height_at(*s.start_xy) for s in specs], dtype=torch.float32, device=device
        )

        root_state[:, 0:2] = origins[:, :2] + starts
        root_state[:, 2] = origins[:, 2] + heights + spawn_height

        # Face the goal at spawn. Without this the policy burns the opening
        # seconds turning around, which inflates timeout rate at high slope for
        # a reason unrelated to the terrain being studied.
        delta = goals - starts
        yaw = torch.atan2(delta[:, 1], delta[:, 0])
        root_state[:, 3] = torch.cos(yaw / 2.0)   # w
        root_state[:, 4] = 0.0                     # x
        root_state[:, 5] = 0.0                     # y
        root_state[:, 6] = torch.sin(yaw / 2.0)   # z
        root_state[:, 7:] = 0.0                    # zero linear + angular velocity

        self.robot.write_root_state_to_sim(root_state, env_ids=idx)
        self.robot.reset(env_ids=idx)

    def _apply_payload_mass(self, env_ids: Any, specs: List[TerrainSpec]) -> None:
        """Set the per-env payload mass for this episode."""
        if not self.payload_enabled:
            return

        masses = torch.tensor(
            [s.params.payload_mass_kg for s in specs], dtype=torch.float32
        )
        idx = env_ids if torch.is_tensor(env_ids) else torch.tensor(env_ids)

        if self.payload_mode == "mass_modifier":
            apply_mass_modifier(
                self.robot,
                base_mass_kg=float(self._raw["robot"]["base_mass_kg"]),
                payload_masses_kg=masses,
                body_index=0,
                env_ids=idx,
            )
        else:
            set_payload_mass_batched(self.payload, masses, env_ids=idx)

    def _reset_episode_buffers(self, env_ids: Any, specs: List[TerrainSpec]) -> None:
        """Zero per-episode accumulators and seed the progress-shaping baseline."""
        device = self.device
        idx = env_ids if torch.is_tensor(env_ids) else torch.tensor(env_ids, device=device)

        starts = torch.tensor([s.start_xy for s in specs], dtype=torch.float32, device=device)
        goals = torch.tensor([s.goal_xy for s in specs], dtype=torch.float32, device=device)
        initial_distance = torch.norm(goals - starts, dim=-1)

        # Seed prev_distance with the true starting distance. Leaving it at zero
        # would make the first step's progress term a large spurious negative
        # spike -- the classic source of a reward curve that starts deeply
        # negative for no physical reason.
        self.prev_distance[idx] = initial_distance
        self.curr_distance[idx] = initial_distance
        self.prev_xy[idx] = starts
        self.actual_path_length[idx] = 0.0
        self.prev_actions[idx] = 0.0
        self.prev_lin_vel[idx] = 0.0
        self.reached_goal[idx] = False
        self.collided[idx] = False
        self.outcome[idx] = OUTCOME_RUNNING
        self._episode_steps[idx] = 0

    # ------------------------------------------------------------------
    # Episode records (for the eval harness)
    # ------------------------------------------------------------------
    def _record_completed_episodes(self, env_ids: Any) -> None:
        """Snapshot finished episodes into ``self._completed`` before the reset wipes them."""
        env_ids_list = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)

        for env_id in env_ids_list:
            spec = self._terrain_specs[env_id]
            if spec is None:
                continue  # first-ever reset: nothing completed yet
            outcome_code = int(self.outcome[env_id].item())
            if outcome_code == OUTCOME_RUNNING:
                continue  # reset for a reason other than termination

            self._completed.append(
                EpisodeRecord(
                    outcome=OUTCOME_NAMES[outcome_code],
                    params=spec.params.as_dict(),
                    optimal_path_length_m=float(spec.optimal_path_length_m),
                    actual_path_length_m=float(self.actual_path_length[env_id].item()),
                    episode_length_steps=int(self._episode_steps[env_id].item()),
                    final_distance_to_goal_m=float(self.curr_distance[env_id].item()),
                    collided=outcome_code == OUTCOME_COLLISION,
                    reached_goal=outcome_code == OUTCOME_SUCCESS,
                    timed_out=outcome_code == OUTCOME_TIMEOUT,
                )
            )

    def drain_completed_episodes(self) -> List[EpisodeRecord]:
        """Return and clear all completed episode records."""
        records, self._completed = self._completed, []
        return records

    # ------------------------------------------------------------------
    # Eval hooks
    # ------------------------------------------------------------------
    def set_forced_params(self, params: Optional[EpisodeParams]) -> None:
        """Pin every subsequent reset to fixed params (or resume sampling if None).

        This is how the OOD harness holds a grid point. Training and evaluation
        therefore share one reset path, differing only in where the parameters
        come from -- which is what makes the RL-vs-Nav2 comparison honest.
        """
        self._forced_params = params

    def get_reward_components(self) -> Dict[str, torch.Tensor]:
        """Per-term reward breakdown from the last step, for W&B logging."""
        return self._reward_components

    def get_episode_extras(self) -> Dict[str, float]:
        """Scalar diagnostics logged every step during training."""
        return {
            "episode/mean_distance_to_goal": float(self.curr_distance.mean().item()),
            "episode/mean_payload_mass": float(self.payload_mass.mean().item()),
            "episode/mean_friction": float(self.friction.mean().item()),
            "episode/mean_slope_deg": float(self.slope_deg.mean().item()),
            "episode/collision_fraction": float(self.collided.float().mean().item()),
            "episode/success_fraction": float(self.reached_goal.float().mean().item()),
        }
