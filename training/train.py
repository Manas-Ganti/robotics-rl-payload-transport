"""Training entrypoint: config -> env -> PPO -> W&B -> checkpoints.

Run on VT ARC via ``arc/train.slurm`` (see setup_notes.md), which invokes:

    python training/train.py --config configs/train.yaml --headless

    # Phase 2 (slope axis) without editing any module:
    python training/train.py --config configs/train.yaml --headless \\
        --set phases.enable_slope=true

    # Phase 5 (inferred payload) -- the told-vs-inferred experiment is one flag:
    python training/train.py --config configs/train.yaml --headless \\
        --set phases.enable_inferred_payload=true

IMPORTANT ORDERING CONSTRAINT
-----------------------------
Isaac Sim's ``AppLauncher`` MUST run before any ``isaaclab.*`` import. That is
why the env import sits inside :func:`main` rather than at module scope --
moving it to the top produces a confusing "Isaac Sim not initialized" crash.
This is a hard requirement of Isaac Sim, not a style choice.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the repo root importable when run as `python training/train.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a payload-transport navigation policy in Isaac Lab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Training config")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--num-envs", type=int, default=None, help="Override env.num_envs")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max_iterations")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path, 'latest' (must exist), or 'auto' (resume if this run "
        "has a checkpoint, else start fresh -- use with a fixed --run-name on SLURM)",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Run dir + W&B run name")
    parser.add_argument(
        "--seed-from-array",
        action="store_true",
        help="Under a SLURM job array: seed = config seed + SLURM_ARRAY_TASK_ID, and "
        "'_s<seed>' is appended to --run-name. One submit -> N seeds in parallel.",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config key by dotted path, e.g. --set phases.enable_slope=true",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Phase 0 bring-up: run random actions for a few hundred steps and exit. "
        "Verifies the robot spawns, senses, and moves before committing to a long run.",
    )

    # Isaac Sim's own args (--headless, --device, --livestream, ...).
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    except ImportError:
        # Allows --help locally without Isaac installed.
        parser.add_argument("--headless", action="store_true", help="Run without a GUI")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config overrides
# ---------------------------------------------------------------------------
def apply_overrides(data: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``--set a.b.c=value`` overrides to a config dict.

    Keeps the A100 loop fast: sweeping a hyperparameter or flipping a phase flag
    never requires editing (or forgetting to revert) a YAML file. Overrides are
    logged to W&B so a run remains reproducible from config + seed + overrides.
    """
    import yaml

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"--set expects KEY=VALUE, got {override!r}")
        key, _, raw_value = override.partition("=")

        # YAML-parse the value so types (bool/int/float/list) come out right.
        value = yaml.safe_load(raw_value)

        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node:
                raise KeyError(f"--set key '{key}' has unknown path segment '{part}'")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(
                f"--set key '{key}' does not exist in the config. "
                "Overrides may only change existing keys, so a typo cannot "
                "silently introduce a setting that nothing reads."
            )
        node[parts[-1]] = value
        print(f"[override] {key} = {value!r}")

    return data


def set_global_seed(seed: int) -> None:
    """Seed every RNG for reproducibility (CLAUDE.md principle 6)."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # ---- 1. Launch Isaac Sim BEFORE importing anything isaaclab.* ----------
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ---- 2. Now Isaac-dependent imports are safe --------------------------
    import torch

    from env.config import Config, load_train_config, validate_train_config
    from env.nav_env import TransportNavEnv, build_env_cfg
    from training.ppo_config import build_runner, resolve_checkpoint_path, wrap_env_for_library

    # ---- 3. Load + override config ---------------------------------------
    cfg = load_train_config(args.config)
    data = cfg.to_dict()
    data = apply_overrides(data, args.set)

    if args.seed is not None:
        data["seed"] = args.seed
    if args.num_envs is not None:
        data["env"]["num_envs"] = args.num_envs
    if args.max_iterations is not None:
        data["algo"][data["algo"]["name"]]["max_iterations"] = args.max_iterations
    if args.run_name is not None:
        data["experiment"]["run_name"] = args.run_name
    if args.no_wandb:
        data["logging"]["wandb"]["enabled"] = False
    if args.seed_from_array:
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id is None:
            raise SystemExit("--seed-from-array given but SLURM_ARRAY_TASK_ID is unset (not an array job)")
        data["seed"] = int(data["seed"]) + int(task_id)
        base = data["experiment"]["run_name"] or "run"
        data["experiment"]["run_name"] = f"{base}_s{data['seed']}"
    if args.resume == "auto" and not data["experiment"]["run_name"]:
        # Without a fixed name every submission gets a fresh timestamped dir, so
        # 'auto' could never find the previous attempt's checkpoints.
        raise SystemExit("--resume auto requires --run-name (a stable run directory)")

    # Re-validate AFTER overrides -- an override is exactly how an invalid
    # config sneaks in, so the assertions must see the final values.
    validate_train_config(data)
    cfg = Config(data, source=args.config)

    set_global_seed(int(data["seed"]))

    # ---- 4. Build the environment ----------------------------------------
    print("\n=== Building environment ===")
    env_cfg = build_env_cfg(cfg)
    env = TransportNavEnv(env_cfg)

    print(env.obs_spec.describe())
    print(f"action_dim = {env.action_spec.dim}")
    print(f"num_envs   = {env.num_envs}")
    print(f"device     = {env.device}")

    # ---- 5. Phase 0 smoke test -------------------------------------------
    if args.smoke_test:
        run_smoke_test(env)
        env.close()
        simulation_app.close()
        return

    # ---- 6. Logging -------------------------------------------------------
    log_dir = setup_logging(cfg, data)

    # ---- 7. Train ---------------------------------------------------------
    wrapped_env = wrap_env_for_library(env, cfg)
    runner = build_runner(wrapped_env, cfg, log_dir=log_dir, device=str(env.device))

    # 'latest'/'auto' search THIS run's directory only -- never another run's.
    checkpoint = resolve_checkpoint_path(log_dir, args.resume)
    if checkpoint:
        print(f"Resuming from {checkpoint}")
        runner.load(checkpoint)
    elif args.resume == "auto":
        print("No checkpoint in this run dir -- starting fresh.")

    # rsl_rl's learn() runs N iterations MORE than the loaded one, so a resumed
    # job must ask only for what is left, not the full budget again.
    # VERIFY ON ARC: OnPolicyRunner.load restores `current_learning_iteration`.
    max_iterations = int(data["algo"][data["algo"]["name"]]["max_iterations"])
    done = int(getattr(runner, "current_learning_iteration", 0))
    remaining = max(0, max_iterations - done)
    print(f"\n=== Training {remaining} iterations ({done}/{max_iterations} done) ===")
    if remaining > 0:
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)

    final_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_path)
    print(f"\nSaved final checkpoint: {final_path}")

    env.close()
    simulation_app.close()


def setup_logging(cfg: Any, data: Dict[str, Any]) -> str:
    """Create the run directory, write the config sidecar, and init W&B.

    The full resolved config is written NEXT TO the checkpoints. Without that,
    a checkpoint six weeks old is unusable -- you cannot know which ranges,
    reward weights, or phase flags produced it, and the observation layout it
    expects becomes guesswork.
    """
    import json
    from datetime import datetime

    experiment = data["experiment"]
    run_name = experiment.get("run_name") or (
        f"{datetime.now():%Y%m%d-%H%M%S}_seed{data['seed']}"
    )

    log_dir = os.path.join(data["logging"]["log_dir"], run_name)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "resolved_config.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    wandb_cfg = data["logging"]["wandb"]
    if wandb_cfg["enabled"]:
        try:
            import wandb

            wandb.init(
                project=wandb_cfg["project"],
                entity=wandb_cfg["entity"],
                name=run_name,
                # Stable id: a walltime-resumed SLURM job appends to the same
                # W&B run instead of starting a disconnected one.
                id=run_name,
                resume="allow",
                mode=wandb_cfg["mode"],
                config=data,
                notes=experiment.get("notes", ""),
                tags=experiment.get("tags", []),
                dir=log_dir,
            )
            print(f"W&B run: {run_name}")
        except ImportError:
            print("W&B not installed; continuing without it.")

    print(f"Logging to {log_dir}")
    return log_dir


def _robot_footprint_radius(prim_path: str = "/World/envs/env_0/Robot") -> Optional[float]:
    """Half-diagonal of env_0's robot x-y bounding box (m), from the live stage.

    Compared against solvability.robot_radius_m: the A* check inflates
    obstacles by that radius, so a wider robot makes "solvable" a lie.
    """
    try:
        import math

        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide]
        )
        box = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
        size = box.GetMax() - box.GetMin()
        return 0.5 * math.hypot(size[0], size[1]) * UsdGeom.GetStageMetersPerUnit(stage)
    except Exception as exc:  # diagnostic only -- never fail the smoke test on it
        print(f"  (footprint check unavailable: {exc})")
        return None


def run_smoke_test(env: Any, num_steps: int = 300) -> None:
    """Phase 0 bring-up: random actions, then report what actually happened.

    VERIFY ON A100 -- this is the checklist Phase 0 exists to satisfy:
      1. the robot SPAWNS (no NaN positions, sane heights)
      2. the robot SENSES (depth is not uniformly max_range -- that means the
         raycaster is missing its target meshes)
      3. the robot MOVES (nonzero displacement under random actions)
      4. terrain, friction, and payload actually VARY across envs

    Check 4 matters most. Several Isaac Lab writes fail silently, and an inert
    randomization axis yields a flat, healthy-looking robustness curve that is
    entirely an artifact. Print the spread and confirm it is nonzero.
    """
    import torch

    print(f"\n=== SMOKE TEST: {num_steps} random-action steps ===")
    print(f"  joints : {env.robot.joint_names}")
    print(f"  bodies : {env.robot.body_names}")
    for name, act in env.robot.actuators.items():
        print(
            f"  actuator[{name}] joints={act.joint_names} stiffness={act.stiffness[0].tolist()} "
            f"damping={act.damping[0].tolist()} effort_limit={act.effort_limit[0].tolist()}"
        )
    footprint = _robot_footprint_radius()
    robot_radius_cfg = float(env._raw["solvability"]["robot_radius_m"])
    if footprint is not None:
        print(f"  footprint radius  : {footprint:.3f} m (solvability.robot_radius_m = {robot_radius_cfg:.3f})")
    obs, _ = env.reset()

    start_pos = env.robot.data.root_pos_w[:, :2].clone()
    depth_samples: List[float] = []

    first_step_terms = 0
    for step in range(num_steps):
        actions = torch.rand(env.num_envs, env.action_spec.dim, device=env.device) * 2.0 - 1.0
        obs, rewards, terminated, truncated, info = env.step(actions)
        if step == 0:
            # Ground contact miscounted as collision would end ~every env here;
            # under random actions, collisions later on are expected and fine.
            first_step_terms = int(terminated.sum().item())

        if step % 50 == 0:
            policy_obs = obs["policy"] if isinstance(obs, dict) else obs
            if env.obs_spec.has("depth"):
                depth = policy_obs[:, env.obs_spec.slice_of("depth")]
                depth_samples.append(float(depth.mean().item()))
            print(
                f"  step {step:4d} | reward {rewards.mean().item():+7.3f} "
                f"| term {terminated.sum().item():4d} | trunc {truncated.sum().item():4d}"
            )

    displacement = torch.norm(env.robot.data.root_pos_w[:, :2] - start_pos, dim=-1)

    print("\n--- SMOKE TEST RESULTS ---")
    print(f"  mean displacement : {displacement.mean().item():.3f} m")
    print(f"  max  displacement : {displacement.max().item():.3f} m")
    if depth_samples:
        print(f"  mean depth (norm) : {sum(depth_samples) / len(depth_samples):.3f}")
    print(f"  payload spread    : {env.payload_mass.min().item():.2f} - {env.payload_mass.max().item():.2f} kg")
    print(f"  friction spread   : {env.friction.min().item():.2f} - {env.friction.max().item():.2f}")
    print(f"  slope spread      : {env.slope_deg.min().item():.2f} - {env.slope_deg.max().item():.2f} deg")

    # Sim-vs-spec obstacle agreement. The analytic lidar reads the SPEC, so if
    # the sim silently failed to move obstacles the lidar would still "see"
    # them -- the policy would learn to dodge ghosts while driving through
    # empty space. Compare every active prim's actual pose with the spec.
    # Skip envs reset on the last step: their prim poses were written this step
    # and may not be reflected in `.data` until the next physics update.
    settled = env.episode_length_buf > 1
    pose_err = 0.0
    for slot, obstacle in enumerate(env.obstacles):
        valid = env.obstacle_valid[:, slot] & settled
        if not bool(valid.any()):
            continue
        sim_xy = obstacle.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
        err = torch.norm(sim_xy - env.obstacle_xyr[:, slot, :2], dim=-1)[valid]
        pose_err = max(pose_err, float(err.max().item()))
    print(f"  obstacle pose err : {pose_err:.3f} m (max |sim - spec| over active obstacles)")

    # Physical readback: what PhysX actually holds, not what we meant to write.
    mats = env.ground.root_physx_view.get_material_properties()
    phys_fric = mats[:, 0, 0]
    print(f"  friction (PhysX)  : {phys_fric.min().item():.2f} - {phys_fric.max().item():.2f}")
    phys_mass = None
    if getattr(env, "payload_mode", "") == "mass_modifier" and hasattr(env, "_chassis_body_id"):
        masses = env.robot.root_physx_view.get_masses()[:, env._chassis_body_id]
        base = env._default_body_masses[0, env._chassis_body_id].item()
        phys_mass = masses
        print(
            f"  chassis mass      : {masses.min().item():.2f} - {masses.max().item():.2f} kg "
            f"(USD chassis {base:.2f} kg + payload)"
        )
    # step() auto-resets finished envs, wiping their live flags; the env
    # snapshots each outcome into EpisodeRecords just before that reset.
    records = env.drain_completed_episodes()
    ended = len(records)
    collided = sum(r.collided for r in records)
    print(f"  episodes ended    : {ended} ({collided} by collision)")

    # ---- kinematics probe: does "forward" go forward, "left" turn left? -----
    # A wrong forward axis, swapped wheels, or a flipped wheel sign all still
    # "move" under random actions -- only commanded, isolated motions show them.
    obs, _ = env.reset()
    z_spawn = env.robot.data.root_pos_w[:, 2].clone()
    idle = torch.zeros(env.num_envs, env.action_spec.dim, device=env.device)
    max_tilt = 0.0
    for _ in range(25):   # 0.5 s, no command: does it land and sit level?
        env.step(idle)
        g = env.robot.data.projected_gravity_b  # (N, 3); (0,0,-1) when level
        tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)))
        max_tilt = max(max_tilt, float(tilt.max().item()))
    drop = float((z_spawn - env.robot.data.root_pos_w[:, 2]).mean().item())
    print(f"  spawn settle      : dropped {drop:+.3f} m, max tilt {max_tilt:.1f} deg (0.5 s idle)")

    obs, _ = env.reset()
    yaw0 = env._robot_yaw().clone()
    xy0 = env.robot.data.root_pos_w[:, :2].clone()
    cmd = torch.zeros(env.num_envs, env.action_spec.dim, device=env.device)
    cmd[:, 0] = 1.0                                   # full forward, no turn
    vx_sum = vy_sum = 0.0
    wheel_ids = env._wheel_joint_ids
    for i in range(50):
        env.step(cmd)
        vx_sum += float(env.robot.data.root_lin_vel_b[:, 0].mean().item())
        vy_sum += float(env.robot.data.root_lin_vel_b[:, 1].abs().mean().item())
        if i == 49:   # steady state: achieved vs targeted wheel speed
            w_act = env.robot.data.joint_vel[:, wheel_ids].abs().mean().item()
            w_tgt = env._last_wheel_targets.abs().mean().item()
    move = env.robot.data.root_pos_w[:, :2] - xy0
    heading = torch.stack([torch.cos(yaw0), torch.sin(yaw0)], dim=-1)
    along = float((move * heading).sum(-1).mean().item())
    lateral = float((move[:, 0] * heading[:, 1] - move[:, 1] * heading[:, 0]).abs().mean().item())
    cmd_v = float(env.commands[:, 0].mean().item())
    print(f"  wheel tracking    : |target| {w_tgt:.2f} rad/s -> |achieved| {w_act:.2f} rad/s "
          f"(ratio {w_act / max(w_tgt, 1e-6):.2f}; low -> motor too weak, ~1 but slow -> slip/radius)")
    print(f"  forward probe     : cmd {cmd_v:.2f} m/s -> body vx {vx_sum / 50:.2f}, |vy| {vy_sum / 50:.2f} m/s; "
          f"moved {along:+.2f} m along heading, {lateral:.2f} m sideways (1 s)")

    obs, _ = env.reset()
    cmd[:, 0], cmd[:, 1] = 0.0, 1.0                   # turn left in place
    wz_sum = 0.0
    for _ in range(50):
        env.step(cmd)
        wz_sum += float(env.robot.data.root_ang_vel_b[:, 2].mean().item())
    cmd_w = float(env.commands[:, 1].mean().item())
    print(f"  turn probe        : cmd {cmd_w:+.2f} rad/s -> body wz {wz_sum / 50:+.2f} rad/s")

    # ---- directed phase: drive full speed at the goal for 10 s --------------
    # Random actions rarely end an episode in 6 s, so they cannot show that
    # collision detection works. Straight at the goal through obstacles, some
    # robots MUST touch one. Contact hits and edge exits are counted apart.
    obs, _ = env.reset()
    env.drain_completed_episodes()   # discard probe-phase episodes
    contact0, oob0 = int(env.contact_hit_count.item()), int(env.out_of_bounds_count.item())
    forward = torch.zeros(env.num_envs, env.action_spec.dim, device=env.device)
    for _ in range(500):
        yaw_err = torch.atan2(obs["policy"][:, env.obs_spec.slice_of("goal_pose")][:, 1],
                              obs["policy"][:, env.obs_spec.slice_of("goal_pose")][:, 0])
        forward[:, 0] = 1.0
        forward[:, 1] = torch.clamp(2.0 * yaw_err, -1.0, 1.0)  # steer at the goal
        obs, *_ = env.step(forward)
    directed = env.drain_completed_episodes()
    d_contact = int(env.contact_hit_count.item()) - contact0
    d_oob = int(env.out_of_bounds_count.item()) - oob0
    d_success = sum(r.reached_goal for r in directed)
    print(
        f"  directed drive    : {len(directed)} ended | {d_success} success | "
        f"{d_contact} obstacle contacts | {d_oob} edge exits"
    )

    hit_frac = None
    if env.obs_spec.has("depth"):
        final_depth = obs["policy"] if isinstance(obs, dict) else obs
        final_depth = final_depth[:, env.obs_spec.slice_of("depth")]
        hit_frac = float((final_depth < 0.999).float().mean().item())
        print(f"  rays hitting      : {100.0 * hit_frac:.1f}% (normalized range < max)")

    print("\n--- INTERPRET ---")
    if displacement.mean().item() < 0.01:
        print("  FAIL: robot did not move. Check wheel joint names and the actuator")
        print("        mode in env/nav_env.py::_apply_action (velocity vs effort).")
    if depth_samples and abs(sum(depth_samples) / len(depth_samples) - 1.0) < 1e-3:
        print("  FAIL: depth is saturated at max range -- the raycaster is hitting")
        print("        nothing. Check RayCasterCfg.mesh_prim_paths in build_env_cfg.")
    if pose_err > 0.05:
        print("  FAIL: sim obstacle poses disagree with the spec -- write_root_pose_to_sim")
        print("        is not moving the kinematic obstacles (_apply_terrain_to_sim). The")
        print("        analytic lidar would report ghosts. Tier 1, setup_notes.md.")
    if hit_frac is not None and hit_frac < 0.01:
        print("  FAIL: almost no ray hits anything. With obstacles present this means")
        print("        the sensor is blind (check sensors.modality and obstacle_valid).")
    if float(phys_fric.max() - phys_fric.min()) < 1e-6:
        print("  FAIL: PhysX ground friction is identical across envs -- the write in")
        print("        _apply_friction did not land. The friction axis is INERT.")
    if phys_mass is not None and float(phys_mass.max() - phys_mass.min()) < 1e-6:
        print("  FAIL: PhysX chassis mass is identical across envs -- the payload axis")
        print("        is INERT (apply_mass_modifier did not land).")
    if first_step_terms > 0.5 * env.num_envs:
        print("  FAIL: most envs terminated on the very first step. That is ground")
        print("        contact being counted as collision -- check the filtered contact")
        print("        sensor (force_matrix_w) in _detect_collision.")
    if footprint is not None and footprint > robot_radius_cfg + 0.05:
        print("  FAIL: the robot is wider than solvability.robot_radius_m -- the A* check")
        print("        certifies gaps the robot cannot fit through (unfair collisions).")
        print("        Raise robot_radius_m (and nav2 costmap radius) to the footprint.")
    if max_tilt > 30.0 or drop < -0.05:
        print("  FAIL: robot tips or is ejected upward at spawn -- robot.spawn_height_m is")
        print("        below the root's height above the wheels (see arc/inspect_usd.py).")
    if w_act / max(w_tgt, 1e-6) < 0.7:
        print("  FAIL: wheels reach < 70% of their target speed -- the wheel actuator is")
        print("        too weak for the load (robot.actuator damping / effort_limit).")
    if along < 0.2:
        print("  FAIL: full-forward for 1 s moved < 0.2 m along the heading. Negative ->")
        print("        wheel sign flipped; ~0 with motion sideways -> the USD's forward")
        print("        axis is not +x (fix the yaw offset in _robot_yaw / spawn).")
    if lateral > abs(along):
        print("  FAIL: the robot moves more sideways than forward -- forward axis mismatch.")
    if wz_sum / 50 * cmd_w < 0:
        print("  FAIL: turn command and measured yaw rate have opposite signs -- left and")
        print("        right wheel joints are swapped (robot.left/right_wheel_joint).")
    if d_contact == 0:
        print("  FAIL: 500 steps driving straight at the goal through obstacles and not")
        print("        one obstacle contact registered -- the filtered contact sensor is")
        print("        dead (check the 'Filter pattern' error in the log, _detect_collision).")
    if len(directed) > 0 and d_oob / len(directed) > 0.5:
        print("  FAIL: most directed episodes left the patch -- the start/goal frame or")
        print("        the goal-bearing observation is likely wrong.")
    if float(env.friction.max() - env.friction.min()) < 1e-6:
        print("  FAIL: friction is identical across envs. The friction axis is INERT;")
        print("        its OOD curve would be a flat artifact. See _apply_friction.")
    print("  (no FAIL lines above means Phase 0 bring-up passed)")


if __name__ == "__main__":
    main()
