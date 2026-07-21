"""Training entrypoint: config -> env -> PPO -> W&B -> checkpoints.

Run on the A100:

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
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path, or 'latest'")
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name")
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

    if args.resume:
        checkpoint = resolve_checkpoint_path(data["logging"]["log_dir"], args.resume)
        print(f"Resuming from {checkpoint}")
        runner.load(checkpoint)

    max_iterations = int(data["algo"][data["algo"]["name"]]["max_iterations"])
    print(f"\n=== Training for {max_iterations} iterations ===")
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)

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
    obs, _ = env.reset()

    start_pos = env.robot.data.root_pos_w[:, :2].clone()
    depth_samples: List[float] = []

    for step in range(num_steps):
        actions = torch.rand(env.num_envs, env.action_spec.dim, device=env.device) * 2.0 - 1.0
        obs, rewards, terminated, truncated, info = env.step(actions)

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

    print("\n--- INTERPRET ---")
    if displacement.mean().item() < 0.01:
        print("  FAIL: robot did not move. Check wheel joint names and the actuator")
        print("        mode in env/nav_env.py::_apply_action (velocity vs effort).")
    if depth_samples and abs(sum(depth_samples) / len(depth_samples) - 1.0) < 1e-3:
        print("  FAIL: depth is saturated at max range -- the raycaster is hitting")
        print("        nothing. Check RayCasterCfg.mesh_prim_paths in build_env_cfg.")
    if float(env.friction.max() - env.friction.min()) < 1e-6:
        print("  FAIL: friction is identical across envs. The friction axis is INERT;")
        print("        its OOD curve would be a flat artifact. See _apply_friction.")
    print("  (no FAIL lines above means Phase 0 bring-up passed)")


if __name__ == "__main__":
    main()
