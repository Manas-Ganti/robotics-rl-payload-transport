"""Record demo MP4s of a trained policy (and baselines) -- VISUALS ONLY.

Runs AFTER training, from a checkpoint, as its own short job. It never touches
training or the evaluation numbers.

Two renderers (--renderer):
  topdown (default): frames drawn from simulator STATE with matplotlib
      (eval/topdown.py) -- robot, obstacles, goal, lidar rays, trail, and a
      readout of speed / height above ground / tilt. No RTX needed.
  isaac: Isaac Sim's RTX viewport camera. On ARC's L40S nodes this SEGFAULTED
      at app startup (omni.kit.widget.viewport __enable_hydra_engine, headless,
      driver 595.x) -- kept for when that is resolved.

    # on ARC (see arc/record.slurm)
    python eval/record_clips.py --headless --checkpoint results/runs/p1/model_final.pt \\
        --tag p1 --set robot.usd_path=/home/<you>/isaac_assets/carter/carter_v1.usd

Scenarios, camera, resolution and takes live in configs/clips.yaml. Scenarios
pin episode parameters exactly as the OOD harness does (env.set_forced_params
over the eval NOMINAL), so a clip is the same condition the tables measure.

PROOF RULE: a clip is a chosen example, not a statistic. Each clip's exact
parameters and outcome go to results/clips/<tag>/clips_manifest.json -- caption
from there; take success rates only from the eval tables.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-config", default="configs/clips.yaml")
    ap.add_argument("--checkpoint", default=None, help="RL checkpoint (.pt); required if 'rl' is recorded")
    ap.add_argument("--tag", required=True, help="output folder name, e.g. the run name")
    ap.add_argument("--policies", nargs="*", default=None, help="override record.policies")
    ap.add_argument("--scenarios", nargs="*", default=None, help="subset of scenario names")
    ap.add_argument("--renderer", choices=["topdown", "isaac"], default="topdown")
    ap.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override a TRAIN-config key, e.g. robot.usd_path=... (same as training/train.py)",
    )
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(ap)
    except ImportError:
        ap.add_argument("--headless", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    from isaaclab.app import AppLauncher

    isaac_view = args.renderer == "isaac"
    args.enable_cameras = isaac_view   # topdown needs no RTX at all
    app = AppLauncher(args).app

    import math

    import imageio.v2 as imageio
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.envs import ViewerCfg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    from eval.topdown import FrameState, render_frame

    from env.config import Config, load_eval_config, load_yaml, validate_train_config
    from env.nav_env import TransportNavEnv, build_env_cfg
    from env.randomization import EpisodeParams
    from eval.run_eval import RandomPolicy, RLPolicy
    from training.train import apply_overrides, set_global_seed

    clips = load_yaml(args.clips_config)
    video, camera, markers_cfg, record = clips["video"], clips["camera"], clips["markers"], clips["record"]

    eval_cfg = load_eval_config(clips["eval_config"])
    edata = eval_cfg.to_dict()

    # Same env as evaluation: the TRAIN config, with a few envs and our seed.
    train_data = apply_overrides(copy.deepcopy(dict(edata["train"])), args.set)
    train_data["env"] = {**train_data["env"], "num_envs": int(record["num_envs"])}
    train_data["seed"] = int(clips["seed"])
    validate_train_config(train_data)
    train_cfg = Config(train_data, source=args.clips_config)
    set_global_seed(int(clips["seed"]))

    env_cfg = build_env_cfg(train_cfg)
    if isaac_view:
        # VERIFY ON ARC: ViewerCfg field names (Isaac Lab 2.1). origin_type
        # "asset_root" makes the viewport camera follow env 0's robot position.
        env_cfg.viewer = ViewerCfg(
            eye=tuple(float(v) for v in camera["eye_offset_m"]),
            lookat=tuple(float(v) for v in camera["lookat_offset_m"]),
            resolution=tuple(int(v) for v in video["resolution"]),
            origin_type="asset_root",
            env_index=0,
            asset_name="robot",
        )
    env = TransportNavEnv(env_cfg, render_mode="rgb_array" if isaac_view else None)

    topdown = clips["topdown"]
    max_range = float(train_data["sensors"]["raycaster"]["max_range_m"])
    lidar_angles = env.lidar_angles.tolist() if hasattr(env, "lidar_angles") else []
    goal_tol = float(train_data["env"]["goal_tolerance_m"])

    def topdown_frame(obs_t: Any, trail: List, title: str, t_s: float) -> Any:
        """One frame for env 0, from state (and the lidar ranges the policy saw)."""
        spec = env._terrain_specs[0]
        loc = (env.robot.data.root_pos_w[0, :3] - env.scene.env_origins[0, :3]).tolist()
        g = env.robot.data.projected_gravity_b[0]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(g[2])))))
        above = loc[2] - spec.height_at(loc[0], loc[1])
        speed = float(env.robot.data.root_lin_vel_b[0, 0])
        ranges = []
        if env.obs_spec.has("depth") and lidar_angles:
            depth = obs_t[0, env.obs_spec.slice_of("depth")]
            ranges = (depth * max_range if env.obs_spec.normalize else depth).tolist()
        p = spec.params
        state = FrameState(
            robot_xy=(loc[0], loc[1]), yaw=float(env._robot_yaw()[0]),
            goal_xy=tuple(spec.goal_xy), goal_tolerance=goal_tol,
            obstacles=[(o.x, o.y, o.radius) for o in spec.obstacles],
            patch_half=float(spec.size_m) / 2.0,
            lidar_angles=lidar_angles, lidar_ranges=ranges, trail=trail,
            footprint_x=tuple(topdown["footprint_x_m"]),
            footprint_half_width=float(topdown["footprint_half_width_m"]),
            title=title,
            readout=(f"t={t_s:5.1f}s  v={speed:+.2f}m/s  above-ground={above:.2f}m  tilt={tilt:4.1f}deg  |  "
                     f"slope {p.slope_angle_deg:.0f}deg  mu {p.friction_coeff:.2f}  "
                     f"payload {p.payload_mass_kg:.0f}kg  density {p.obstacle_density:.2f}"),
        )
        return render_frame(state, size_px=int(topdown["size_px"]))

    goal_markers = None if not isaac_view else VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/GoalMarkers",
            markers={
                "goal": sim_utils.SphereCfg(
                    radius=float(markers_cfg["goal_radius_m"]),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(float(c) for c in markers_cfg["goal_color"])
                    ),
                )
            },
        )
    )

    def show_goals() -> None:
        """Goal spheres at each env's goal, on the (possibly sloped) ground."""
        if goal_markers is None:
            return
        xy = env.goal_pos + env.scene.env_origins[:, :2]
        z = torch.tensor(
            [
                (spec.height_at(*spec.goal_xy) if spec is not None else 0.0)
                for spec in env._terrain_specs
            ],
            dtype=torch.float32,
            device=env.device,
        ) + env.scene.env_origins[:, 2] + float(markers_cfg["goal_radius_m"])
        goal_markers.visualize(translations=torch.cat([xy, z.unsqueeze(-1)], dim=-1))

    # ---- policies (built once, reused across scenarios) --------------------
    wanted = args.policies or list(record["policies"])
    policies: Dict[str, Any] = {}
    for name in wanted:
        if name == "rl":
            if not args.checkpoint:
                raise SystemExit("--checkpoint is required to record the 'rl' policy")
            policies[name] = RLPolicy(args.checkpoint, env, train_cfg, deterministic=True)
        elif name == "nav2":
            from baselines.nav2_runner import Nav2Policy

            policies[name] = Nav2Policy(env, eval_cfg)
        elif name == "random":
            policies[name] = RandomPolicy(env)
        else:
            raise SystemExit(f"unknown policy {name!r} (rl | nav2 | random)")

    nominal = EpisodeParams(**{k: float(v) for k, v in edata["nominal"].items()})
    scenarios = [s for s in clips["scenarios"] if not args.scenarios or s["name"] in args.scenarios]

    policy_dt = float(train_data["sim"]["dt"]) * int(train_data["sim"]["decimation"])
    every = int(video["frame_every_n_steps"])
    fps = 1.0 / (policy_dt * every)
    max_steps = int(float(video["max_seconds"]) / policy_dt)

    out_dir = Path("results/clips") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []

    for scenario in scenarios:
        params = nominal.with_overrides(**{k: float(v) for k, v in scenario["params"].items()})
        env.set_forced_params(params)
        for pname, policy in policies.items():
            for take in range(int(record["episodes_per_scenario"])):
                obs, _ = env.reset()
                policy.reset()
                env.drain_completed_episodes()
                show_goals()
                if isaac_view:
                    for _ in range(3):   # let the renderer settle; first frames can be blank
                        env.render()
                trail: List = []
                title = f"{scenario['name']}  |  {pname}  |  take {take + 1}"

                path = out_dir / f"{args.tag}_{scenario['name']}_{pname}_take{take + 1}.mp4"
                writer = imageio.get_writer(
                    str(path), fps=fps, codec=str(video["codec"]),
                    quality=int(video["quality"]), pixelformat=str(video["pixel_format"]),
                    macro_block_size=16,
                )
                outcome, steps = "clip_cap", max_steps
                for step in range(max_steps):
                    obs_t = obs["policy"] if isinstance(obs, dict) else obs
                    obs, _, terminated, truncated, _ = env.step(policy.act(obs_t))
                    show_goals()   # other envs may have reset to new goals
                    if bool(terminated[0]) or bool(truncated[0]):
                        # env 0 already auto-reset inside step(); its record
                        # (snapshotted before the reset) holds the outcome.
                        rec = [r for r in env.drain_completed_episodes() if r.env_id == 0]
                        outcome = rec[-1].outcome if rec else "unknown"
                        steps = step + 1
                        break   # no frame: the view already shows the next spawn
                    if step % every == 0:
                        if isaac_view:
                            writer.append_data(env.render())
                        else:
                            loc = env.robot.data.root_pos_w[0, :2] - env.scene.env_origins[0, :2]
                            trail.append((float(loc[0]), float(loc[1])))
                            obs_now = obs["policy"] if isinstance(obs, dict) else obs
                            writer.append_data(topdown_frame(obs_now, trail, title, (step + 1) * policy_dt))
                writer.close()

                entry = {
                    "file": path.name,
                    "renderer": args.renderer,
                    "scenario": scenario["name"],
                    "note": scenario.get("note", ""),
                    "policy": pname,
                    "take": take + 1,
                    "outcome": outcome,
                    "seconds": round(steps * policy_dt, 2),
                    "params": params.as_dict(),
                    "checkpoint": args.checkpoint if pname == "rl" else None,
                    "size_mb": round(path.stat().st_size / 1e6, 2),
                }
                manifest.append(entry)
                print(f"[clip] {path.name}: {outcome} after {entry['seconds']} s ({entry['size_mb']} MB)")

    (out_dir / "clips_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {len(manifest)} clips + clips_manifest.json to {out_dir}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
