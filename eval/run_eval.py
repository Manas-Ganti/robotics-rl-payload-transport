"""Evaluation entrypoint: checkpoint + eval_ood.yaml -> results table.

Run on the A100:

    # RL policy across the full OOD grid
    python eval/run_eval.py --checkpoint results/runs/<run>/model_final.pt \\
        --config configs/eval_ood.yaml --headless

    # Nav2 baseline through the SAME harness
    python eval/run_eval.py --policy nav2 --config configs/eval_ood.yaml --headless

    # One axis only (fast iteration)
    python eval/run_eval.py --checkpoint ... --axes slope_angle_deg --headless

Loading the eval config ALREADY asserts the train/OOD split, so evaluation
cannot run against a contaminated grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep the OOD grid for an RL checkpoint or the Nav2 baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/eval_ood.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="RL checkpoint (.pt)")
    parser.add_argument(
        "--policy",
        type=str,
        default="rl",
        choices=["rl", "nav2", "random"],
        help="Which policy to evaluate. 'random' is a sanity floor for the metrics.",
    )
    parser.add_argument("--axes", nargs="*", default=None, help="Subset of axes to sweep")
    parser.add_argument("--episodes-per-cell", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None, help="Suffix for output filenames")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a TRAIN-config key the eval env reuses, e.g. "
        "robot.usd_path=/path/carter_v1.usd (same semantics as training/train.py)",
    )

    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    except ImportError:
        parser.add_argument("--headless", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Policy wrappers implementing eval.ood_harness.Policy
# ---------------------------------------------------------------------------
class RLPolicy:
    """Wraps a trained checkpoint behind the harness's Policy interface."""

    def __init__(self, checkpoint_path: str, env: Any, cfg: Any, *, deterministic: bool = True):
        import torch

        from training.ppo_config import build_runner, wrap_env_for_library

        self.name = "rl"
        self.deterministic = deterministic
        self._torch = torch

        # Reconstruct the runner to restore the policy graph, then extract the
        # inference callable. Rebuilding the runner (rather than loading raw
        # weights) guarantees the observation normalizer travels with the
        # policy -- with empirical_normalization on, a checkpoint loaded without
        # its normalizer sees wrongly-scaled inputs and appears to have
        # forgotten how to navigate.
        wrapped = wrap_env_for_library(env, cfg)
        runner = build_runner(wrapped, cfg, log_dir="/tmp/eval", device=str(env.device))
        runner.load(checkpoint_path)

        # VERIFY ON A100: rsl_rl exposes inference via
        # `get_inference_policy(device=...)` in recent versions. Older forks use
        # `runner.alg.actor_critic.act_inference`. Check which exists.
        if hasattr(runner, "get_inference_policy"):
            self._policy = runner.get_inference_policy(device=env.device)
        else:
            self._policy = runner.alg.actor_critic.act_inference

        print(f"Loaded RL checkpoint: {checkpoint_path}")

    def act(self, observations: Any) -> Any:
        with self._torch.inference_mode():
            return self._policy(observations)

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        """Stateless feed-forward policy -- nothing to reset.

        Would need to clear hidden state here if the policy became recurrent.
        """
        return None


class RandomPolicy:
    """Uniform random actions -- a sanity floor for the metrics.

    Worth running once: it establishes what success rate the TASK yields with no
    competence at all. If a trained policy is near this floor, the problem is
    training, not generalization -- and if the random policy scores well, the
    task is too easy for the OOD study to say anything.
    """

    def __init__(self, env: Any):
        import torch

        self.name = "random"
        self._torch = torch
        self._env = env

    def act(self, observations: Any) -> Any:
        return (
            self._torch.rand(
                self._env.num_envs, self._env.action_spec.dim, device=self._env.device
            )
            * 2.0
            - 1.0
        )

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        return None


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
def print_results_table(result: Any) -> None:
    """Print the headline degradation table -- the project's core output."""
    print("\n" + "=" * 78)
    print(f"OOD ROBUSTNESS SUMMARY -- policy: {result.policy_name}")
    print("=" * 78)
    header = f"{'axis':<22} {'in-dist':>9} {'ood':>9} {'retention':>10} {'max drop':>9} {'class':>14}"
    print(header)
    print("-" * 78)

    for axis, profile in result.profiles.items():
        print(
            f"{axis:<22} "
            f"{profile.in_distribution_mean:>9.3f} "
            f"{profile.ood_mean:>9.3f} "
            f"{profile.retention:>10.3f} "
            f"{profile.max_drop:>9.3f} "
            f"{profile.classification:>14}"
        )

    print("-" * 78)
    print(
        "retention = ood_success / in_dist_success   |   "
        "class: robust (>=0.9) / graceful / catastrophic (cliff)"
    )
    print(f"wall clock: {result.wall_clock_s / 60.0:.1f} min")
    print("=" * 78)


def main() -> None:
    args = parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from env.config import Config, load_eval_config
    from env.nav_env import TransportNavEnv, build_env_cfg
    from eval.ood_harness import OODHarness
    from env.config import validate_train_config
    from training.train import apply_overrides, set_global_seed

    # Loading this asserts the train/OOD split before anything else happens.
    eval_cfg = load_eval_config(args.config)
    data = eval_cfg.to_dict()

    if args.episodes_per_cell is not None:
        data["eval"]["episodes_per_cell"] = args.episodes_per_cell
    if args.num_envs is not None:
        data["eval"]["num_envs"] = args.num_envs
    if args.seed is not None:
        data["seed"] = args.seed

    set_global_seed(int(data["seed"]))

    # Evaluation reuses the TRAINING env config so the robot, sensors, and
    # observation layout are identical -- only the env count, the seed, and the
    # source of the domain parameters differ. Anything else would make the
    # comparison to training performance invalid.
    train_data = apply_overrides(dict(data["train"]), args.set)
    validate_train_config(train_data)
    train_data["env"] = {**train_data["env"], "num_envs": int(data["eval"]["num_envs"])}
    train_data["seed"] = int(data["seed"])
    train_cfg = Config(train_data, source=args.config)

    env_cfg = build_env_cfg(train_cfg)
    env = TransportNavEnv(env_cfg)
    print(env.obs_spec.describe())

    # ---- select the policy ------------------------------------------------
    if args.policy == "rl":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required when --policy rl")
        policy: Any = RLPolicy(
            args.checkpoint,
            env,
            train_cfg,
            deterministic=bool(data["eval"]["deterministic_policy"]),
        )
    elif args.policy == "nav2":
        from baselines.nav2_runner import Nav2Policy

        policy = Nav2Policy(env, eval_cfg)
    else:
        policy = RandomPolicy(env)

    # ---- sweep ------------------------------------------------------------
    harness = OODHarness(eval_cfg, env_factory=lambda: env, verbose=True)
    result = harness.run(policy, axes=args.axes)

    harness.write_results(result, tag=args.tag)
    print_results_table(result)

    harness.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
