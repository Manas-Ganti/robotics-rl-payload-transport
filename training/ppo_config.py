"""RL algorithm configuration -- the swappable seam between config and library.

CLAUDE.md: "Prefer Isaac Lab's RL integration (e.g. rsl_rl / rl_games wrapper)
over a hand-rolled trainer -- isolate the choice so it can be swapped."

That isolation is this module. ``training/train.py`` never imports rsl_rl
directly; it calls :func:`build_runner_cfg` and :func:`build_runner`. Switching
to rl_games or skrl means editing this file and nothing else.

All hyperparameters trace to ``configs/train.yaml`` under ``algo.*`` -- there
are no literals below, only translation.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

# NOTE: rsl_rl is imported lazily inside the builder functions so this module
# stays importable on the local machine (where rsl_rl and torch may be absent)
# for config-shape tests.


def build_runner_cfg(cfg: Any) -> Dict[str, Any]:
    """Translate the YAML ``algo`` block into an rsl_rl runner config dict.

    Returned as a plain dict rather than an rsl_rl dataclass so it can be
    validated and logged without importing the library -- and so a version bump
    that renames a field fails at one obvious place.

    VERIFY ON A100: rsl_rl config field names have shifted across releases.
    Cross-check against the installed version's
    ``rsl_rl.runners.OnPolicyRunner`` docstring. Commonly renamed:
      * ``num_steps_per_env`` (was ``num_transitions_per_env`` in older forks)
      * ``empirical_normalization`` (absent in some versions)
      * ``class_name`` keys inside the policy/algorithm sub-dicts
    """
    data = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    algo = data["algo"]
    name = algo["name"]

    if name == "ppo":
        return _build_ppo_cfg(algo["ppo"], data)
    if name == "sac":
        return _build_sac_cfg(algo["sac"], data)
    raise ValueError(f"algo.name must be 'ppo' or 'sac', got {name!r}")


def _build_ppo_cfg(ppo: Mapping[str, Any], data: Mapping[str, Any]) -> Dict[str, Any]:
    """PPO runner config (the default path)."""
    logging_cfg = data["logging"]
    experiment = data["experiment"]

    return {
        "seed": int(data["seed"]),
        "num_steps_per_env": int(ppo["num_steps_per_env"]),
        "max_iterations": int(ppo["max_iterations"]),
        "save_interval": int(ppo["save_interval"]),
        "empirical_normalization": bool(ppo["empirical_normalization"]),
        "experiment_name": str(experiment["name"]),
        "run_name": experiment.get("run_name") or "",
        "logger": "wandb" if logging_cfg["wandb"]["enabled"] else "tensorboard",
        "wandb_project": logging_cfg["wandb"]["project"],
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": float(ppo["policy"]["init_noise_std"]),
            "actor_hidden_dims": list(ppo["policy"]["actor_hidden_dims"]),
            "critic_hidden_dims": list(ppo["policy"]["critic_hidden_dims"]),
            "activation": str(ppo["policy"]["activation"]),
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": float(ppo["algorithm"]["value_loss_coef"]),
            "use_clipped_value_loss": bool(ppo["algorithm"]["use_clipped_value_loss"]),
            "clip_param": float(ppo["algorithm"]["clip_param"]),
            "entropy_coef": float(ppo["algorithm"]["entropy_coef"]),
            "num_learning_epochs": int(ppo["algorithm"]["num_learning_epochs"]),
            "num_mini_batches": int(ppo["algorithm"]["num_mini_batches"]),
            "learning_rate": float(ppo["algorithm"]["learning_rate"]),
            "schedule": str(ppo["algorithm"]["schedule"]),
            "gamma": float(ppo["algorithm"]["gamma"]),
            "lam": float(ppo["algorithm"]["lam"]),
            "desired_kl": float(ppo["algorithm"]["desired_kl"]),
            "max_grad_norm": float(ppo["algorithm"]["max_grad_norm"]),
        },
    }


def _build_sac_cfg(sac: Mapping[str, Any], data: Mapping[str, Any]) -> Dict[str, Any]:
    """SAC runner config (lowest-priority comparison path per CLAUDE.md).

    VERIFY ON A100: rsl_rl is an ON-POLICY library and mainline versions ship no
    SAC implementation. Enabling ``algo.name: sac`` therefore likely requires
    switching ``algo.library`` to skrl (which does ship SAC for Isaac Lab) and
    adding a branch in :func:`build_runner`. The config block is written and
    validated so the switch is a small, contained change -- but it is NOT
    expected to run as-is.
    """
    experiment = data["experiment"]
    return {
        "seed": int(data["seed"]),
        "max_iterations": int(sac["max_iterations"]),
        "save_interval": int(sac["save_interval"]),
        "experiment_name": str(experiment["name"]),
        "agent": {
            "class_name": "SAC",
            "batch_size": int(sac["batch_size"]),
            "replay_buffer_size": int(sac["replay_buffer_size"]),
            "learning_rate": float(sac["learning_rate"]),
            "gamma": float(sac["gamma"]),
            "tau": float(sac["tau"]),
            "alpha": float(sac["alpha"]),
            "learning_starts": int(sac["learning_starts"]),
            "actor_hidden_dims": list(sac["actor_hidden_dims"]),
            "critic_hidden_dims": list(sac["critic_hidden_dims"]),
        },
    }


def build_runner(env: Any, cfg: Any, log_dir: str, device: str) -> Any:
    """Construct the RL runner for the configured algorithm.

    This function and :func:`build_runner_cfg` are the ONLY places the RL
    library appears. ``train.py`` treats the result as an opaque object with
    ``.learn()`` and ``.save()``.

    VERIFY ON A100: OnPolicyRunner's constructor signature
    ``(env, train_cfg, log_dir, device)`` and whether it expects a dict or a
    dataclass for ``train_cfg``.
    """
    data = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    algo_name = data["algo"]["name"]
    library = data["algo"]["library"]

    if algo_name == "sac":
        raise NotImplementedError(
            "algo.name='sac' is configured but not wired to a runner. rsl_rl is "
            "on-policy only. Switch algo.library to 'skrl' and add the branch here "
            "-- the SAC hyperparameter block is already defined and validated. "
            "PPO is the default and supported path (CLAUDE.md marks SAC lowest priority)."
        )

    if library != "rsl_rl":
        raise NotImplementedError(
            f"algo.library='{library}' is not wired. Only 'rsl_rl' is implemented. "
            "Add the branch here -- this function is the intended swap point."
        )

    from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415  (lazy: A100-only import)

    runner_cfg = build_runner_cfg(cfg)
    return OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=device)


def wrap_env_for_library(env: Any, cfg: Any) -> Any:
    """Wrap the Isaac Lab env in the library's expected interface.

    VERIFY ON A100: wrapper import path and constructor.
      Isaac Lab 1.x : from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
      older/Orbit   : from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
    """
    data = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    library = data["algo"]["library"]

    if library == "rsl_rl":
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: PLC0415

        return RslRlVecEnvWrapper(env)

    raise NotImplementedError(f"No env wrapper implemented for algo.library='{library}'")


def resolve_checkpoint_path(log_dir: str, checkpoint: Optional[str]) -> Optional[str]:
    """Resolve a checkpoint path, accepting 'latest' as a convenience alias."""
    from pathlib import Path

    if checkpoint is None:
        return None
    if checkpoint != "latest":
        return checkpoint

    candidates = sorted(Path(log_dir).glob("**/model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching 'model_*.pt' under {log_dir}")
    return str(candidates[-1])
