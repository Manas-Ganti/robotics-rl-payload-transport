"""The SHARED OOD evaluation harness -- used by BOTH the RL policy and Nav2.

CLAUDE.md principle 5 ("honest baseline") lives here. Nav2 and the learned
policy are not merely compared; they are driven through this exact code, over
the same grid, the same seeds, the same terrain layouts, and the same metric
definitions. The only thing that differs is the object implementing
:class:`Policy`.

Sweep design
------------
One axis varies at a time; all other axes are pinned to their in-distribution
nominal values (``eval_ood.yaml: nominal``). That makes any measured
degradation attributable to a single distribution shift instead of a confounded
mixture. Grid points are visited in CONFIGURED ORDER, which must run from
easiest to hardest -- ``eval/metrics.py:classify_degradation`` reads adjacency
to detect cliffs, so a shuffled axis would invent or hide them.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from env.config import Config, Range, classify_point, get_ood_ranges, get_train_ranges
from env.payload import climb_feasible
from env.randomization import EpisodeParams, params_from_grid_point
from eval.metrics import (
    CellMetrics,
    DegradationProfile,
    EpisodeResult,
    aggregate_cell,
    classify_degradation,
    results_to_rows,
)


# ---------------------------------------------------------------------------
# Policy interface -- the seam that makes RL and Nav2 interchangeable
# ---------------------------------------------------------------------------
@runtime_checkable
class Policy(Protocol):
    """Anything that can drive the environment through the harness.

    Implemented by the RL checkpoint wrapper (``eval/run_eval.py``) and by
    ``baselines/nav2_runner.py``. Deliberately minimal: a wider interface would
    tempt the harness into policy-specific special cases, and the honesty of the
    comparison rests on there being none.
    """

    name: str

    def act(self, observations: Any) -> Any:
        """Map a batched observation to a batched action in ``[-1, 1]``."""
        ...

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        """Clear any internal state for the given envs (planners, filters)."""
        ...


@dataclass
class SweepResult:
    """Everything one policy produced across the whole OOD grid."""

    policy_name: str
    cells: List[CellMetrics]
    profiles: Dict[str, DegradationProfile]
    episode_rows: List[Dict[str, Any]]
    wall_clock_s: float

    def profile_for(self, axis: str) -> Optional[DegradationProfile]:
        return self.profiles.get(axis)


class OODHarness:
    """Sweeps the OOD grid for ANY policy and logs verifiable metrics."""

    def __init__(
        self,
        eval_cfg: Config,
        env_factory: Callable[[], Any],
        *,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        eval_cfg
            Loaded eval config. Loading it has ALREADY asserted the train/OOD
            split (``env.config.load_eval_config``), so by the time the harness
            runs, the split is guaranteed.
        env_factory
            Zero-arg callable returning a stepped environment exposing
            ``reset()``, ``step(actions)``, ``set_forced_params(params)``, and
            ``drain_completed_episodes()``. ``env.nav_env.TransportNavEnv``
            satisfies this; a mock can too, which is what keeps the harness
            testable without Isaac.
        """
        self.cfg = eval_cfg
        self.data = eval_cfg.to_dict()
        self.env_factory = env_factory
        self.verbose = verbose

        self.train_ranges: Dict[str, Range] = get_train_ranges(eval_cfg)
        self.ood_ranges: Dict[str, Range] = get_ood_ranges(eval_cfg)
        self.nominal: Dict[str, float] = dict(self.data["nominal"])

        eval_block = self.data["eval"]
        self.episodes_per_cell = int(eval_block["episodes_per_cell"])
        self.output_dir = Path(eval_block["output_dir"])
        self.metrics_cfg = self.data["metrics"]
        self.seed = int(self.data["seed"])

        self._env: Optional[Any] = None

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------
    def enabled_axes(self) -> List[str]:
        """Axes marked enabled in the eval grid, in config order."""
        return [axis for axis, spec in self.data["grid"].items() if spec.get("enabled", False)]

    def grid_points(self, axis: str) -> List[Tuple[float, str]]:
        """Return ``[(value, regime), ...]`` for one axis, in difficulty order.

        The regime label ("train" / "ood" / "gap") is derived from the declared
        ranges, never hand-assigned -- so a point cannot be mislabelled as
        held-out when it is actually inside the training distribution.
        """
        points = [float(v) for v in self.data["grid"][axis]["points"]]
        self._assert_monotonic(axis, points)

        train_range = self.train_ranges[axis]
        ood_range = self.ood_ranges.get(axis)
        return [(value, classify_point(value, train_range, ood_range)) for value in points]

    @staticmethod
    def _assert_monotonic(axis: str, points: Sequence[float]) -> None:
        """Grid points must be monotonic, i.e. ordered by difficulty.

        Cliff detection compares ADJACENT points, so a non-monotonic axis would
        manufacture spurious cliffs (or conceal real ones) purely from ordering.
        Note that some axes get harder as the value DECREASES (friction), which
        is why either direction is accepted -- but not a zigzag.
        """
        ascending = all(b >= a for a, b in zip(points[:-1], points[1:]))
        descending = all(b <= a for a, b in zip(points[:-1], points[1:]))
        if not (ascending or descending):
            raise ValueError(
                f"grid.{axis}.points must be monotonic (ordered easiest -> hardest), "
                f"got {list(points)}. Cliff detection in metrics.classify_degradation "
                "compares adjacent points and would report ordering artifacts as cliffs."
            )

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def run(self, policy: Policy, *, axes: Optional[Sequence[str]] = None) -> SweepResult:
        """Sweep the full grid for one policy and return aggregated results."""
        start_time = time.time()
        axes_to_run = list(axes) if axes is not None else self.enabled_axes()

        if not axes_to_run:
            raise ValueError(
                "No axes enabled in the eval grid. Set grid.<axis>.enabled: true "
                "in configs/eval_ood.yaml."
            )

        if self._env is None:
            self._env = self.env_factory()

        all_cells: List[CellMetrics] = []
        all_rows: List[Dict[str, Any]] = []
        profiles: Dict[str, DegradationProfile] = {}

        for axis in axes_to_run:
            self._log(f"\n=== axis: {axis} ===")
            axis_cells: List[CellMetrics] = []

            for value, regime in self.grid_points(axis):
                results = self._run_cell(policy, axis=axis, value=value)
                cell_params = params_from_grid_point(self.nominal, axis=axis, value=value)
                feasible = climb_feasible(
                    self.data["train"]["robot"], cell_params.payload_mass_kg, cell_params.slope_angle_deg
                )

                cell = aggregate_cell(
                    results,
                    axis=axis,
                    value=value,
                    regime=regime,
                    compute_ci=bool(self.metrics_cfg.get("bootstrap_ci", True)),
                    bootstrap_samples=int(self.metrics_cfg.get("bootstrap_samples", 1000)),
                    ci_level=float(self.metrics_cfg.get("ci_level", 0.95)),
                    seed=self.seed,
                    feasible=feasible,
                )
                axis_cells.append(cell)
                all_cells.append(cell)
                all_rows.extend(
                    results_to_rows(
                        results, axis=axis, value=value, regime=regime, policy=policy.name
                    )
                )

                self._log(
                    f"  {axis}={value:<6g} [{regime:5s}] "
                    f"success={cell.success_rate:.2f} "
                    f"collision={cell.collision_rate:.2f} "
                    f"timeout={cell.timeout_rate:.2f} "
                    f"path_eff={cell.path_efficiency:.2f} "
                    f"(n={cell.num_episodes})"
                    + ("" if feasible else "  [INFEASIBLE: exceeds motor torque cap]")
                )

            profile = classify_degradation(
                axis_cells,
                cliff_drop_threshold=float(self.metrics_cfg.get("cliff_drop_threshold", 0.25)),
            )
            profiles[axis] = profile
            self._log(
                f"  -> {axis}: {profile.classification.upper()} "
                f"(retention={profile.retention:.2f}, max_drop={profile.max_drop:.2f})"
            )

        return SweepResult(
            policy_name=policy.name,
            cells=all_cells,
            profiles=profiles,
            episode_rows=all_rows,
            wall_clock_s=time.time() - start_time,
        )

    def _run_cell(self, policy: Policy, *, axis: str, value: float) -> List[EpisodeResult]:
        """Collect ``episodes_per_cell`` episodes at one grid point.

        Every axis except ``axis`` is pinned to its nominal in-distribution
        value, so the shift under test is exactly one-dimensional.
        """
        env = self._env
        assert env is not None

        params: EpisodeParams = params_from_grid_point(self.nominal, axis=axis, value=value)
        env.set_forced_params(params)

        # Full reset so every env immediately adopts the pinned params -- without
        # this, envs mid-episode would finish under the PREVIOUS grid point's
        # parameters and contaminate this cell's results.
        env.reset()
        policy.reset()
        env.drain_completed_episodes()  # discard records from the previous cell

        collected: List[EpisodeResult] = []
        max_steps = self._max_steps_per_cell(env)

        for _ in range(max_steps):
            obs = self._current_observations(env)
            actions = policy.act(obs)
            env.step(actions)

            for record in env.drain_completed_episodes():
                collected.append(
                    EpisodeResult(
                        outcome=record.outcome,
                        optimal_path_length_m=record.optimal_path_length_m,
                        actual_path_length_m=record.actual_path_length_m,
                        episode_length_steps=record.episode_length_steps,
                        final_distance_to_goal_m=record.final_distance_to_goal_m,
                        params=dict(record.params),
                    )
                )

            if len(collected) >= self.episodes_per_cell:
                break

        if len(collected) < self.episodes_per_cell:
            # Under-collection biases results: the episodes that finish FIRST are
            # the short ones (quick collisions), so a truncated cell over-reports
            # collisions and under-reports slow successes. Surface it rather than
            # quietly aggregating a skewed sample.
            import warnings

            warnings.warn(
                f"Cell {axis}={value} collected only {len(collected)}/"
                f"{self.episodes_per_cell} episodes within the step budget. "
                "Short episodes finish first, so this sample is biased toward "
                "early failures. Raise eval.num_envs or the episode budget.",
                stacklevel=2,
            )

        return collected[: self.episodes_per_cell]

    def _max_steps_per_cell(self, env: Any) -> int:
        """Step budget for one cell, with headroom for stragglers.

        Enough steps for every parallel env to run several full-length episodes;
        the loop exits early once the quota is met, so this is only a ceiling.
        """
        num_envs = getattr(env, "num_envs", 1)
        episode_steps = int(getattr(env, "max_episode_length", 1500))
        waves_needed = int(np.ceil(self.episodes_per_cell / max(1, num_envs)))
        return int(episode_steps * (waves_needed + 1))

    @staticmethod
    def _current_observations(env: Any) -> Any:
        """Fetch the current observation from the env, tolerating both shapes.

        Isaac Lab returns ``{"policy": tensor}``; a mock env may return a bare
        array. Both are accepted so the harness stays testable without Isaac.
        """
        obs = env._get_observations() if hasattr(env, "_get_observations") else env.get_observations()
        return obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def write_results(self, result: SweepResult, *, tag: Optional[str] = None) -> Dict[str, Path]:
        """Write per-episode, per-cell, and per-axis CSV/JSON outputs.

        Three granularities on purpose: ``analysis/plots.py`` reads the cell CSV,
        the summary JSON feeds the README results table, and the per-episode CSV
        allows any metric to be recomputed later without re-running the sim --
        which matters because sim time on the A100 is the scarce resource here.
        """
        suffix = f"_{tag}" if tag else ""
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        episodes_path = out_dir / f"episodes_{result.policy_name}{suffix}.csv"
        cells_path = out_dir / f"cells_{result.policy_name}{suffix}.csv"
        summary_path = out_dir / f"summary_{result.policy_name}{suffix}.json"

        if result.episode_rows:
            fieldnames = sorted({key for row in result.episode_rows for key in row})
            with episodes_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(result.episode_rows)

        if result.cells:
            cell_rows = [cell.as_row() for cell in result.cells]
            with cells_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(cell_rows[0]))
                writer.writeheader()
                writer.writerows(cell_rows)

        summary = {
            "policy": result.policy_name,
            "wall_clock_s": result.wall_clock_s,
            "episodes_per_cell": self.episodes_per_cell,
            "seed": self.seed,
            "profiles": {axis: profile.as_row() for axis, profile in result.profiles.items()},
        }
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        self._log(f"\nWrote:\n  {episodes_path}\n  {cells_path}\n  {summary_path}")
        return {"episodes": episodes_path, "cells": cells_path, "summary": summary_path}

    def close(self) -> None:
        """Release the environment (and the simulator behind it)."""
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
        self._env = None
