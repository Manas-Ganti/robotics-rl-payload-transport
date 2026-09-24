"""Verifiable metrics -- PURE LOGIC, no Isaac Sim import.

CLAUDE.md principle 4: every metric is computed mechanically against ground
truth. Nothing here is subjective, and nothing here depends on which policy
produced the episodes -- the RL policy and Nav2 are scored by identical code.

Metric definitions
------------------
success_rate      fraction of episodes reaching the goal within tolerance
collision_rate    fraction terminating on a collision
timeout_rate      fraction exhausting the episode budget
path_efficiency   SPL-style: optimal_path / max(actual_path, optimal_path),
                  averaged over SUCCESSFUL episodes only
degradation       slope + cliff analysis of success rate along an OOD axis

The graceful-vs-catastrophic distinction is THE headline output of this project
(see "The One Finding We Are Hunting" in CLAUDE.md), so it is defined
explicitly and mechanically in :func:`classify_degradation` rather than being
eyeballed off a plot.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EpisodeResult:
    """One episode's ground-truth outcome. Mirrors ``env.nav_env.EpisodeRecord``.

    Duplicated deliberately: this module must stay importable without Isaac Sim
    so metrics can be recomputed from CSVs on a laptop.
    """

    outcome: str                      # "success" | "collision" | "timeout"
    optimal_path_length_m: float
    actual_path_length_m: float
    episode_length_steps: int
    final_distance_to_goal_m: float
    params: Dict[str, float] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.outcome == "success"

    @property
    def collision(self) -> bool:
        return self.outcome == "collision"

    @property
    def timeout(self) -> bool:
        return self.outcome == "timeout"


@dataclass
class CellMetrics:
    """Aggregated metrics for one grid cell (one axis value)."""

    axis: str
    value: float
    regime: str                       # "train" | "ood" | "gap"
    num_episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    path_efficiency: float
    mean_final_distance_m: float
    success_ci_low: float = 0.0
    success_ci_high: float = 0.0
    # False when the cell is physically impossible for ANY policy (e.g. the
    # motors cannot climb that slope with that payload). Reported, but excluded
    # from the degradation classification -- see classify_degradation.
    feasible: bool = True

    def as_row(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core rates
# ---------------------------------------------------------------------------
def success_rate(results: Sequence[EpisodeResult]) -> float:
    """Fraction of episodes that reached the goal."""
    if not results:
        return 0.0
    return sum(r.success for r in results) / len(results)


def collision_rate(results: Sequence[EpisodeResult]) -> float:
    if not results:
        return 0.0
    return sum(r.collision for r in results) / len(results)


def timeout_rate(results: Sequence[EpisodeResult]) -> float:
    if not results:
        return 0.0
    return sum(r.timeout for r in results) / len(results)


def path_efficiency(results: Sequence[EpisodeResult]) -> float:
    """Mean SPL-style path efficiency over SUCCESSFUL episodes.

    ``optimal / max(actual, optimal)``, clamped to [0, 1]. The max() guards the
    case where the robot's measured path is marginally shorter than the A*
    optimum -- legitimate, since A* moves cell-to-cell on a discrete grid while
    the robot moves continuously and can cut corners. Without the clamp those
    episodes would score above 1.0 and inflate the mean.

    Failed episodes are EXCLUDED rather than scored 0. Mixing them in would
    conflate "got there inefficiently" with "did not get there", and the success
    rate already carries the latter. Read this metric strictly as "when it
    succeeded, how direct was the route".
    """
    successes = [r for r in results if r.success]
    if not successes:
        return 0.0

    ratios = []
    for r in successes:
        optimal = max(r.optimal_path_length_m, 1e-6)
        actual = max(r.actual_path_length_m, 1e-6)
        ratios.append(min(1.0, optimal / max(actual, optimal)))

    return float(np.mean(ratios))


def mean_final_distance(results: Sequence[EpisodeResult]) -> float:
    """Mean distance to goal at termination, over ALL episodes.

    Included because it degrades CONTINUOUSLY where success rate is a step
    function. When success collapses to zero across several OOD points, this is
    what still distinguishes "stalls just short of the goal" from "never leaves
    the start" -- which is precisely the graceful-vs-catastrophic question.
    """
    if not results:
        return 0.0
    return float(np.mean([r.final_distance_to_goal_m for r in results]))


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(
    results: Sequence[EpisodeResult],
    *,
    statistic: str = "success",
    num_samples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Bootstrap CI for a binary rate.

    Reported because episodes_per_cell is typically ~100, where a success rate
    carries roughly +/-10% noise. Without a CI, ordinary sampling noise between
    adjacent grid points is easy to over-read as the onset of degradation.
    """
    if not results:
        return (0.0, 0.0)

    outcomes = np.array(
        [getattr(r, statistic) for r in results], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    n = len(outcomes)
    means = np.array([rng.choice(outcomes, size=n, replace=True).mean() for _ in range(num_samples)])

    alpha = (1.0 - ci_level) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_cell(
    results: Sequence[EpisodeResult],
    *,
    axis: str,
    value: float,
    regime: str,
    compute_ci: bool = True,
    bootstrap_samples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
    feasible: bool = True,
) -> CellMetrics:
    """Aggregate all episodes at one grid point into a single metrics row."""
    ci_low, ci_high = (0.0, 0.0)
    if compute_ci:
        ci_low, ci_high = bootstrap_ci(
            results, num_samples=bootstrap_samples, ci_level=ci_level, seed=seed
        )

    return CellMetrics(
        axis=axis,
        value=float(value),
        regime=regime,
        num_episodes=len(results),
        success_rate=success_rate(results),
        collision_rate=collision_rate(results),
        timeout_rate=timeout_rate(results),
        path_efficiency=path_efficiency(results),
        mean_final_distance_m=mean_final_distance(results),
        success_ci_low=ci_low,
        success_ci_high=ci_high,
        feasible=feasible,
    )


# ---------------------------------------------------------------------------
# THE HEADLINE ANALYSIS: graceful vs catastrophic degradation
# ---------------------------------------------------------------------------
@dataclass
class DegradationProfile:
    """How a policy degrades along ONE study axis. The project's key output."""

    axis: str
    values: List[float]
    success_rates: List[float]
    regimes: List[str]

    in_distribution_mean: float
    ood_mean: float
    retention: float                  # ood_mean / in_distribution_mean
    max_drop: float                   # largest drop between adjacent points
    max_drop_at: Optional[float]      # axis value where that drop occurs
    classification: str               # "graceful" | "catastrophic" | "robust" | "undefined"
    cliff_threshold: float
    infeasible_values: List[float] = field(default_factory=list)  # excluded cells

    def as_row(self) -> Dict[str, Any]:
        return asdict(self)


def classify_degradation(
    cells: Sequence[CellMetrics],
    *,
    cliff_drop_threshold: float = 0.25,
) -> DegradationProfile:
    """Classify a policy's degradation along one axis. THE headline metric.

    Definitions (mechanical, so RL and Nav2 are judged identically):

    * **robust**       -- OOD success retains >= 90% of in-distribution success.
    * **catastrophic** -- some pair of ADJACENT grid points shows a success drop
                          of at least ``cliff_drop_threshold``. A cliff means the
                          policy has an abrupt competence boundary rather than a
                          soft margin, which is the operationally dangerous mode:
                          nothing warns you it is about to fail.
    * **graceful**     -- performance declines, but no single step is a cliff.
    * **undefined**    -- fewer than 2 points, or zero in-distribution success
                          (nothing to degrade FROM -- a policy that never worked
                          cannot be said to generalize badly).

    Cells must be ordered along the axis in the direction of INCREASING
    difficulty. The harness guarantees this; see ``eval/ood_harness.py``.

    Physically INFEASIBLE cells (``feasible=False``) are excluded from the
    means and from cliff detection, and listed in ``infeasible_values``. A drop
    at a cell no policy can pass is the robot's limit, not the policy's.
    """
    all_cells = list(cells)
    infeasible_values = [c.value for c in all_cells if not c.feasible]
    cells = [c for c in all_cells if c.feasible]
    if len(cells) < 2:
        return DegradationProfile(
            axis=all_cells[0].axis if all_cells else "unknown",
            values=[c.value for c in all_cells],
            success_rates=[c.success_rate for c in all_cells],
            regimes=[c.regime for c in all_cells],
            in_distribution_mean=0.0,
            ood_mean=0.0,
            retention=0.0,
            max_drop=0.0,
            max_drop_at=None,
            classification="undefined",
            cliff_threshold=cliff_drop_threshold,
            infeasible_values=infeasible_values,
        )

    rates = [c.success_rate for c in cells]
    values = [c.value for c in cells]
    regimes = [c.regime for c in cells]

    in_dist = [r for r, reg in zip(rates, regimes) if reg == "train"]
    ood = [r for r, reg in zip(rates, regimes) if reg == "ood"]

    in_dist_mean = float(np.mean(in_dist)) if in_dist else 0.0
    ood_mean = float(np.mean(ood)) if ood else 0.0
    retention = ood_mean / in_dist_mean if in_dist_mean > 1e-9 else 0.0

    max_drop, max_drop_at = 0.0, None
    for i in range(len(rates) - 1):
        drop = rates[i] - rates[i + 1]
        if drop > max_drop:
            max_drop, max_drop_at = drop, values[i + 1]

    if in_dist_mean < 1e-9:
        classification = "undefined"
    elif max_drop >= cliff_drop_threshold:
        classification = "catastrophic"
    elif retention >= 0.9:
        classification = "robust"
    else:
        classification = "graceful"

    return DegradationProfile(
        axis=cells[0].axis,
        values=[c.value for c in all_cells],
        success_rates=[c.success_rate for c in all_cells],
        regimes=[c.regime for c in all_cells],
        in_distribution_mean=in_dist_mean,
        ood_mean=ood_mean,
        retention=retention,
        max_drop=max_drop,
        max_drop_at=max_drop_at,
        classification=classification,
        cliff_threshold=cliff_drop_threshold,
        infeasible_values=infeasible_values,
    )


def degradation_slope(cells: Sequence[CellMetrics]) -> float:
    """Least-squares slope of success rate vs axis value.

    A compact scalar for the RL-vs-Nav2 comparison: less negative means the
    policy sheds performance more slowly per unit of distribution shift.
    """
    if len(cells) < 2:
        return 0.0
    x = np.array([c.value for c in cells], dtype=np.float64)
    y = np.array([c.success_rate for c in cells], dtype=np.float64)
    if np.allclose(x, x[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def failure_taxonomy(results: Sequence[EpisodeResult]) -> Dict[str, float]:
    """Break failures into interpretable modes.

    Distinguishes a timeout that ended NEAR the goal (the policy was working,
    just too slow) from one that ended far away (the policy was lost or stuck).
    Those look identical in the success rate but imply very different fixes, and
    the distinction is what makes the failure-taxonomy plot worth reading.
    """
    if not results:
        return {}

    total = len(results)
    near_threshold = 2.0  # metres; "almost made it"

    stuck = sum(1 for r in results if r.timeout and r.final_distance_to_goal_m > near_threshold)
    slow = sum(1 for r in results if r.timeout and r.final_distance_to_goal_m <= near_threshold)

    return {
        "success": sum(r.success for r in results) / total,
        "collision": sum(r.collision for r in results) / total,
        "timeout_stuck_or_lost": stuck / total,
        "timeout_near_goal": slow / total,
    }


def compare_policies(
    profile_a: DegradationProfile,
    profile_b: DegradationProfile,
    *,
    name_a: str = "rl",
    name_b: str = "nav2",
) -> Dict[str, Any]:
    """Head-to-head summary along one axis, for the results table.

    Both profiles must come from the same axis and the same grid -- the whole
    point of the shared harness (principle 5) is that this comparison is
    apples-to-apples.
    """
    if profile_a.axis != profile_b.axis:
        raise ValueError(
            f"Cannot compare different axes: {profile_a.axis!r} vs {profile_b.axis!r}"
        )
    if profile_a.values != profile_b.values:
        raise ValueError(
            f"Cannot compare profiles evaluated on different grids for axis "
            f"{profile_a.axis!r}: {profile_a.values} vs {profile_b.values}"
        )

    return {
        "axis": profile_a.axis,
        f"{name_a}_in_dist": profile_a.in_distribution_mean,
        f"{name_b}_in_dist": profile_b.in_distribution_mean,
        f"{name_a}_ood": profile_a.ood_mean,
        f"{name_b}_ood": profile_b.ood_mean,
        f"{name_a}_retention": profile_a.retention,
        f"{name_b}_retention": profile_b.retention,
        f"{name_a}_class": profile_a.classification,
        f"{name_b}_class": profile_b.classification,
        "retention_advantage": profile_a.retention - profile_b.retention,
    }


# ---------------------------------------------------------------------------
# CSV round-trip (so analysis can rerun without touching a GPU)
# ---------------------------------------------------------------------------
def results_to_rows(
    results: Sequence[EpisodeResult],
    *,
    axis: str,
    value: float,
    regime: str,
    policy: str,
) -> List[Dict[str, Any]]:
    """Flatten per-episode results into CSV rows (one row per episode)."""
    rows: List[Dict[str, Any]] = []
    for r in results:
        row: Dict[str, Any] = {
            "policy": policy,
            "axis": axis,
            "axis_value": value,
            "regime": regime,
            "outcome": r.outcome,
            "optimal_path_length_m": r.optimal_path_length_m,
            "actual_path_length_m": r.actual_path_length_m,
            "episode_length_steps": r.episode_length_steps,
            "final_distance_to_goal_m": r.final_distance_to_goal_m,
        }
        row.update({f"param_{k}": v for k, v in r.params.items()})
        rows.append(row)
    return rows


def rows_to_results(rows: Iterable[Mapping[str, Any]]) -> List[EpisodeResult]:
    """Rebuild :class:`EpisodeResult` objects from CSV rows."""
    out: List[EpisodeResult] = []
    for row in rows:
        params = {
            k[len("param_") :]: float(v)
            for k, v in row.items()
            if k.startswith("param_") and v not in ("", None)
        }
        out.append(
            EpisodeResult(
                outcome=str(row["outcome"]),
                optimal_path_length_m=float(row["optimal_path_length_m"]),
                actual_path_length_m=float(row["actual_path_length_m"]),
                episode_length_steps=int(float(row["episode_length_steps"])),
                final_distance_to_goal_m=float(row["final_distance_to_goal_m"]),
                params=params,
            )
        )
    return out
