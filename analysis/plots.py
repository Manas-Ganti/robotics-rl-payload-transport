"""Study figures, generated from results CSVs -- no simulator required.

Runs on the LOCAL machine: copy ``results/eval/*.csv`` back from the A100 and
regenerate every figure without a GPU. That is deliberate -- plot iteration is
where most of the fiddling happens, and it should never occupy A100 time.

    python analysis/plots.py --results-dir results/eval --out-dir results/figures

Figure design rules applied here
--------------------------------
* **Categorical colors in a FIXED order, never cycled.** The palette is
  Okabe-Ito, which is colorblind-safe by construction. A policy keeps its color
  across every figure, so "orange" always means Nav2.
* **Single y-axis, always.** Comparing two measures of different scale gets two
  panels, never a second axis -- dual-axis charts let the author imply any
  correlation they like by rescaling.
* **Identity is never color-alone.** Every series carries a distinct marker and
  a direct label as well as a color.
* **The OOD region is shaded**, so the in-distribution/held-out boundary is
  visible in the figure itself rather than living only in the caption.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: no display on the A100 or in CI
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Palette: Okabe-Ito. Colorblind-safe by construction, assigned in FIXED order.
# ---------------------------------------------------------------------------
POLICY_COLORS: Dict[str, str] = {
    "rl": "#0072B2",      # blue
    "nav2": "#D55E00",    # vermillion
    "random": "#999999",  # gray -- a reference floor, deliberately recessive
}
POLICY_MARKERS: Dict[str, str] = {"rl": "o", "nav2": "s", "random": "^"}
POLICY_LABELS: Dict[str, str] = {"rl": "RL policy", "nav2": "Nav2 baseline", "random": "Random"}

# Status colors are RESERVED for outcome states and never reused as series colors.
OUTCOME_COLORS: Dict[str, str] = {
    "success": "#009E73",              # green
    "collision": "#D55E00",            # vermillion
    "timeout_near_goal": "#E69F00",    # amber -- "almost made it"
    "timeout_stuck_or_lost": "#56B4E9",  # light blue
}

GRID_COLOR = "#DDDDDD"
OOD_SHADE = "#F5F0E8"
TEXT_COLOR = "#333333"

AXIS_LABELS: Dict[str, str] = {
    "obstacle_density": "Obstacle density",
    "slope_angle_deg": "Terrain slope (deg)",
    "friction_coeff": "Ground friction coefficient",
    "payload_mass_kg": "Payload mass (kg)",
    "depth_dropout_prob": "Depth dropout probability",
    "depth_noise_std": "Depth noise std (m)",
}


def _style_axes(ax: plt.Axes) -> None:
    """Recessive grid and axes so the data marks dominate."""
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#BBBBBB")
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)


def _shade_ood_region(ax: plt.Axes, values: Sequence[float], regimes: Sequence[str]) -> None:
    """Shade the held-out region and mark the boundary.

    Drawn from the REGIME LABELS in the data, not from a hardcoded threshold, so
    the shading always reflects the split the evaluation actually used.
    """
    ood_values = [v for v, r in zip(values, regimes) if r == "ood"]
    if not ood_values:
        return

    ascending = len(values) > 1 and values[-1] > values[0]
    if ascending:
        boundary, span = min(ood_values), (min(ood_values), max(values))
    else:
        boundary, span = max(ood_values), (max(values), min(values))

    ax.axvspan(span[0], span[1], color=OOD_SHADE, zorder=0)
    ax.axvline(boundary, color="#B0A99F", linestyle="--", linewidth=1.0, zorder=1)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_cells(results_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load every ``cells_<policy>*.csv`` into ``{policy: [row, ...]}``."""
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for path in sorted(results_dir.glob("cells_*.csv")):
        with path.open("r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                policy = path.stem.replace("cells_", "").split("_")[0]
                out[policy].append(
                    {
                        "axis": row["axis"],
                        "value": float(row["value"]),
                        "regime": row["regime"],
                        "success_rate": float(row["success_rate"]),
                        "collision_rate": float(row["collision_rate"]),
                        "timeout_rate": float(row["timeout_rate"]),
                        "path_efficiency": float(row["path_efficiency"]),
                        "mean_final_distance_m": float(row["mean_final_distance_m"]),
                        "success_ci_low": float(row.get("success_ci_low", 0.0)),
                        "success_ci_high": float(row.get("success_ci_high", 0.0)),
                        "num_episodes": int(float(row["num_episodes"])),
                    }
                )

    if not out:
        raise FileNotFoundError(
            f"No cells_*.csv found in {results_dir}. Run eval/run_eval.py on the A100 "
            "and copy results/eval/ back before plotting."
        )
    return dict(out)


def load_episodes(results_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load per-episode CSVs (used for the failure taxonomy figure)."""
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in sorted(results_dir.glob("episodes_*.csv")):
        with path.open("r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out[row["policy"]].append(row)
    return dict(out)


def _cells_for_axis(rows: Sequence[Dict[str, Any]], axis: str) -> List[Dict[str, Any]]:
    """Rows for one axis, preserving the evaluation's difficulty ordering."""
    return [r for r in rows if r["axis"] == axis]


# ---------------------------------------------------------------------------
# Figure 1: robustness curves (THE headline figure)
# ---------------------------------------------------------------------------
def plot_robustness_curves(
    cells: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
    *,
    metric: str = "success_rate",
) -> Optional[Path]:
    """One panel per axis: metric vs axis value, all policies overlaid.

    This figure IS the research question: how far past the training distribution
    does each approach hold up, and does it fail off a cliff or on a slope.
    """
    axes_present = sorted({r["axis"] for rows in cells.values() for r in rows})
    if not axes_present:
        return None

    ncols = min(2, len(axes_present))
    nrows = int(np.ceil(len(axes_present) / ncols))
    fig, axarr = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.4 * nrows), squeeze=False)

    for idx, axis_name in enumerate(axes_present):
        ax = axarr[idx // ncols][idx % ncols]
        _style_axes(ax)

        shaded = False
        for policy in ("random", "nav2", "rl"):  # RL drawn last -> on top
            if policy not in cells:
                continue
            rows = _cells_for_axis(cells[policy], axis_name)
            if not rows:
                continue

            values = [r["value"] for r in rows]
            metric_values = [r[metric] for r in rows]

            if not shaded:
                _shade_ood_region(ax, values, [r["regime"] for r in rows])
                shaded = True

            # 95% bootstrap CI band -- at ~100 episodes/cell, sampling noise is
            # large enough that a bare line invites over-reading small wiggles.
            if metric == "success_rate":
                ax.fill_between(
                    values,
                    [r["success_ci_low"] for r in rows],
                    [r["success_ci_high"] for r in rows],
                    color=POLICY_COLORS[policy],
                    alpha=0.15,
                    linewidth=0,
                    zorder=2,
                )

            ax.plot(
                values,
                metric_values,
                color=POLICY_COLORS[policy],
                marker=POLICY_MARKERS[policy],
                markersize=6,
                linewidth=2,
                label=POLICY_LABELS[policy],
                zorder=3,
            )

            # Direct label at the line's end: identity without a legend lookup.
            ax.annotate(
                POLICY_LABELS[policy],
                xy=(values[-1], metric_values[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                color=POLICY_COLORS[policy],
                fontsize=8,
                va="center",
                fontweight="bold",
            )

        ax.set_xlabel(AXIS_LABELS.get(axis_name, axis_name), fontsize=10, color=TEXT_COLOR)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10, color=TEXT_COLOR)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            f"{AXIS_LABELS.get(axis_name, axis_name)}",
            fontsize=11,
            color=TEXT_COLOR,
            fontweight="bold",
        )

    # Hide unused panels
    for idx in range(len(axes_present), nrows * ncols):
        axarr[idx // ncols][idx % ncols].set_visible(False)

    handles = [
        plt.Line2D(
            [], [], color=POLICY_COLORS[p], marker=POLICY_MARKERS[p], label=POLICY_LABELS[p], linewidth=2
        )
        for p in ("rl", "nav2", "random")
        if p in cells
    ]
    handles.append(mpatches.Patch(color=OOD_SHADE, label="Held-out (OOD) region"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)

    fig.suptitle(
        "Robustness under distribution shift: in-distribution -> held-out OOD",
        fontsize=13,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    path = out_dir / f"robustness_curves_{metric}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: retention comparison
# ---------------------------------------------------------------------------
def plot_retention_comparison(cells: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> Optional[Path]:
    """Grouped bars: OOD success retained, per axis, per policy.

    Compresses each robustness curve to one number, which is what the README
    results table reports. Read alongside Figure 1 -- retention alone cannot
    distinguish a smooth decline from a cliff.
    """
    axes_present = sorted({r["axis"] for rows in cells.values() for r in rows})
    policies = [p for p in ("rl", "nav2") if p in cells]
    if not axes_present or not policies:
        return None

    fig, ax = plt.subplots(figsize=(1.7 * len(axes_present) + 3.5, 4.6))
    _style_axes(ax)

    width = 0.8 / len(policies)
    x = np.arange(len(axes_present))

    for i, policy in enumerate(policies):
        retentions = []
        for axis_name in axes_present:
            rows = _cells_for_axis(cells[policy], axis_name)
            in_dist = [r["success_rate"] for r in rows if r["regime"] == "train"]
            ood = [r["success_rate"] for r in rows if r["regime"] == "ood"]
            in_dist_mean = float(np.mean(in_dist)) if in_dist else 0.0
            ood_mean = float(np.mean(ood)) if ood else 0.0
            retentions.append(ood_mean / in_dist_mean if in_dist_mean > 1e-9 else 0.0)

        offset = (i - (len(policies) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            retentions,
            width * 0.92,  # 2px-equivalent gap between adjacent bars
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
            zorder=3,
        )
        for bar, value in zip(bars, retentions):
            ax.annotate(
                f"{value:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=TEXT_COLOR,
            )

    ax.axhline(1.0, color="#999999", linestyle=":", linewidth=1.0)
    ax.annotate(
        "no degradation",
        xy=(len(axes_present) - 0.5, 1.0),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color="#777777",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [AXIS_LABELS.get(a, a).replace(" (", "\n(") for a in axes_present], fontsize=9
    )
    ax.set_ylabel("OOD success retained\n(ood / in-distribution)", fontsize=10, color=TEXT_COLOR)
    ax.set_title(
        "How much performance survives the shift, by axis",
        fontsize=12,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    path = out_dir / "retention_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: failure taxonomy
# ---------------------------------------------------------------------------
def plot_failure_taxonomy(
    episodes: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
    *,
    near_goal_threshold_m: float = 2.0,
) -> Optional[Path]:
    """Stacked outcome composition per grid point, one row per policy.

    Success rate says how OFTEN it fails; this says HOW. A shift from success
    into collision means the policy is still trying and mis-handling the
    dynamics; a shift into "stuck or lost" means it stopped navigating at all.
    Those imply different fixes, and only this figure separates them.
    """
    if not episodes:
        return None

    axes_present = sorted({row["axis"] for rows in episodes.values() for row in rows})
    policies = [p for p in ("rl", "nav2") if p in episodes]
    if not axes_present or not policies:
        return None

    fig, axarr = plt.subplots(
        len(policies),
        len(axes_present),
        figsize=(4.0 * len(axes_present), 3.4 * len(policies)),
        squeeze=False,
    )

    for row_idx, policy in enumerate(policies):
        for col_idx, axis_name in enumerate(axes_present):
            ax = axarr[row_idx][col_idx]
            _style_axes(ax)

            rows = [r for r in episodes[policy] if r["axis"] == axis_name]
            values = sorted({float(r["axis_value"]) for r in rows})

            categories = ["success", "collision", "timeout_near_goal", "timeout_stuck_or_lost"]
            fractions: Dict[str, List[float]] = {c: [] for c in categories}

            for value in values:
                at_value = [r for r in rows if float(r["axis_value"]) == value]
                total = max(1, len(at_value))
                counts = {c: 0 for c in categories}
                for r in at_value:
                    outcome = r["outcome"]
                    if outcome == "success":
                        counts["success"] += 1
                    elif outcome == "collision":
                        counts["collision"] += 1
                    else:
                        near = float(r["final_distance_to_goal_m"]) <= near_goal_threshold_m
                        counts["timeout_near_goal" if near else "timeout_stuck_or_lost"] += 1
                for c in categories:
                    fractions[c].append(counts[c] / total)

            bottom = np.zeros(len(values))
            for category in categories:
                ax.bar(
                    range(len(values)),
                    fractions[category],
                    bottom=bottom,
                    color=OUTCOME_COLORS[category],
                    label=category.replace("_", " "),
                    width=0.78,
                    zorder=3,
                )
                bottom += np.array(fractions[category])

            ax.set_xticks(range(len(values)))
            ax.set_xticklabels([f"{v:g}" for v in values], fontsize=8)
            ax.set_ylim(0, 1)
            if col_idx == 0:
                ax.set_ylabel(f"{POLICY_LABELS[policy]}\nepisode fraction", fontsize=9)
            if row_idx == 0:
                ax.set_title(AXIS_LABELS.get(axis_name, axis_name), fontsize=10, fontweight="bold")

    handles = [mpatches.Patch(color=OUTCOME_COLORS[c], label=c.replace("_", " ")) for c in OUTCOME_COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Failure taxonomy: how each policy fails, not just how often",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))

    path = out_dir / "failure_taxonomy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4: path efficiency
# ---------------------------------------------------------------------------
def plot_path_efficiency(cells: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> Optional[Path]:
    """Path efficiency vs axis value -- quality of the successful runs.

    Kept in its own figure rather than sharing an axis with success rate. They
    have different scales and different meanings, and a second y-axis would let
    the reader infer a relationship that the data does not support.
    """
    return plot_robustness_curves(cells, out_dir, metric="path_efficiency")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def write_summary_table(cells: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> Path:
    """Write the README results table as Markdown."""
    axes_present = sorted({r["axis"] for rows in cells.values() for r in rows})
    policies = [p for p in ("rl", "nav2", "random") if p in cells]

    lines = [
        "| Axis | Policy | In-dist success | OOD success | Retention | Max drop | Class |",
        "|---|---|---|---|---|---|---|",
    ]

    for axis_name in axes_present:
        for policy in policies:
            rows = _cells_for_axis(cells[policy], axis_name)
            if not rows:
                continue

            rates = [r["success_rate"] for r in rows]
            in_dist = [r["success_rate"] for r in rows if r["regime"] == "train"]
            ood = [r["success_rate"] for r in rows if r["regime"] == "ood"]

            in_dist_mean = float(np.mean(in_dist)) if in_dist else 0.0
            ood_mean = float(np.mean(ood)) if ood else 0.0
            retention = ood_mean / in_dist_mean if in_dist_mean > 1e-9 else 0.0
            max_drop = max((rates[i] - rates[i + 1] for i in range(len(rates) - 1)), default=0.0)

            if in_dist_mean < 1e-9:
                classification = "undefined"
            elif max_drop >= 0.25:
                classification = "catastrophic"
            elif retention >= 0.9:
                classification = "robust"
            else:
                classification = "graceful"

            lines.append(
                f"| {AXIS_LABELS.get(axis_name, axis_name)} | {POLICY_LABELS[policy]} | "
                f"{in_dist_mean:.3f} | {ood_mean:.3f} | {retention:.3f} | "
                f"{max_drop:.3f} | {classification} |"
            )

    path = out_dir / "results_table.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate study figures from results CSVs.")
    parser.add_argument("--results-dir", type=str, default="results/eval")
    parser.add_argument("--out-dir", type=str, default="results/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_cells(results_dir)
    episodes = load_episodes(results_dir)

    written: List[Path] = []
    for produced in (
        plot_robustness_curves(cells, out_dir),
        plot_retention_comparison(cells, out_dir),
        plot_failure_taxonomy(episodes, out_dir),
        plot_path_efficiency(cells, out_dir),
        write_summary_table(cells, out_dir),
    ):
        if produced is not None:
            written.append(produced)

    print(f"Wrote {len(written)} artifacts to {out_dir}:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
