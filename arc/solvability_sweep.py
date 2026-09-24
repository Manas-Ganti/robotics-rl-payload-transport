"""Measure how often the terrain generator finds a SOLVABLE layout -- login node, no GPU.

Pure Python over env/terrain_factory.py + env/solvability.py (the exact code the
env runs at every reset). Answers, before a GPU job does: at each obstacle
density on the train/OOD grid, does generation succeed within the attempt
budget, how many attempts does it take, and does the pool cap bind?

A density where generation fails would crash training/eval with
UnsolvableLayoutError; a density needing many attempts slows every reset.

Run with the project env (numpy + yaml, nothing Isaac):
    ~/miniconda3/envs/rtn/bin/python arc/solvability_sweep.py
    # try alternatives without editing YAML (any combination, comma lists):
    ~/miniconda3/envs/rtn/bin/python arc/solvability_sweep.py \\
        --robot-radius 0.55 --area-fraction 0.30,0.20,0.15 --terrain-size 10,14
"""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.config import Config, load_eval_config, load_train_config  # noqa: E402
from env.randomization import EpisodeParams  # noqa: E402
from env.solvability import UnsolvableLayoutError  # noqa: E402
from env.terrain_factory import TerrainFactory  # noqa: E402


def floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--eval-config", default="configs/eval_ood.yaml")
    ap.add_argument("--robot-radius", type=floats, default=None, help="solvability.robot_radius_m values")
    ap.add_argument("--area-fraction", type=floats, default=None, help="terrain.max_obstacle_area_fraction values")
    ap.add_argument("--terrain-size", type=floats, default=None, help="env.terrain_size_m values")
    ap.add_argument("--episodes", type=int, default=40, help="layouts generated per density")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = load_train_config(args.config).to_dict()
    densities = sorted(set(load_eval_config(args.eval_config).to_dict()["grid"]["obstacle_density"]["points"]))
    nominal = load_eval_config(args.eval_config).to_dict()["nominal"]

    radii = args.robot_radius or [base["solvability"]["robot_radius_m"]]
    fracs = args.area_fraction or [base["terrain"]["max_obstacle_area_fraction"]]
    sizes = args.terrain_size or [base["env"]["terrain_size_m"]]

    print(f"budget: terrain.max_generation_attempts = {base['terrain']['max_generation_attempts']}, "
          f"pool cap = {base['terrain']['max_obstacles_per_env']}, {args.episodes} layouts per density\n")

    for radius, frac, size in itertools.product(radii, fracs, sizes):
        data = copy.deepcopy(base)
        data["solvability"]["robot_radius_m"] = radius
        data["terrain"]["max_obstacle_area_fraction"] = frac
        data["env"]["terrain_size_m"] = size
        factory = TerrainFactory(Config(data))
        print(f"== robot_radius={radius}  area_fraction={frac}  terrain_size={size} m ==")
        print(f"   {'density':>7} {'solved':>8} {'attempts(mean/max)':>19} {'obstacles':>9} {'detour':>7} {'ms/reset':>9}")
        rng = np.random.default_rng(args.seed)
        for density in densities:
            params = EpisodeParams(
                obstacle_density=density,
                slope_angle_deg=float(nominal["slope_angle_deg"]),
                friction_coeff=float(nominal["friction_coeff"]),
                payload_mass_kg=float(nominal["payload_mass_kg"]),
                depth_dropout_prob=0.0,
                depth_noise_std=0.0,
            )
            ok, attempts, counts, detours = 0, [], [], []
            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(args.episodes):
                    try:
                        spec = factory.generate(params, rng)
                    except UnsolvableLayoutError:
                        continue
                    ok += 1
                    attempts.append(spec.generation_attempts)
                    counts.append(len(spec.obstacles))
                    detours.append(spec.optimal_path_length_m / max(spec.straight_line_distance_m, 1e-6))
            ms = (time.perf_counter() - t0) * 1000.0 / args.episodes
            capped = any("bound before" in str(w.message) for w in caught)
            att = f"{np.mean(attempts):.1f}/{max(attempts)}" if attempts else "-"
            obs = f"{np.mean(counts):.0f}" if counts else "-"
            det = f"{np.mean(detours):.2f}" if detours else "-"
            tag = "train" if density <= base["domain"]["train"]["obstacle_density"][1] else "OOD"
            flag = "  <-- POOL CAP BINDS" if capped else ""
            flag += "  <-- FAILS" if ok < args.episodes else ""
            print(f"   {density:>7.2f} {ok:>4}/{args.episodes:<3} {att:>19} {obs:>9} {det:>7} {ms:>9.1f}  {tag}{flag}")
            TerrainFactory._pool_cap_warned = False  # re-arm the once-per-process warning
        print()


if __name__ == "__main__":
    main()
