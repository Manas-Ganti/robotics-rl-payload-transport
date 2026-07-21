# Robust Payload-Transport Navigation under Terrain & Dynamics Shift

Training a mobile robot to **transport a variable-mass payload across
parameterized, sloped terrain**, then measuring how far the learned policy
generalizes to **out-of-distribution terrain, friction, and payload** — against
a classical **Nav2** baseline run through the *same* evaluation harness.

> **The question:** How far beyond its training distribution can a learned
> transport policy generalize before failing — **gracefully or
> catastrophically** — compared to a classical planner under identical shifts?

---

## Environment first

The environment *is* the contribution. It is built so the generalization
question can be answered mechanically rather than argued:

| Property | How it is enforced |
|---|---|
| **Parameterized, not hand-designed** | Every layout is a deterministic function of `(EpisodeParams, RNG)`. No hand-built levels; all ranges in YAML. |
| **Train/OOD split is sacred** | Disjointness is **asserted at config load** (`env/config.py::assert_no_train_ood_overlap`). Touching endpoints count as overlap. A contaminated grid cannot run. |
| **Solvability guaranteed** | Every episode passes an A* check *with robot-radius clearance* before spawn; unsolvable layouts are regenerated, and budget exhaustion **raises** rather than falling back. |
| **Verifiable metrics** | Success, collision, path efficiency, and the graceful/catastrophic classification are computed against ground truth. No subjective scoring. |
| **Honest baseline** | Nav2 implements the same `Policy` interface and runs through the same harness, grid, seeds, and metric code. |
| **Reproducible** | Global seeding, versioned YAML, resolved-config sidecar next to every checkpoint, W&B, Dockerfile. |

**Payload replaces gravity randomization.** A robot carrying an unknown load is
a real deployment condition; a robot on a planet with different gravity is not.
The prohibition is enforced by a config-tree walk that rejects any gravity
randomization key.

---

## The study axes

| Axis | Parameter | Training (inner) | Held-out OOD (outer) |
|---|---|---|---|
| Geometry | `obstacle_density` | 0.2 – 0.6 | 0.7 – 0.9 |
| Terrain | `slope_angle_deg` | 0 – 10 | 12 – 20 |
| Physical | `friction_coeff` | 0.4 – 0.9 | 0.1 – 0.3 |
| Dynamics | `payload_mass_kg` | 1 – 5 | 6 – 9 |
| Sensor | `depth_dropout` / `noise` | mild | severe |

Each axis is swept **one at a time**, with all others pinned to their
in-distribution nominal — so measured degradation is attributable to a single
shift rather than a confounded mixture.

---

## Task

- **Robot:** differential-drive mobile base (Isaac Lab built-in; isolated in `configs/robot.yaml`)
- **Mission:** carry a sampled-mass payload from start to goal without collision
- **Observation:** 64-ray depth + goal-relative pose `(cos, sin, dist)` + proprioception + payload mass *(the last is toggleable off — that flag is the Phase 5 told-vs-inferred experiment)*
- **Action:** continuous `(linear velocity, angular velocity)`
- **Termination:** goal reached (success) | collision (fail) | timeout (fail)

---

## Quickstart

### Locally (no GPU, no Isaac)

```bash
pip install numpy pyyaml pytest matplotlib
pytest tests/ -v          # reward, solvability, and config tests
```

The pure-logic core — reward, solvability, terrain generation, spaces, config —
imports without a simulator. Only `env/nav_env.py` requires Isaac.

### On the A100

```bash
# 0. Bring-up. Read every FAIL: line before continuing.
python training/train.py --config configs/train.yaml --headless --smoke-test

# 1. Train
python training/train.py --config configs/train.yaml --headless

# 2. Sweep the OOD grid
python eval/run_eval.py --checkpoint results/runs/<run>/model_final.pt --headless

# 3. Baseline, same harness
python eval/run_eval.py --policy nav2 --headless

# 4. Figures (back on the laptop, from CSVs)
python analysis/plots.py --results-dir results/eval --out-dir results/figures
```

**Read [`setup_notes.md`](setup_notes.md) before the first long run.** It is the
collected `# VERIFY ON A100:` checklist, ordered by how badly a silent failure
corrupts the study.

---

## Config-driven iteration

The A100 loop is *run → inspect → tweak → rerun*, so tweaking must not require
code edits. Every phase is a flag:

```bash
python training/train.py --headless --set phases.enable_slope=true
python training/train.py --headless --set phases.enable_sensor_noise=true
python training/train.py --headless --set phases.enable_reward_v2=true
python training/train.py --headless --set phases.enable_inferred_payload=true

# reward ablation without touching a module
python training/train.py --headless --set reward.terms.progress_shaping.enabled=false
```

`--set` only accepts keys that **already exist**, so a typo cannot silently
introduce a setting nothing reads. Overrides are re-validated after application
and logged to W&B.

---

## Repository layout

```
configs/       train.yaml (inner ranges, PPO, reward) | eval_ood.yaml (held-out grid) | robot.yaml
env/           terrain_factory · payload · solvability · randomization · spaces · reward   [PURE]
               nav_env.py                                                        [ISAAC-ONLY]
training/      train.py (entrypoint, W&B, checkpoints) · ppo_config.py (library seam)
baselines/     nav2_runner.py — implements the same Policy interface
eval/          ood_harness.py (shared sweep) · metrics.py (pure) · run_eval.py
analysis/      plots.py — every figure from CSVs, no GPU
tests/         reward · solvability · config — all pass WITHOUT Isaac
```

The split is deliberate: everything except `nav_env.py` is pure enough to unit
test on a laptop, which is what makes the invariants (the train/OOD split, the
solvability guarantee, the reward signs) *testable* rather than merely asserted.

---

## Results

*(Populated from the A100. Regenerate with `python analysis/plots.py`, which
writes `results/figures/results_table.md`.)*

**Classification (mechanical, `eval/metrics.py::classify_degradation`):**
`robust` = OOD retains ≥ 90% of in-distribution success · `catastrophic` = some
adjacent grid pair drops ≥ 25% (an abrupt competence boundary — the
operationally dangerous mode, because nothing warns you) · `graceful` =
declines without a cliff · `undefined` = no in-distribution success to degrade from.

| Axis | Policy | In-dist success | OOD success | Retention | Max drop | Class |
|---|---|---|---|---|---|---|
| Obstacle density | RL | _ | _ | _ | _ | _ |
| Obstacle density | Nav2 | _ | _ | _ | _ | _ |
| Terrain slope | RL | _ | _ | _ | _ | _ |
| Terrain slope | Nav2 | _ | _ | _ | _ | _ |
| Friction | RL | _ | _ | _ | _ | _ |
| Friction | Nav2 | _ | _ | _ | _ | _ |
| Payload mass | RL | _ | _ | _ | _ | _ |
| Payload mass | Nav2 | _ | _ | _ | _ | _ |

**Figures:** `robustness_curves_success_rate.png` (headline) ·
`retention_comparison.png` · `failure_taxonomy.png` ·
`robustness_curves_path_efficiency.png`

### Hypothesis (recorded before running, so it can be wrong)

Nav2 receives a full occupancy costmap while the policy sees only a forward
scan — an asymmetry that **favours the baseline**, deliberately, so it is not a
straw man. But Nav2 plans *geometrically*: it knows nothing about payload mass
or friction. The expected contrast is that **Nav2 holds on the geometry axis and
degrades on the dynamics axes**, where the RL policy at least had the
opportunity to learn compensation. If the RL policy does *not* beat Nav2 on
payload and friction, that is the interesting negative result and it should be
reported as such.

---

## Reading the results honestly

Three failure modes that would make results look good while meaning nothing.
All three are guarded in code, and all three are worth re-checking by hand:

1. **An inert axis.** If a per-env physics write silently fails, that axis's
   curve is flat — which reads as robustness. Tier 1 of `setup_notes.md` exists
   entirely for this. A flat curve is more often a bug than a finding.
2. **An impossible cell.** Heavy payload × steep slope may exceed the robot's
   static tip-over margin. A cliff at a physically impossible cell says nothing
   about generalization. Check `env/payload.py::tip_over_margin_deg` against the
   OOD grid before interpreting.
3. **A broken baseline.** A Nav2 that fails in-distribution makes any policy
   look robust. Confirm its in-distribution success rate first.
