# CLAUDE.md

## Project: Robust Payload-Transport Navigation under Terrain & Dynamics Shift

An RL research project training a mobile robot to **transport a variable-mass payload
across parameterized, sloped terrain**, then measuring how the learned policy generalizes
to **out-of-distribution (OOD) terrain, friction, and payload** — benchmarked against a
**classical Nav2 baseline**.

---

## HOW TO WORK ON THIS PROJECT (read first)

**This is a CODE-FIRST workflow. Do not run anything.**

- The development machine (where you, Claude, are running) has **no GPU and does not
  execute this code**. Isaac Sim/Isaac Lab is **not installed here and must not be
  invoked**.
- Your job is to **write the complete codebase** — all modules, all phases, configs,
  tests, docs — as clean, runnable, well-structured code that the user will transfer to
  a remote **A100** machine to actually run.
- **Write all phases now, in full.** Do not stop after Phase 1 waiting for results. The
  entire repository should exist and be internally consistent by the time you finish.
- Optimize for: **correctness on first read, runnability on the A100, and easy iteration.**
  The user will run on the A100, bring back results/logs/errors, and ask you to adjust.
  So the code must be easy to modify per-phase without rearchitecting.
- Where an Isaac Lab API detail is uncertain (versions differ: Isaac Gym -> Orbit ->
  Isaac Lab), **write against the current Isaac Lab API, isolate the uncertain call
  behind a clearly-commented function, and add a `# VERIFY ON A100:` note** so the user
  knows exactly what to check. Never silently guess and bury it.
- **Do not execute, pip install, or launch simulators locally.** Write code and configs
  only. Static review is fine; running is not.

**Deliverable of the local session:** a complete, transfer-ready repo the user can `scp`
to the A100, `pip install`, and run phase by phase.

---

## The One Finding We Are Hunting

> *How far beyond its training distribution (terrain slope, friction, payload mass) can a
> learned transport policy generalize before failing — gracefully or catastrophically —
> compared to a classical Nav2 planner under the same shifts?*

Every module exists to enable answering this. Code that does not contribute is out of scope.

---

## Non-Negotiable Principles (encode these in the code)

1. **Parameterized, not hand-designed.** Terrain/friction/payload generated from config
   via a factory. No hand-built levels. All ranges live in YAML, never hardcoded.
2. **Train/OOD split is sacred.** Training samples inner ranges; evaluation includes
   held-out outer ranges. The split is enforced in code (separate config blocks; an
   assertion that train and OOD ranges do not overlap).
3. **Solvability guarantee.** Every generated episode verifies a valid path to goal
   (A*/graph check) before spawn; unsolvable layouts are rejected and regenerated.
4. **Verifiable metrics.** Success/collision/path-efficiency computed mechanically
   against ground truth. No subjective scoring.
5. **Honest baseline.** Nav2 runs through the *same* eval harness as the policy.
6. **Reproducibility.** Global seeding, versioned YAML configs, W&B logging, Dockerfile.
7. **Payload replaces gravity randomization.** Payload is a real deployment condition;
   gravity is not. **Do NOT randomize gravity anywhere in the code.**

---

## Task Definition

- **Robot:** wheeled/differential-drive mobile base. Use an Isaac Lab built-in mobile
  robot; do not model a custom robot. Isolate the robot choice in `configs/robot.yaml`.
- **Mission:** transport a sampled-mass payload from start pose to goal pose across
  terrain without collision.
- **Observation (v1):** depth (or 2D lidar) + goal-relative pose (dir + distance) +
  proprioception (lin/ang vel) + payload_mass (included in v1; toggleable off for the
  inferred-payload experiment via config flag).
- **Action:** continuous (linear vel, angular vel).
- **Termination:** goal reached (success) | collision (fail) | timeout (fail).

---

## Independent Variables (study axes -- put in configs)

| Axis | Parameter | Training (inner) | Held-out OOD (outer) |
|---|---|---|---|
| Geometry | obstacle_density | 0.2 - 0.6 | 0.7 - 0.9 |
| Terrain | slope_angle (deg) | 0 - 10 | 12 - 20 |
| Physical | friction_coeff | 0.4 - 0.9 | 0.1 - 0.3 |
| Dynamics | payload_mass (kg) | 1 - 5 | 6 - 9 |
| Sensor | depth_dropout/noise | mild | severe |

Numbers are defaults; keep them in YAML and keep the inner/outer split. Code must assert
no overlap between train and OOD ranges at load time.

---

## Reward Design (implement both tiers; default to v1 via config)

**v1 (base, default):**
```
+ goal_reached           (terminal)
- collision              (terminal penalty)
- step_cost              (efficiency / anti-stall)
+ progress_shaping       (dense: distance-to-goal reduction per step)
```
**v2 (transport-aware, config-toggle):**
```
- excessive_jerk/accel   (smooth transport)
- energy/time under load
```
Make reward terms **individually toggleable and individually weighted via config**, so the
user can ablate them on the A100 without code changes. Add a docstring in `reward.py`
reserving space for a "reward-design failure history" the user will fill in with observed
failure modes.

---

## Tech Stack

- **Sim:** Isaac Sim + Isaac Lab (target runtime = A100). Write against current Isaac Lab.
- **RL:** PPO default (Isaac Lab-supported), SAC behind a config switch for later
  comparison. Prefer Isaac Lab's RL integration (e.g. rsl_rl / rl_games wrapper) over a
  hand-rolled trainer -- isolate the choice so it can be swapped.
- **Baseline:** ROS 2 Nav2, run through the shared eval harness.
- **Tracking:** Weights & Biases.
- **Language:** Python, type hints, modular (no notebooks).
- **Packaging:** Dockerfile (A100/CUDA base), YAML configs, requirements.txt.

---

## Repository Structure (write ALL of this)

```
robust-transport-nav/
|- README.md                 # env-first framing, how-to-run on A100, results table stub
|- CLAUDE.md                 # this file
|- Dockerfile                # CUDA/Isaac-compatible base for the A100
|- requirements.txt
|- setup_notes.md            # A100 setup steps + every `# VERIFY ON A100:` item collected
|- configs/
|  |- train.yaml             # inner param ranges, PPO hyperparams, reward weights/toggles
|  |- eval_ood.yaml          # held-out OOD grid definition
|  |- robot.yaml             # robot + sensor config
|- env/
|  |- terrain_factory.py     # procedural terrain: slope, scattered obstacles, friction
|  |- payload.py             # variable-mass payload attach + dynamics
|  |- solvability.py         # A*/graph path-existence check + regenerate-on-fail
|  |- randomization.py       # domain-randomization sampler (reads train ranges)
|  |- nav_env.py             # Isaac Lab env: reset/step/obs/reward/termination
|  |- spaces.py              # obs/action space defs (single source of truth)
|- training/
|  |- train.py               # entrypoint: config -> env -> PPO -> W&B -> checkpoints
|  |- ppo_config.py          # algo hyperparams (or reference to configs/train.yaml)
|- baselines/
|  |- nav2_runner.py         # drives Nav2 through the shared eval harness
|- eval/
|  |- ood_harness.py         # sweeps OOD grid for ANY policy (RL or Nav2), logs metrics
|  |- metrics.py             # success, collision, path efficiency, graceful-degradation
|  |- run_eval.py            # entrypoint: checkpoint + eval_ood.yaml -> results table
|- analysis/
|  |- plots.py               # robustness curves, RL-vs-Nav2 overlays, failure taxonomy
|- tests/
|  |- test_reward.py         # reward term edge cases (runs WITHOUT Isaac -- mock state)
|  |- test_solvability.py    # path-existence check on synthetic grids (no Isaac)
|  |- test_config.py         # asserts train/OOD ranges don't overlap; schema valid
|- results/                  # (empty; populated on A100)
```

**Important for testability:** write `tests/` so they run on the **local, GPU-less
machine** -- reward logic, solvability, and config validation must be importable and
testable WITHOUT Isaac Sim. Structure `reward.py`, `solvability.py`, and config loading as
pure functions over plain arrays/dicts, so the user can `pytest` locally before touching
the A100. Isaac-dependent code (`nav_env.py`) is written but not unit-tested locally.

---

## What Each Phase Means in a CODE-FIRST World

Phases are no longer "run then proceed" -- they are **modules to write completely now**,
plus a note on what the user will verify on the A100. Write all of them.

- **Phase 0 (bring-up):** `nav_env.py` skeleton -- reset/step loop with a built-in robot,
  sensors wired, random-action smoke-test script. `# VERIFY ON A100:` robot spawns/senses/moves.
- **Phase 1 (minimal task):** flat floor + scattered obstacles + solvability + static known
  payload + friction randomization + reward v1 + `train.py`. Fully written.
- **Phase 2 (robustness study):** slope axis + `ood_harness.py` + `metrics.py` +
  `eval_ood.yaml`. Full OOD sweep for a policy checkpoint.
- **Phase 3 (baseline):** `nav2_runner.py` driving Nav2 through `ood_harness`. Mark Nav2
  wiring `# VERIFY ON A100:`.
- **Phase 4 (sensor + transport reward):** depth-degradation randomization axis + reward
  v2 toggles.
- **Phase 5 (stretch, inferred payload):** config flag to drop payload_mass from obs;
  written so told-vs-inferred is a one-line config change.
- **Phase 6 (analysis/packaging):** `plots.py` producing all figures from results CSVs;
  README results-table stub; Dockerfile; `setup_notes.md`.

Each module includes `# VERIFY ON A100:` comments wherever an Isaac Lab API call, Nav2 hook,
or GPU-specific behavior can't be checked locally. Collect all of these into `setup_notes.md`
as a checklist.

---

## Config-Driven Everything (critical for the iterate-on-A100 loop)

The loop is: run on A100 -> see results -> tweak -> rerun. Minimize code edits in that loop.

- **All tunables in YAML:** param ranges, reward weights + toggles, PPO hyperparams,
  episode length, obs components, phase flags.
- **No magic numbers in code.** Every constant traces to a config key.
- **Phase/feature flags** so the user enables slope, sensor-degradation, transport-reward,
  or inferred-payload from config, not by editing modules.
- **Single entrypoints:** `training/train.py --config ...` and
  `eval/run_eval.py --checkpoint ... --config ...`. Clear CLI args. Reproducible from
  config + seed.

---

## Deliverables of the Local (Code-Writing) Session

- [ ] Every file in the tree above written and internally consistent
- [ ] `env/` complete: terrain factory, payload, solvability, randomization, Isaac Lab env
- [ ] `training/train.py` runnable-on-A100 entrypoint, W&B + checkpointing
- [ ] `eval/ood_harness.py` + `metrics.py` + `run_eval.py` complete
- [ ] `baselines/nav2_runner.py` complete (Nav2 hooks marked for A100 verification)
- [ ] `analysis/plots.py` producing all study figures from result CSVs
- [ ] `configs/*.yaml` with train/OOD split + assertions
- [ ] `tests/` that pass LOCALLY without Isaac (reward, solvability, config)
- [ ] Dockerfile + requirements.txt + setup_notes.md (with all `# VERIFY ON A100:` items)
- [ ] README: framing + A100 run instructions + results-table stub

**Never omit:** the train/OOD split + overlap assertion, the solvability check, the shared
eval harness usable by both RL and Nav2, config-driven tunables, locally-runnable tests.
**Lowest priority (write last, simplest form):** inferred-payload (Phase 5), SAC switch,
uneven heightfield terrain, RGB observations (depth-only is fine for v1).

---

## Conventions for the Assistant

- Write the **whole tree** in this session; do not defer modules waiting for run results.
- Never execute code, install packages, or launch a simulator here. Write only.
- Isolate every uncertain Isaac Lab / Nav2 API call behind a small function with a
  `# VERIFY ON A100:` comment; mirror it into `setup_notes.md`.
- Keep Isaac-dependent code separate from pure logic so pure logic is locally testable.
- All parameter ranges, weights, and flags go to YAML -- no hardcoded constants.
- When adding a reward term, add its local unit test in the same change.
- Prefer extending Isaac Lab's built-in env/terrain utilities over reinventing them; where
  you reference them, name the specific Isaac Lab module and flag for verification.
- Ask before adding a new dependency or introducing a second simulator.