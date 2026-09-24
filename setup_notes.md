# VT ARC Setup & Verification Notes

Runtime is **VT ARC** (SLURM), on **NVIDIA L40S** nodes (48 GB, RT cores —
a supported Isaac Sim GPU). A100 / H200 remain available as a headless-only
fallback. Everything that could not be verified on the code-writing
machine is collected here as an ordered checklist. Work through **Part 2
before launching any long run** — several items fail *silently*, and a silent
failure here produces a complete, plausible-looking set of results for an
experiment that never actually varied the thing it claims to study.

General ARC operating knowledge (QOS, mail, paste corruption, the zero-byte
interpreter) comes from the VLM project's `arc_runbook.md` / `arc_quickref.md`.
Where this project differs, this file wins.

---

## Part 0 — Decisions (and why)

| Decision | Choice | Why |
|---|---|---|
| GPU | **1 × L40S** per job | Has RT cores, so Isaac Sim is a *supported* config (rendering, cameras, video all work). A100/H200 have none — headless physics may run, but that is unsupported and rules out a camera sensor. 48 GB is ample for a single-GPU PhysX sim + ~400k-param MLP. Fallback: 1 × A100, headless only. |
| Parallelism | **Job arrays, not multi-GPU** | Seeds are independent runs: `--array=0-2` → 3 one-GPU jobs that backfill separately. No NCCL, no DDP. |
| QOS | the L40S partition's **short / highest-priority** QOS | Check the name (Part 1.1). On the A100s it was `*_normal_short`: highest priority, 24 h cap. Every job here fits under 24 h. |
| Walltime | 8 h train / 4 h eval / 45 min smoke | Small + short is what backfills. A walltime kill is recoverable (`--resume auto`). |
| Install | **pip Isaac Sim in a dedicated conda env** (`rtn`) | No Docker on ARC; no root needed; same absolute-path-env pattern already proven there. Apptainer is the fallback. |
| Versions | Isaac Sim **4.5.0** · Isaac Lab **v2.1.0** · Python 3.10 · torch 2.5.1/cu121 | The code imports the Isaac Lab 2.x namespace (`isaaclab.*`). The earlier 4.2 / v1.4 pin used `omni.isaac.lab.*` and would have failed at import. |

---

## Part 1 — Environment setup (once, on the login node)

### 1.1 Account, partition, QOS

The accounts are active (confirmed 2026-09). Launchers never hardcode one
(public repo) — pass `--account=` on every `sbatch`.

The L40S names are **not yet confirmed** — the launchers assume
`--partition=l40s_normal_q --gres=gpu:l40s:1` and set no QOS. Check once:

```bash
sinfo -o "%P %G %D %N" | grep -i l40s                       # partition + gres string
sacctmgr show qos format=name%32,priority,maxwall | grep -i l40s
sacctmgr show assoc user=$USER format=account%30,partition%20,qos%80
```

If the names differ, edit the two `#SBATCH` lines in `arc/train.slurm` and
`arc/eval.slurm` (a wrong name fails loudly at submit — cheap). If a
short/high-priority L40S QOS exists, add it to `S` below as `--qos=...`.

**Which cluster?** If the L40S nodes live on a different ARC cluster than the
A100s (e.g. Falcon vs Tinkercliffs), `sbatch` must be run from *that*
cluster's login node, and `setup_env.sh` should be run there too — its glibc
preflight is only meaningful on the OS the jobs will actually use.

### 1.2 Checkout + install

```bash
bind 'set enable-bracketed-paste off'          # VS Code terminal paste corruption
cd ~/ondemand/data
git clone git@github.com:Manas-Ganti/robotics-rl-payload-transport.git
cd robotics-rl-payload-transport

bash arc/setup_env.sh --check                  # glibc, conda, free space, quota
bash arc/setup_env.sh                          # ~30-60 min; idempotent, re-run on failure
```

`setup_env.sh` creates `~/miniconda3/envs/rtn`, installs torch → isaacsim →
Isaac Lab → `requirements.txt`, then verifies with metadata checks and the
pure-logic `pytest` suite. Override locations with `RTN_ENV=`, `CONDA_BIN=`,
`ISAACLAB_DIR=` if needed.

- **Never** reuse `vrr` / `vrr-train` / `vrr-gen`. Isaac Sim pins torch 2.5.1
  and numpy < 2; crossing envs fails at import, late.
- W&B: the launchers source `~/.config/vrr/secrets.env` (already holds the
  W&B key and optional Telegram creds). Override with `RTN_SECRETS=`.

**Disk (in `$HOME`, 640 GB quota, ~334 GB already used by the VLM project):**

| Item | Size |
|---|---|
| `rtn` env (torch + isaacsim[all,extscache] + Isaac Lab) | ~30–35 GB |
| Kit / shader / compute caches (`~/.cache/ov`, `~/.nv`) | ~5–10 GB |
| Checkpoints (~5 MB each × 30 per run × ~10 runs) | ~1.5 GB |
| Eval CSVs, W&B, SLURM logs | < 1 GB |
| **Total** | **~40–50 GB** |

### 1.3 Fallback: Apptainer (only if preflight fails on glibc)

pip Isaac Sim needs glibc ≥ 2.34. If ARC's nodes are older, use Isaac Lab's
documented cluster route (`docker/cluster/` in the Isaac Lab repo): build the
image from this `Dockerfile` on a Docker-capable machine, `apptainer build` it
into a `.sif`, copy to ARC, and in `arc/arc_env.sh` set `PY` to
`apptainer exec --nv <sif> /isaac-sim/python.sh`. Not pre-wired — decide only
if the preflight forces it.

### 1.4 Branch discipline (two checkouts)

Edit on the Mac, **push, then `git pull --ff-only` on ARC before submitting.**
`.slurm` files are copied at *submit* time (a queued job ignores later edits);
Python and `arc_env.sh` are read at *run* time (a pull under a pending job
does take effect). The log line `[arc_env] git=<sha>` records what ran, and
flags `(DIRTY)` if the ARC checkout had uncommitted edits.

---

## Part 2 — The verification checklist

Ordered by *how badly a silent failure corrupts the study*, not by module.

### ⚫ TIER 0 — does Isaac Sim start on the node? (smoke test 0a)

On L40S this is a supported config, so this tier should pass — but a first
launch on a new cluster can still fail on drivers, and it is cheaper to learn
that at 64 envs / 45 min than at 2048 envs. (On the A100/H200 fallback the
risk is real: no RT cores, rendering unsupported, headless physics only.)

- [ ] Log reaches `=== Building environment ===` without a Vulkan / RTX /
      `Failed to create any GPU devices` error.
- [ ] If Kit dies on Vulkan: check the node has an NVIDIA ICD
      (`srun --jobid=<id> --overlap ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d`).
      No ICD on compute nodes → ARC support ticket, or the Apptainer route
      (`--nv` binds the driver's Vulkan libs).
- [ ] `nvidia-smi` in the log header shows an **L40S** (not a fallback GPU).
- [ ] Video is *possible* on L40S but **not wired**: the `logging.video*`
      keys exist in `train.yaml` and nothing reads them yet. Once training
      works, wiring it (`--enable_cameras` + a `RecordVideo` wrapper) is how
      the portfolio gets its demo GIF.
- [ ] First launch is slow (extension + shader cache build, 5–15 min). The
      second smoke test should start noticeably faster; if not, the cache
      dir is not persisting (`~/.cache/ov`).

### 🔴 TIER 1 — silent failures that invalidate results

These produce **no error**. If any one is broken, the corresponding robustness
curve is a flat artifact that reads as an exciting finding.

- [ ] **Per-env friction actually changes** — `env/nav_env.py::_apply_friction`
      Verify `root_physx_view.set_material_properties(props, indices)` signature
      and that the shape axis is right.
      **Test it physically:** set friction to 0.05, drive at full speed, confirm
      the robot visibly slips. *Absence of an error is not evidence it worked.*
      → If inert: the friction OOD sweep returns a flat curve that looks like
      genuine friction robustness.

- [ ] **Payload mass writes actually land** — `env/payload.py::set_payload_mass_batched`
      and `::apply_mass_modifier`. `set_masses` typically wants a **CPU tensor**
      and shape `(num_envs, num_bodies)` even in the GPU pipeline.
      **Test it:** read masses back after writing; assert they changed.
      → If inert: the entire payload axis — the project's headline dynamics
      variable — does nothing.

- [ ] **Inertia is updated with mass.** PhysX does **not** rescale inertia
      automatically. If it doesn't, call `set_inertias` using
      `env/payload.py::box_inertia_diagonal`, or heavy payloads will spin up
      unrealistically easily and heavy-payload results will be too optimistic.

- [ ] **Obstacle poses actually move** — `env/nav_env.py::_apply_terrain_to_sim`
      Confirm `write_root_pose_to_sim` moves a **kinematic** rigid body.
      → If inert: obstacles stay parked at z = −50, the robot drives through
      empty space, and collision rate is implausibly low across every cell.

- [x] **The depth sensor sees the obstacles** — *resolved by design.* Isaac
      Lab's `RayCaster` only handles static meshes and was likely blind to the
      re-posed kinematic obstacles. The default is now `sensors.modality:
      analytic_lidar` (`env/lidar.py`): exact ray–circle casting in torch
      against the episode's `TerrainSpec`, unit-tested locally against a
      brute-force oracle (`tests/test_lidar.py`). The old RayCaster path is kept
      as `modality: raycaster` for comparison only.
      **Its one blind spot:** it reads the *spec*, not the sim. If obstacles
      fail to move (item above), the lidar still reports them — ghosts. The
      smoke test now compares every active obstacle prim's pose to the spec and
      prints `FAIL:` if they differ by > 5 cm.

- [ ] **Obstacle prim radii match their slots** — obstacles are bound to pool
      slots (`Obstacle.slot`) and slot *k* is spawned with
      `pool_slot_radii(...)[k]`, so sim, A* and lidar share one radius. (This
      replaced a latent bug: every prim used to spawn at the *mean* radius
      while the planner assumed sampled ones.) Verify by printing a few
      `Obstacle_k` radii from the stage after cloning.

`training/train.py --smoke-test` checks several of these automatically and
prints explicit `FAIL:` lines. **Run it first.**

### 🟠 TIER 2 — loud failures (they crash; fix and move on)

- [ ] **Isaac Lab import paths** — `isaaclab.*` vs legacy `omni.isaac.orbit.*`
      (`env/nav_env.py` top-level imports).
- [ ] **Robot cfg import path** — `configs/robot.yaml: robot.cfg_import_path`
      (`isaaclab_assets.robots.jetbot.JETBOT_CFG`). Fallback: `CARTER_CFG`.
- [ ] **Wheel joint names** — print `robot.joint_names`; reconcile with
      `robot.left_wheel_joint` / `right_wheel_joint`.
- [ ] **Chassis body name** — `robot.base_body_name` must match the USD, or the
      contact sensor and raycaster attach to nothing.
- [ ] **Actuator mode** — does the robot take `set_joint_velocity_target`, or an
      effort command? Wrong mode ⇒ robot does not move at all.
- [ ] **`LidarPatternCfg` field names** — `horizontal_fov_range` vs
      `horizontal_fov` differ across versions.
- [ ] **`PhysxCfg` GPU buffer keys** — renamed between releases; an unknown key
      may be dropped silently rather than rejected.
- [ ] **rsl_rl config field names** — `training/ppo_config.py::build_runner_cfg`.
- [ ] **`RslRlVecEnvWrapper` import path** — `isaaclab_rl.rsl_rl` vs the older
      `omni.isaac.orbit_tasks.utils.wrappers.rsl_rl`.
- [ ] **Inference policy accessor** — `runner.get_inference_policy(device=...)`
      vs `runner.alg.actor_critic.act_inference` (`eval/run_eval.py`).

- [ ] **Resume restores the iteration counter** — `training/train.py` trains
      `max_iterations - runner.current_learning_iteration`. Confirm the log
      prints e.g. `Training 1200 iterations (1800/3000 done)` after a resume,
      not `(0/3000 done)` (which would silently train 3000 more).
- [ ] **`isaaclab.sh` used the `rtn` python** — its install output names the
      interpreter; it must be `$RTN_ENV/bin/python`.
- [ ] **W&B reaches the internet from compute nodes.** If `wandb.init` hangs
      or errors, rerun with `--set logging.wandb.mode=offline` and
      `wandb sync results/runs/<run>/wandb/offline-run-*` from the login node.

### 🟡 TIER 3 — physics correctness

- [ ] **Quaternion convention** is `(w, x, y, z)` — `_robot_yaw`,
      `_apply_ground_slope`, `_reset_robot_state`. Wrong ⇒ headings rotated 90°
      or mirrored; the robot drives sideways relative to its goal.
- [ ] **Slope sign** — a rotation about **+y** must make **+x** uphill, matching
      `TerrainSpec.height_at`. If they disagree the robot slides the wrong way
      and the "uphill" start/goal placement is actually downhill.
- [ ] **Contact force history shape** —
      `(num_envs, history_length, num_bodies, 3)` in `_detect_collision`.
- [ ] **Collision threshold** (`env.collision_force_threshold: 1.0` N) — tune
      it. Too low ⇒ ground contact registers as a crash and everything fails.
      Too high ⇒ real collisions are missed and success rate is inflated.
- [ ] **Payload FixedJoint survives `sim.reset()`** — `attach_payload_joint`.
      If the payload visibly lags, falls through, or drops, start here.
- [ ] **Restitution 0.0** suits the robot — a bouncy chassis corrupts
      collision-based terminations.

### 🟢 TIER 4 — performance & scale

- [ ] **Scene budget.** Kinematic colliders = `max_obstacles_per_env × num_envs`
      = 48 × 2048 ≈ **98k**. If PhysX errors or startup crawls, lower
      `env.num_envs` to 1024 or 512 **first** — it costs wall-clock, not validity.
- [ ] **Reset cost.** Terrain generation (NumPy + A*) is CPU-side, per env, per
      reset. If resets dominate the step time, pre-generate a pool of specs per
      parameter set and sample from it instead of generating fresh each time.
- [ ] **GPU buffer sizes** — raise `sim.physx.gpu_*` if PhysX reports overflow.
- [ ] **Solvability budget.** At `obstacle_density = 0.9`, watch for
      `UnsolvableLayoutError`. Raise `terrain.max_generation_attempts` before
      touching the density range — the range is the study, the budget is not.

### 🔵 TIER 5 — Nav2 baseline (Phase 3)

**On ARC, use `mode: inproc_planner` only.** ROS 2 is not installed on the
cluster, and the bridge mode would add a second runtime to debug on nodes you
cannot see. Everything below the next paragraph applies only off-cluster.

**Recommendation: run `mode: inproc_planner` for the full grid.** It needs no
ROS 2, batches across all envs, and reuses this repo's own A* — the same planner
that certified each episode solvable. The ROS 2 path is a spot-check, not the
main run.

For `mode: ros2_bridge` only:

- [ ] ROS 2 (Humble/Iron) sourced in the **same shell** as Isaac Sim.
- [ ] `rclpy` importable from **Isaac's bundled Python** — a system-installed
      rclpy will not import.
- [ ] `omni.isaac.ros2_bridge` extension enabled; `/scan`, `/odom`, `/tf` publishing.
- [ ] Nav2 lifecycle nodes ACTIVE: `ros2 lifecycle get /bt_navigator`.
- [ ] `navigate_to_pose` action server reachable.
- [ ] `_act_ros2` implemented (it currently raises `NotImplementedError` —
      the topic wiring depends on what the bridge exposes on your machine).
- [ ] **cmd_vel converted to `[-1, 1]` using the same `action.*` ranges as the
      policy** — otherwise the baseline is effectively driving a different robot
      and the head-to-head is meaningless.

> **Before reading anything into the OOD columns: confirm Nav2 achieves a high
> in-distribution success rate.** A broken baseline makes any policy look robust.

---

## Part 3 — Phase run order (sbatch)

All from the repo root on ARC. `S` is shorthand — put it in your shell, not in
a file (it carries your email):

```bash
S="sbatch --account=<ACCT> --mail-user=<you>@vt.edu"
```

Everything after the `.slurm` name is passed to the Python entrypoint's
argparse — a typo fails loudly. Do **not** pass settings as `VAR=x sbatch`;
the launchers read exactly one env var (`RTN_ENV`) and would drop the rest.

```bash
# ---- Phase 0 — bring-up. Two passes. Read every FAIL: line. -------------------
# (a) tiny scene: does Isaac Sim start at all on a datacenter GPU (see Tier 0)?
$S --time=00:45:00 arc/train.slurm --smoke-test --num-envs 64
# (b) full scene budget: 48 obstacles x 2048 envs (~98k colliders), PhysX buffers
$S --time=00:45:00 arc/train.slurm --smoke-test

# ---- Phase 1 — flat floor, obstacles, friction, payload, reward v1 ------------
$S arc/train.slurm --run-name p1 --resume auto
#   short sanity run first if you like:  ... --run-name p1_short --max-iterations 200

# ---- Sanity floor + Nav2 in-distribution check (can run alongside Phase 1) ----
$S arc/eval.slurm --policy random --tag random
$S arc/eval.slurm --policy nav2   --tag nav2

# ---- Phase 2 — slope axis, 3 seeds in parallel, then the OOD sweep ------------
$S --array=0-2 arc/train.slurm --run-name p2_slope --resume auto --seed-from-array \
     --set phases.enable_slope=true
#   -> results/runs/p2_slope_s42, _s43, _s44
$S arc/eval.slurm --checkpoint results/runs/p2_slope_s42/model_final.pt --tag p2_s42
#   (repeat per seed; if a run hit walltime, use its highest model_<N>.pt)

# ---- Phase 4 — sensor degradation + transport-aware reward v2 -----------------
$S --array=0-2 arc/train.slurm --run-name p4 --resume auto --seed-from-array \
     --set phases.enable_slope=true --set phases.enable_sensor_noise=true \
     --set phases.enable_reward_v2=true

# ---- Phase 5 — inferred payload (told-vs-inferred is this one flag) -----------
$S --array=0-2 arc/train.slurm --run-name p5_inferred --resume auto --seed-from-array \
     --set phases.enable_slope=true --set phases.enable_inferred_payload=true

# ---- Phase 6 — figures: on the Mac, from copied-back CSVs (no GPU) ------------
python analysis/plots.py --results-dir results/eval --out-dir results/figures
```

**Walltime kill ≠ lost run.** Resubmit the *identical* line: `--resume auto`
finds the highest `model_<N>.pt` in that run's directory, trains only the
remaining iterations, and appends to the same W&B run.

**Monitoring:**

```bash
squeue -u $USER -o "%.10i %.14j %.9T %.11M %.11L %.22R %N"
tail -f logs/slurm/rtn-train-<jobid>.out
srun --jobid=<id> --overlap nvidia-smi          # is it actually computing?
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

**A job that "completes" in seconds with exit 0 is not a success.** Check the
`[arc_env] python=...` line; `arc_env.sh` aborts loudly on a broken
interpreter, but confirm it printed at all.

**Budget (estimates, not measurements):** ~147M env steps per training run →
1.5–4 GPU-h each; ~10 training jobs (P1 + 3 seeds × P2/P4/P5) + ~6 eval sweeps
≈ **25–45 GPU-hours** (L40S physics throughput is comparable to A100 for this workload), plus bring-up. Measure the real steps/s from the Phase 1
log and re-size `--time` from that.

---

## Part 4 — What to bring back

`results/` is gitignored, so copy it back with `rsync`/`scp` (or the ARC
OnDemand file browser). The whole analysis then reruns on the Mac with no GPU:

```
results/eval/episodes_*.csv     # per-episode records — any metric recomputable
results/eval/cells_*.csv        # per-grid-point aggregates — what plots.py reads
results/eval/summary_*.json     # degradation profiles + classifications
results/runs/*/resolved_config.json   # exact config that produced the run
```

Bring back **failures too** — a stack trace, a flat curve, an implausible
success rate. The `# VERIFY ON A100:` / `# VERIFY ON ARC:` list above is where to look first, and a
flat curve is far more often an inert axis (Tier 1) than a real finding.

---

## Part 5 — Known-uncertain design decisions

Judgement calls made while writing without a simulator. Worth revisiting once
there is real data:

1. **`obstacle_density` is a normalized knob, not a literal area fraction.** It
   scales to `terrain.max_obstacle_area_fraction` (0.30). Read literally, the
   OOD point 0.9 would mean 90% floor coverage — trivially unsolvable, so the
   top of the sweep would measure the regeneration budget rather than the policy.
2. **Obstacle pool cap of 48/env** bounds the achievable density. If
   `TerrainFactory` warns that the cap bound before the density target was met,
   the high-density curve is flattened by the *scene budget*, not the policy.
3. **Payload attach mode defaults to `mass_modifier`** (fast, no separate body).
   Switch to `rigid_body_with_joint` before drawing conclusions from the v2
   transport reward — penalizing jerk to protect a payload with no independent
   dynamics is not measuring what it claims to.
4. **Collision threshold of 1.0 N is a guess.** Calibrate against observed
   ground contact forces during bring-up.
5. **Start/goal are placed along the slope gradient** so the slope axis is
   actually exercised. A pair placed across the contour would leave slope nearly
   irrelevant and flatten the Phase 2 curve for the wrong reason.
6. **`tip_over_margin_deg`** (`env/payload.py`) — run it across the OOD grid
   *before* the sweep. If heavy-payload × steep-slope cells are physically
   impossible rather than merely hard, a degradation cliff there says nothing
   about generalization, and the write-up must say so.
