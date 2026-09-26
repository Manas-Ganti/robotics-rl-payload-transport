# Isaac Sim + Isaac Lab on VT ARC — Playbook

**Audience:** a future agent (or person) setting up or extending an Isaac Lab
project on VT ARC. It records what it took to go from "code written on a laptop"
to "Phase 0 passes and a policy trains" (2026-09-24/25). Every item cost at least
one GPU job to discover. Read this before writing Isaac code or submitting a job.

**Status labels used below**

- **VERIFIED**: observed on ARC hardware, with the evidence stated.
- **UNVERIFIED**: written or designed, but not yet exercised on ARC.
- **OPEN**: a known anomaly that has not been explained yet.

---

## 0. The ten rules (the short version)

1. **No error does not mean it works.** Most Isaac/PhysX failures are silent:
   the sim runs and produces plausible, wrong numbers. Every study axis needs a
   **forced-condition probe** that measures the physical response (§7).
2. **Read values back from PhysX** (`root_physx_view.get_*`), never from your
   own buffers, to confirm a write landed.
3. **Diagnose from evidence before fixing.** A guessed fix (joint-state reset for
   a tilt bug) cost a queue round and changed nothing. The real cause (spawns at
   the patch edge) was visible in data already collected.
4. **Measure robot geometry; never trust a bounding box.** Stale `extent`
   attributes and rotated cylinders both give wrong sizes (§4).
5. **Pin the version pair** Isaac Sim 4.5.0 ↔ Isaac Lab v2.1.0. The current
   online docs are for Isaac Sim 5.x and differ (asset paths, APIs).
6. **Isaac Lab calls `_get_dones()` BEFORE `_get_rewards()`.** Compute per-step
   state in `_get_dones` (§5.6).
7. **PhysX warnings are log lines, not exceptions.** Grep every job log for
   `PhysX error` (the launcher does this automatically).
8. **Use an RT-core GPU (L40S) on ARC.** A100/H200 have no RT cores. Headless
   physics may run there, but rendering and video will not.
9. **Do cheap checks on the login node first:** pytest, USD inspection,
   solvability sweep. They are seconds of CPU and need no queue.
10. **Batch fixes per `sbatch` round.** Each round costs a queue wait. Before
    pushing, review the neighbouring code paths for the same class of bug.

---

## 1. Versions and install (VERIFIED)

| Component | Version | Why |
|---|---|---|
| Isaac Sim | **4.5.0** via pip (`isaacsim[all,extscache]`) | ARC has no Docker. `extscache` avoids extension downloads on compute nodes. |
| Isaac Lab | **v2.1.0** (git tag) | The code uses the 2.x namespace `isaaclab.*`. Lab 1.x (`omni.isaac.lab.*`) fails at import. |
| Python | 3.10 | Required by the Isaac Sim 4.5 wheels. |
| torch | 2.5.1+cu121 | Install **before** isaacsim. Never let a later install upgrade it. |
| numpy | < 2 | Isaac Sim 4.x. |
| protobuf | **< 5** | `isaaclab_rl` / `isaaclab_tasks` need it. See pitfall below. |
| setuptools | **< 75** | Needed to build `flatdict`. See pitfall below. |
| rsl-rl-lib | 2.3.1 (from `isaaclab.sh --install rsl_rl`) | Do not override the version. |

The one-shot installer is `arc/setup_env.sh`: preflight, then install, then verify. It uses a
dedicated conda env `~/miniconda3/envs/rtn`. **Never reuse or merge envs from other
projects** (e.g. vLLM/TRL envs): their torch/transformers pins conflict.

**Install pitfalls hit (all fixed in `setup_env.sh` / `requirements.txt`):**

- **Core `isaaclab` silently not installed.** Its extensions installed fine, but
  the core package failed on `flatdict==4.0.1`, whose `setup.py` imports
  `pkg_resources`. pip's isolated build environment pulls the newest setuptools,
  which no longer ships `pkg_resources`.
  - Fix: `pip install "setuptools<75"`, then `pip install flatdict==4.0.1 --no-build-isolation`, then `pip install -e source/isaaclab`.
  - Verify with `pip show isaaclab`. `find_spec` on the extensions alone is not enough.
- **protobuf 7 broke Isaac Lab.** An unpinned `wandb` pulled the newest release
  (0.30), which requires protobuf ≥ 5.
  - Fix: pin `protobuf>=3.20.2,<5` in `requirements.txt`; pip then selects wandb 0.28.
  - The wandb 0.30 install left orphans behind (`opentelemetry-*`, `googleapis-common-protos`). Check with `pip show … | grep Required-by` before uninstalling.
- **pip only warns about conflicts.** Verification runs `pip check` and fails on
  any conflict involving `isaac*` / `rsl*` packages.
- **Login-node `/tmp` is 20 GB,** which is too small for the isaacsim wheels.
  Set `TMPDIR=$HOME/tmp/...` and delete it afterwards (it holds about 20 GB of quota).
- **Don't re-run the full installer to fix one package.** It reinstalls torch and isaacsim.

**Disk:** the env is about 30–35 GB. Kit caches add 5–10 GB. The install peaks
about 20 GB higher (temp). Checkpoints are about 5 MB each. `df` on `/home` shows
the shared filesystem; **`quota`** shows your real limit.

---

## 2. VT ARC specifics (VERIFIED)

- **GPU choice:** L40S (48 GB, RT cores) is a supported Isaac Sim GPU.
  - It lives on the **Falcon** cluster. Submit from `falcon1`/`falcon2`; `/home` is shared with Tinkercliffs.
  - Partition `l40s_normal_q`, gres `gpu:l40s:1`. No QOS flag is needed.
  - A100/H200 (Tinkercliffs) have no RT cores: headless physics only.
- **One GPU per job.** Run seeds as SLURM job arrays (`--array=0-2`, plus `--seed-from-array`), not multi-GPU.
- **Launchers:** `arc/train.slurm`, `arc/eval.slurm` and `arc/record.slurm` all source `arc/arc_env.sh`, which:
  - uses an absolute env path and asserts that `python -V` prints (a 0-byte interpreter once made jobs "succeed" in 2 s)
  - checks the Isaac stack with `find_spec` (importing `isaaclab` submodules before AppLauncher is itself an error)
  - sets the EULA env vars and sources `~/.config/vrr/secrets.env` (W&B key, optional Telegram)
  - on exit, greps the job log for `PhysX error` and prints/notifies
- **Pass every setting as argparse passthrough** after the script name (`--set key=value`).
  `VAR=x sbatch` is silently dropped if the launcher doesn't read `VAR`.
- **`logs/slurm/` must exist before `sbatch`**; the repo tracks it with `.gitkeep`.
  Always set an explicit `--mem` (`--mem=0` means "whole node" and never backfills).
- **Account and email** go on the command line (`--account=`, `--mail-user=`),
  never in the public repo. `--mail-type=BEGIN,END,FAIL,TIME_LIMIT_80` is in the
  launchers. Outlook junk-filters the first SLURM mail.
- **W&B online works from Falcon compute nodes** (logged in via `WANDB_API_KEY`).
- **Two checkouts, one branch.** Edit on the Mac, push, then `git pull` on ARC.
  `.slurm` files are copied at submit time; Python is read at run time.

---

## 3. Throughput and timing (VERIFIED, L40S, Carter, 2048 envs)

| Item | Number |
|---|---|
| App start to env built | ~30 s (first launch builds caches, 5–15 min once) |
| Training throughput | **28–36k env-steps/s** (collection ~1.5 s, learning ~0.065 s per iteration) |
| PPO 3000 iterations × 24 steps × 2048 envs (147M steps) | **74–81 min** |
| Eval sweep (4 axes × 6 values × 100 episodes, 256 envs) | **3–5 min** |
| Smoke test, 64 envs | a few minutes once running |

Collection-bound: terrain generation plus A* runs on the CPU at every reset (use `--cpus-per-task=16`).

---

## 4. Assets (VERIFIED)

- **Isaac Sim 4.5 content layout is `Isaac/Robots/<Name>/…`.** Current docs show
  `Robots/NVIDIA/<Name>/…`, which is the 5.x layout and **returns 404 on the 4.5 server**.
  Base URL: `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/`
  - JetBot: `Robots/Jetbot/jetbot.usd`
  - Carter: `Robots/Carter/carter_v1.usd` (also `carter_v1_physx_lidar.usd`, `nova_carter.usd`, `nova_carter_sensors.usd`)
  - Probe with `curl -s -o /dev/null -w '%{http_code}' "$BASE/$p"`.
- **Use a local copy** (`--set robot.usd_path=/home/<user>/isaac_assets/...`). It avoids
  depending on the content server during jobs.
- **Sub-files are required.** Carter references `Props/carter_main.usd`,
  `carter_wheel_left/right/center.usd` and `carter_backwheel_caster.usd`. **They hold
  the collision shapes**; without them the robot falls through the floor.
  `Props/materials.usd` is visual only. Download it only for rendering.
- **Inspect a USD on the login node** with `arc/inspect_usd.py` in a throwaway
  `usd-core` venv (`python3 -m venv ~/usdtools && ~/usdtools/bin/pip install usd-core`).
  It prints joints and drive gains, body masses, collision shapes, footprint,
  wheel radius and unresolved dependencies.
- **Geometry-measurement traps:**
  1. **Stale authored `extent`** (the extents hint) on Carter primitives made
     `BBoxCache` report a 3 km robot. Compute extents with
     `UsdGeom.Boundable.ComputeExtentFromPlugins`, then transform the corners.
  2. **A rotated cylinder's bounding box overstates its radius.** It gave
     0.278/0.271 m; the truth is **0.24 m** (`radius` attr 0.5 × world scale 0.48).
     Read the radius from shape attributes.
  3. **Ground-truth any dimension by a physical measurement.** Resting root height
     above flat ground (**0.240 m**) confirmed the wheel radius. The 12.6% error
     made every command run at 87% speed.
  4. The USD ships its own ground (`/staticPlaneActor`) outside the default prim.
     Isaac Lab only references the default prim, so ignore it.
- **Harmless log noise:**
  - `UsdPreviewSurface.mdl` resolve warnings
  - `omni.isaac.dynamic_control is deprecated`
  - `improvePatchFriction` deprecation on Carter's `caster_material`
  - `Not all actuators are configured! 2 != 4` (Carter's casters are passive on purpose)

**Carter v1 facts (measured):**
- Joints: `left_wheel` and `right_wheel` (driven); `rear_pivot` and `rear_axle` (passive caster).
- Bodies: `chassis_link` 45.88 kg, `com_offset` 20 kg ballast (fixed-jointed behind the axle), wheels 4.2 kg each, caster 1.8 kg, imu 0.1 kg. **Total 76.2 kg.**
- Wheel radius **0.24 m**; separation **0.538 m** (collision-cylinder centres).
- USD wheel drive: velocity-style, damping about 1e6 (Isaac Lab units), **maxForce = inf**.
- **Two front drive wheels plus one rear caster. No front support:** it balances only because the ballast sits behind the axle.
- Footprint: 0.443 m, the farthest collision corner from the turning centre.
- Root rests 0.240 m above flat ground.

---

## 5. Isaac Lab 2.1 API gotchas (VERIFIED unless marked)

1. **`PhysxCfg` has no `solver_position/velocity_iteration_count`** (a TypeError).
   These counts are per actor in 2.x: set them on
   `robot.spawn.articulation_props` (`ArticulationRootPropertiesCfg`). `PhysxCfg`
   holds only the min/max clamps and the `gpu_*` buffers. Validate YAML keys
   against `PhysxCfg.__dataclass_fields__`.
2. **No robot configs for NVIDIA mobile bases in `isaaclab_assets`** (no JETBOT/CARTER).
   Define an `ArticulationCfg` inline (`env/robots.py::make_diff_drive_cfg`).
3. **Actuate only the drive joints** (`joint_names_expr=[left, right]`). Using `".*"`
   would also drive passive casters.
4. **For implicit actuators the physics torque cap is `effort_limit_sim`.** (For
   velocity, plain `velocity_limit` is ignored; `effort_limit` is only an alias or
   deprecated.)
5. **`ContactSensor` needs `activate_contact_sensors=True`** on the spawn cfg.
   Filtering:
   - `filter_prim_paths_expr` needs **one expression per filtered prim per env**.
     A wildcard matching 48 obstacles per env fails at sim start: `Filter pattern … expected 64, found 3072`.
     It logs an error and leaves `force_matrix_w` empty or zero. It does **not** raise.
   - One sensor body against many filtered bodies works.
   - Use `data.force_matrix_w` (obstacle contacts), **never `net_forces_w`**: that includes ground support and flags every step as a collision.
6. **`DirectRLEnv.step()` order is `_get_dones` → `_get_rewards` → `_reset_idx` → `_get_observations`.**
   Computing goal and collision flags in `_get_rewards` made terminations one step
   late and **double-counted every terminal reward**. Compute step state in `_get_dones`.
7. **Spawning creates only the leaf prim.** `/World/envs/env_.*/Obstacles/Obstacle_k`
   fails if `Obstacles/` does not exist, so spawn flat (`env_.*/Obstacle_k`).
8. **`RayCaster` only hits static meshes** (and older versions only one mesh). It is
   blind to kinematic obstacles re-posed each episode. We replaced it with an
   **analytic lidar** (`env/lidar.py`): exact ray–circle casting in torch against the
   episode's spec, one implementation for numpy tests and torch.
9. **PhysX view writes** (`set_material_properties`, `set_masses`, `set_inertias`):
   **CPU tensors**, full `(N, …)` read-modify-write, then write back with indices
   (Isaac Lab's randomizers do this). Writing only the reset rows misassigns envs.
   **PhysX does not rescale inertia when mass changes**, so write inertia too.
10. **Kinematic `RigidObject` poses** set via `write_root_pose_to_sim` do move
    (verified: 0.000 m sim-vs-spec error). A tilted kinematic cuboid works as
    per-env sloped ground. Rotating it with **R_y(−θ)** makes +x uphill; R_y(+θ)
    sends +x downhill.
11. **Friction combine modes:**
    - PhysX picks the higher-priority mode of the two materials (average < min < multiply < max).
    - Ground uses `friction_combine_mode="min"` so the sampled ground friction governs.
    - **Carter's tyre material is low-friction**, so min(ground, tyre) was always the tyre, and the friction axis was **inert** (launch slip identical at μ 0.9 and 0.1).
    - Fix: write `robot.tire_friction` (1.0) onto every robot contact shape at startup (`_apply_tire_friction`). This holds unless the tyre mode is `max`, which the probe would catch.
12. **Reset joint state as well as the root** (`write_joint_state_to_sim` with the
    defaults). Otherwise wheel spin and caster angle carry over between episodes.
13. **Seed Isaac Lab** via `env_cfg.seed`; otherwise it warns "Seed not set" and runs are nondeterministic.
14. **PhysX GPU buffers:** at 2048 envs × 48 obstacles, PhysX logged
    `increase … foundLostPairsCapacity to 2312192, otherwise the simulation will miss interactions`,
    meaning silently dropped contacts. `gpu_found_lost_pairs_capacity: 4194304` fixed it.
15. **Adding a body to the robot:** author an `Xform` with `RigidBodyAPI` and `MassAPI`
    under `/World/envs/env_0/Robot` plus a `UsdPhysics.FixedJoint` to the chassis,
    **before `clone_environments`**. It becomes an articulation link (same as Carter's
    `com_offset`) and PhysX composes mass, COM and inertia. Used for the payload
    (`env/payload.py::author_payload_link`).
16. **Out-of-bounds:** nothing stops a robot driving off a floating ground patch.
    Terminate on it (treated as a collision with the boundary).
17. **RTX rendering segfaults on ARC L40S** (VERIFIED, 2026-09-26). With
    `enable_cameras=True` the app loads `isaaclab.python.headless.rendering.kit`
    and crashes in `omni.kit.widget.viewport … __enable_hydra_engine` during
    `SimulationApp` start, before any project code runs (driver 595.x; GLFW has no
    display). Physics-only runs are unaffected. The workaround is the default
    **top-down renderer** (`eval/topdown.py`), which draws frames from simulator state.
18. **Action 0 is not "stop"** when the action ranges are asymmetric. Each dim maps
    [−1, 1] onto its range, so 0 is the **midpoint** (lin [−0.5, 1.5] → 0.5 m/s). Any
    "parked" probe must send the action that maps to zero velocity.
19. **Video, Isaac viewport path (UNVERIFIED; see 17):**
    - AppLauncher `enable_cameras=True`, `render_mode="rgb_array"`, and `env_cfg.viewer = ViewerCfg(origin_type="asset_root", asset_name="robot", env_index=0, …)`.
    - `env.render()` gives frames and `VisualizationMarkers` draws goal spheres.
    - See `eval/record_clips.py` / `arc/record.slurm`. Needs `imageio[ffmpeg]` and an RT-core GPU.

---

## 6. Robot and physics design lessons (VERIFIED)

- **Check that the robot can physically do the task before training.** The JetBot
  (2 kg, 3 cm wheels) reached 0.38 of a 1.5 m/s command and about 1% of the commanded
  turn rate with a 1–5 kg payload, so it was replaced by Carter.
- **Scale the study to the robot.** Carter at 76 kg made a 1–9 kg payload trivial
  (≤ 12% of mass). It was rescaled to 10–45 kg train and 55–90 kg OOD.
- **Put payload mass where a real payload sits.** Adding it to `chassis_link`
  (the `mass_modifier` mode) moved the COM toward the axle of a robot with no
  front support. Model it as a deck link inside the support triangle instead
  (x −0.12, z 0.508 in the root frame).
- **An infinite motor torque (USD default) removes the payload × slope effect.**
  Derive a finite cap from a stated rule:
  - Rule: climb the train extreme (45 kg at 10°) with 1.5× margin, so τ = M g sinθ / 2 · r gives **37.2 N·m**.
  - It is motor-limited at the OOD extremes.
  - Mark cells above the cap **infeasible** and exclude them from degradation classification (`env/payload.py::climb_feasible`, `CellMetrics.feasible`).
- **The terrain must keep the whole robot on the patch.** Obstacle inflation cannot see
  the patch edge. Starts near the edge let Carter's caster overhang it, and the robot
  tipped tail-down (7 of 64 envs, every run). Use an edge band of one robot radius
  (`solvability.edge_mask`) for start/goal sampling, A*, and the Nav2 planner.
- **Obstacle pool:** Isaac Lab clones one prim graph, so obstacles are a fixed pool
  of kinematic prims, re-posed each episode. PhysX cannot rescale a cloned collider
  per env, so **radii are fixed per pool slot** and terrain generation draws from
  slots (`pool_slot_radii`). The sim, A* and lidar then agree.
- **Robot radius for solvability** must be ≥ the footprint: 0.55 m (footprint 0.443).
  `arc/solvability_sweep.py` (login node) measured all densities solvable, with up
  to 43 of 50 attempts at 0.9. The budget was raised to 100.

---

## 7. The verification method: smoke test and probes

`training/train.py --smoke-test` (use `--num-envs 64` for bring-up; the default of
2048 checks scale). Every line maps to a bug this project actually had:

| Probe / line | What it catches |
|---|---|
| `joints` / `bodies` / `actuator[...]` | wrong names; actuating passive joints; torque cap not applied |
| `footprint radius` vs `solvability.robot_radius_m` | A* certifying gaps the robot can't fit |
| `obstacle pose err` | kinematic obstacles not moving (analytic lidar would then "see" ghosts) |
| `friction (PhysX)`, `payload link mass`, `tyre friction` | writes not landing (read back from PhysX) |
| `spawn settle` (tilt, light vs heavy split, gravity-x sign) | tipping at spawn and its cause (payload COM vs edge overhang vs spawn height) |
| `start edge dist` | robot spawned overhanging the patch edge |
| `wheel tracking` | weak motors vs slip |
| `forward probe` / `turn probe` | wrong wheel sign / swap / forward axis; wrong radius; veer |
| `directed drive` (obstacle contacts vs edge exits) | dead collision sensor; goal-frame bugs |
| `friction probe` (launch slip at μ 0.9 vs 0.1) | **inert friction axis** |
| `slope probe` (root height above expected ground, flat vs 10°) | slope sign/axis mismatch |
| first-step terminations | ground contact counted as collision |
| end-of-log `PhysX error` count | silent buffer overflows |

**Probe hygiene:**
- Probes force every env to one condition, so **snapshot all spread checks before probes run.** Forgetting this produced three false FAILs.
- Checks must match what is physically expected. A ">90% of episodes end in collision" check was wrong, because under random actions a collision is the only way to end within 6 s.

**Evaluation sanity triad:** always run (1) a **random** policy (the floor; here 0%),
(2) the **baseline** (Nav2 must succeed reliably in-distribution before its OOD numbers
mean anything), and (3) the trained policy, through the same harness.

---

## 8. Training notes (VERIFIED)

- rsl_rl PPO, 2048 envs, 24 steps/env, 3000 iterations. `--resume auto` plus a fixed
  `--run-name` survives walltime kills (numeric checkpoint order; trains only the
  remaining iterations; same W&B run id).
- **Clipped actions plus `entropy_coef` 0.005 made action noise grow** (std 1.0 → 1.96):
  the policy leaned on clipping. **`entropy_coef` 0.001 → std converged to 0.21**,
  reward ~240, episodes ~6 s. Use 0.001.
- Reading rsl_rl logs: with this reward (+200 goal, −100 collision, progress, step
  cost), a mean reward of about 230–240 means most episodes succeed. Confirm with
  the eval, not the training reward.

---

## 9. OPEN issues (as of 2026-09-25)

- **Slope 5° anomaly (Phase 1 eval, first run):** 0% success for RL, Nav2 *and* random,
  with large timeouts even for random, while 10° and 12° were fine. Not a tilt bug
  (slope probe passes). The per-episode breakdown (did robots move?) is still pending.
  Re-check in the corrected environment (`p1b` evals).
- **Robots thrown at 5° and 15° but not 10°** (smoke test, 2026-09-26): max
  1.57 / 4.71 m moved in 1 s and height spread 0.14 / 0.12 m, versus 0.000 at 0°
  and 10°. This matches the eval failures at those angles. Launch forensics (start,
  yaw vs uphill, height at reset, nearest obstacle) and a repeat-angle test are in
  the smoke test.
- **Friction still unverified:** the p1b eval showed identical drive times from
  μ 0.9 to 0.1. The first slide test was invalid (it sent action 0, see 5.18).
- **Launch slip 0.48 at μ 0.9** is higher than rigid-body physics predicts (traction
  should exceed motor force). Suspects: PhysX convex approximation of cylinder
  wheels; the deprecated patch-friction flag on Carter's materials. The friction axis
  is live (0.48 vs 0.60), but weaker than intended.
- **Nav2 in-process baseline** had about 20% collisions on flat in-distribution runs
  before the wheel-radius fix. Re-check with `nav2_b`. It must be high before any
  RL-vs-Nav2 claim.
- **UNVERIFIED code:** `eval/record_clips.py` (video); `payload.attach_mode:
  rigid_body_with_joint` (not wired, raises); the `raycaster` modality (legacy);
  `sensors.modality: camera` (not wired).

---

## 10. Command cheat sheet (Falcon login node)

```bash
cd ~/ondemand/data/robotics-rl-payload-transport && git pull --ff-only
PY=~/miniconda3/envs/rtn/bin/python
S="sbatch --account=<ACCT> --mail-user=<you>@vt.edu"
U="--set robot.usd_path=/home/<you>/isaac_assets/carter/carter_v1.usd"

$PY -m pytest tests/ -q                                  # pure logic, no GPU
$PY arc/solvability_sweep.py                             # terrain solvability per density
~/usdtools/bin/python arc/inspect_usd.py <robot.usd>     # robot geometry, joints, masses

$S --time=00:45:00 arc/train.slurm --smoke-test --num-envs 64 $U       # bring-up
$S --time=02:00:00 arc/train.slurm --run-name <run> --resume auto $U \
   --set algo.ppo.algorithm.entropy_coef=0.001                          # train
$S arc/eval.slurm --checkpoint results/runs/<run>/model_final.pt --tag <run> $U
$S arc/eval.slurm --policy random --tag random $U
$S arc/eval.slurm --policy nav2   --tag nav2   $U
$S arc/record.slurm --checkpoint results/runs/<run>/model_final.pt --tag <run> $U  # video

squeue -u $USER ; tail -f logs/slurm/rtn-<job>-<id>.out
grep -c "PhysX error" logs/slurm/<log>.out               # must be 0
```

---

## 11. Where things live

| Concern | File |
|---|---|
| Robot cfg factory | `env/robots.py` |
| Env (scene, reset, dones/rewards, collision, friction, payload) | `env/nav_env.py` |
| Terrain + solvability (pure) | `env/terrain_factory.py`, `env/solvability.py` |
| Analytic lidar (pure) | `env/lidar.py` |
| Payload math, climb feasibility, PhysX writes | `env/payload.py` |
| Smoke test + probes | `training/train.py::run_smoke_test` |
| Eval harness / metrics / feasibility exclusion | `eval/ood_harness.py`, `eval/metrics.py`, `eval/run_eval.py` |
| Video | `eval/record_clips.py`, `configs/clips.yaml` |
| ARC launchers and install | `arc/*.slurm`, `arc/arc_env.sh`, `arc/setup_env.sh` |
| Login-node tools | `arc/inspect_usd.py`, `arc/solvability_sweep.py` |
| Robot / sim constants and their derivations | `configs/robot.yaml`, `configs/train.yaml` |
| A100-era verification checklist (older, partly superseded) | `setup_notes.md` |
