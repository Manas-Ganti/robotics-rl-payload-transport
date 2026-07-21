# A100 Setup & Verification Notes

Everything that could not be verified on the code-writing machine, collected as
an ordered checklist. Work through **Part 2 before launching any long run** —
several items fail *silently*, and a silent failure here produces a complete,
plausible-looking set of results for an experiment that never actually varied
the thing it claims to study.

---

## Part 1 — Environment setup

### 1.1 Option A: Docker (recommended)

```bash
docker login nvcr.io                      # NGC credentials required
docker build -t robust-transport-nav .

docker run --name transport-nav --entrypoint bash -it --runtime=nvidia --gpus all \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
  -v $(pwd)/results:/workspace/robust-transport-nav/results:rw \
  robust-transport-nav
```

The build runs `pytest tests/` and **fails the image** if the pure-logic tests
fail — a broken train/OOD split cannot reach the A100 inside a green image.

### 1.2 Option B: native install

```bash
# 1. Isaac Sim (Omniverse Launcher or NGC), then:
export ISAACSIM_PATH=~/.local/share/ov/pkg/isaac-sim-4.2.0

# 2. Isaac Lab, pinned to a tag
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab
git checkout v1.4.0
ln -s ${ISAACSIM_PATH} _isaac_sim
./isaaclab.sh --install rsl_rl

# 3. This project's deps INTO ISAAC SIM'S PYTHON (not the system python)
cd /path/to/robotics-rl-payload-transport
${ISAACSIM_PATH}/python.sh -m pip install -r requirements.txt
```

> **Do not `pip install torch`.** Isaac Sim ships its own build. Installing a
> second one into that interpreter reliably breaks it.

### 1.3 First checks

```bash
# Pure-logic tests — no GPU, no Isaac. Run these FIRST.
${ISAACSIM_PATH}/python.sh -m pytest tests/ -v

# W&B
wandb login
```

---

## Part 2 — The verification checklist

Ordered by *how badly a silent failure corrupts the study*, not by module.

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

- [ ] **Obstacle scale matches the sampled radii.** The occupancy grid (and
      therefore the solvability guarantee) uses per-obstacle sampled radii, but
      the prim pool spawns one fixed radius. Verify per-instance scale writes.
      → If wrong: the sim disagrees with the A* check that certified the episode
      solvable, producing collisions the harness swore were impossible.

- [ ] **The raycaster hits the terrain** — `mesh_prim_paths` in `build_env_cfg`
      must include `/World/envs/env_.*/Ground` **and** the obstacle pool path.
      → If wrong: depth returns uniformly max-range; the policy is blind but
      trains anyway, badly, and it looks like a learning problem.

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

## Part 3 — Phase run order

```bash
# Phase 0 — bring-up. Read every FAIL: line before continuing.
python training/train.py --config configs/train.yaml --headless --smoke-test

# Phase 1 — flat floor, obstacles, friction, payload, reward v1
python training/train.py --config configs/train.yaml --headless

# Phase 2 — slope axis + first full OOD sweep
python training/train.py --config configs/train.yaml --headless \
    --set phases.enable_slope=true
python eval/run_eval.py --checkpoint results/runs/<run>/model_final.pt --headless

# Phase 3 — Nav2 baseline through the SAME harness
python eval/run_eval.py --policy nav2 --headless

# Phase 4 — sensor degradation + transport-aware reward v2
python training/train.py --config configs/train.yaml --headless \
    --set phases.enable_sensor_noise=true --set phases.enable_reward_v2=true

# Phase 5 — inferred payload (told-vs-inferred is exactly this one flag)
python training/train.py --config configs/train.yaml --headless \
    --set phases.enable_inferred_payload=true

# Phase 6 — figures (runs locally, no GPU)
python analysis/plots.py --results-dir results/eval --out-dir results/figures
```

**Sanity floor, worth one run:** `python eval/run_eval.py --policy random` tells
you what success rate the *task* yields with no competence at all. If the
trained policy sits near that floor the problem is training, not generalization;
if the random policy scores well, the task is too easy for the OOD study to say
anything.

---

## Part 4 — What to bring back

Copy back and the whole analysis reruns locally with no GPU:

```
results/eval/episodes_*.csv     # per-episode records — any metric recomputable
results/eval/cells_*.csv        # per-grid-point aggregates — what plots.py reads
results/eval/summary_*.json     # degradation profiles + classifications
results/runs/*/resolved_config.json   # exact config that produced the run
```

Bring back **failures too** — a stack trace, a flat curve, an implausible
success rate. The `# VERIFY ON A100:` list above is where to look first, and a
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
