"""Nav2 classical-planner baseline, driven through the SHARED eval harness.

CLAUDE.md principle 5: "Honest baseline. Nav2 runs through the *same* eval
harness as the policy." This module implements ``eval.ood_harness.Policy`` and
nothing else -- it cannot see the grid, choose its own episodes, or reach a
different metric implementation. Same terrain seeds, same solvability guarantee,
same success/collision/path-efficiency definitions.

WHAT NAV2 GETS THAT THE RL POLICY DOES NOT
------------------------------------------
Be explicit about this in the write-up. Nav2 receives a full occupancy costmap
of the environment, while the policy sees only a 64-ray forward scan. That
asymmetry FAVOURS Nav2 and is intentional: it makes Nav2 a strong baseline
rather than a straw man. A learned policy that matches a fully-informed
classical planner OOD is a much stronger result than one that beats a
deliberately blinded planner.

Conversely, Nav2 knows nothing about payload mass or friction -- it plans
geometrically. The expected headline is that Nav2 holds up on the GEOMETRY axis
(obstacle_density) and degrades on the DYNAMICS axes (payload, friction, slope),
where the RL policy has at least the opportunity to have learned compensation.
That contrast is the study.

STATUS: every ROS 2 / Nav2 hook below is marked ``# VERIFY ON A100:`` and is
unrunnable on the local dev machine. Two integration paths are provided; pick
one on the A100 and delete the other.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from env.solvability import Cell, astar, grid_to_world, inflate_occupancy, radius_to_cells


# ---------------------------------------------------------------------------
# Integration mode
# ---------------------------------------------------------------------------
MODE_ROS2_BRIDGE = "ros2_bridge"
MODE_INPROC_PLANNER = "inproc_planner"


@dataclass
class Nav2Config:
    """Nav2 planner/controller parameters, loaded from baselines/nav2_params.yaml."""

    mode: str
    lookahead_distance_m: float
    max_lin_vel: float
    max_ang_vel: float
    goal_tolerance_m: float
    robot_radius_m: float
    replan_period_steps: int
    heading_gain: float
    slow_down_angle_rad: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Nav2Config":
        import yaml

        with Path(path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        nav2 = data["nav2"]
        return cls(
            mode=str(nav2["mode"]),
            lookahead_distance_m=float(nav2["controller"]["lookahead_distance_m"]),
            max_lin_vel=float(nav2["controller"]["max_lin_vel"]),
            max_ang_vel=float(nav2["controller"]["max_ang_vel"]),
            goal_tolerance_m=float(nav2["controller"]["goal_tolerance_m"]),
            robot_radius_m=float(nav2["costmap"]["robot_radius_m"]),
            replan_period_steps=int(nav2["planner"]["replan_period_steps"]),
            heading_gain=float(nav2["controller"]["heading_gain"]),
            slow_down_angle_rad=float(nav2["controller"]["slow_down_angle_rad"]),
        )


class Nav2Policy:
    """Classical planner baseline implementing the harness ``Policy`` interface.

    Two modes, selected by ``nav2.mode`` in ``baselines/nav2_params.yaml``:

    ``inproc_planner`` (default, recommended first)
        Runs Nav2's ALGORITHMS in-process: A* global planning on the same
        occupancy grid, plus a pure-pursuit controller mirroring Nav2's
        Regulated Pure Pursuit. No ROS 2 required, fully batched across
        thousands of envs, and it reuses this repo's own A* -- so the baseline's
        global planner is literally the same code that certified the episode
        solvable. Fast, reproducible, and directly comparable.

    ``ros2_bridge``
        Drives the REAL Nav2 stack over ROS 2. Higher fidelity (real costmap
        inflation, recovery behaviours, DWB) but requires a live ROS 2 bridge to
        Isaac Sim and realistically runs one environment at a time -- hours per
        OOD grid instead of minutes.

    RECOMMENDATION: run ``inproc_planner`` for the full grid, then spot-check a
    few cells with ``ros2_bridge`` to confirm the in-process controller is not
    flattering the baseline. Report which produced the headline numbers.
    """

    def __init__(self, env: Any, eval_cfg: Any):
        import torch

        self.name = "nav2"
        self._torch = torch
        self._env = env

        data = eval_cfg.to_dict() if hasattr(eval_cfg, "to_dict") else dict(eval_cfg)
        params_path = data.get("baseline", {}).get("config", "baselines/nav2_params.yaml")
        self.cfg = Nav2Config.from_yaml(params_path)

        self.num_envs = int(env.num_envs)
        self.device = env.device

        # Per-env plan state
        self._paths: List[Optional[List[Tuple[float, float]]]] = [None] * self.num_envs
        self._waypoint_idx: List[int] = [0] * self.num_envs
        self._steps_since_replan: List[int] = [0] * self.num_envs

        if self.cfg.mode == MODE_ROS2_BRIDGE:
            self._init_ros2()

        print(f"Nav2 baseline initialized in '{self.cfg.mode}' mode")

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------
    def act(self, observations: Any) -> Any:
        """Return actions in ``[-1, 1]`` -- the same space the RL policy outputs.

        Nav2 does NOT consume ``observations``: it plans from the ground-truth
        occupancy grid and robot pose (its structural advantage, documented in
        the module docstring). The argument is accepted to satisfy the shared
        Policy interface, which is what keeps the harness policy-agnostic.
        """
        if self.cfg.mode == MODE_ROS2_BRIDGE:
            return self._act_ros2()
        return self._act_inproc()

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        """Discard cached plans so the next episode replans from scratch."""
        ids = range(self.num_envs) if env_ids is None else env_ids
        for env_id in ids:
            self._paths[env_id] = None
            self._waypoint_idx[env_id] = 0
            self._steps_since_replan[env_id] = 0

    # ------------------------------------------------------------------
    # In-process planner
    # ------------------------------------------------------------------
    def _act_inproc(self) -> Any:
        """Global A* plan + pure-pursuit control, batched across envs."""
        torch = self._torch
        env = self._env

        robot_xy = (env.robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]).cpu().numpy()
        yaws = env._robot_yaw().cpu().numpy()

        actions = np.zeros((self.num_envs, 2), dtype=np.float32)

        for env_id in range(self.num_envs):
            spec = env._terrain_specs[env_id]
            if spec is None:
                continue

            if self._needs_replan(env_id):
                self._plan(env_id, spec, robot_xy[env_id])

            actions[env_id] = self._pure_pursuit(env_id, robot_xy[env_id], float(yaws[env_id]))
            self._steps_since_replan[env_id] += 1

        return torch.tensor(actions, dtype=torch.float32, device=self.device)

    def _needs_replan(self, env_id: int) -> bool:
        return (
            self._paths[env_id] is None
            or self._steps_since_replan[env_id] >= self.cfg.replan_period_steps
        )

    def _plan(self, env_id: int, spec: Any, robot_xy: np.ndarray) -> None:
        """Global plan: A* over the inflated occupancy grid.

        Uses this repo's own A* (``env/solvability.py``) -- the same planner that
        certified the episode solvable. Nav2's global planner (NavFn/Smac) is
        also grid A*/Dijkstra, so this is a faithful stand-in for the planning
        stage while eliminating any chance of the baseline failing an episode
        that the harness guaranteed was achievable.
        """
        from env.solvability import edge_mask, world_to_grid

        radius_cells = radius_to_cells(self.cfg.robot_radius_m, spec.resolution_m)
        planning_grid = inflate_occupancy(spec.occupancy, radius_cells)
        # Nav2's costmap is bounded by the map: keep the footprint on the patch,
        # exactly as the solvability check does, or the baseline would plan
        # along the edge and "fail" episodes the harness certified achievable.
        planning_grid |= edge_mask(planning_grid.shape[0], spec.resolution_m, self.cfg.robot_radius_m)

        start = world_to_grid(
            robot_xy, terrain_size_m=spec.size_m, resolution_m=spec.resolution_m
        )
        goal = spec.goal_cell

        # If inflation buries the robot's CURRENT cell (it has drifted next to an
        # obstacle mid-episode), fall back to the uninflated grid. Nav2's real
        # behaviour here is a recovery routine; refusing to plan would score as a
        # timeout and overstate the baseline's fragility for a purely
        # implementation reason.
        if planning_grid[start]:
            planning_grid = np.asarray(spec.occupancy, dtype=bool)

        path_cells = astar(planning_grid, start, goal, connectivity=8)

        if path_cells is None:
            self._paths[env_id] = None
            self._waypoint_idx[env_id] = 0
        else:
            self._paths[env_id] = [
                grid_to_world(cell, terrain_size_m=spec.size_m, resolution_m=spec.resolution_m)
                for cell in path_cells
            ]
            self._waypoint_idx[env_id] = 0

        self._steps_since_replan[env_id] = 0

    def _pure_pursuit(self, env_id: int, robot_xy: np.ndarray, yaw: float) -> np.ndarray:
        """Regulated pure-pursuit controller, mirroring Nav2's default.

        Returns NORMALIZED actions in ``[-1, 1]`` -- the same space the RL policy
        emits, so the environment applies identical scaling to both. Feeding
        physical velocities here instead would give the baseline a different
        effective action range and silently break the comparison.

        "Regulated" = slow down when the heading error is large, exactly as Nav2
        does, so the robot turns toward the path before accelerating along it.
        """
        path = self._paths[env_id]
        if not path:
            return np.zeros(2, dtype=np.float32)

        target = self._advance_to_lookahead(env_id, robot_xy, path)

        dx, dy = target[0] - robot_xy[0], target[1] - robot_xy[1]
        heading_error = math.atan2(dy, dx) - yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))  # wrap to [-pi, pi]

        ang_vel = float(np.clip(self.cfg.heading_gain * heading_error, -1.0, 1.0))

        # Regulate forward speed by heading error: full speed when aligned,
        # near zero when the target is off to the side.
        alignment = max(0.0, 1.0 - abs(heading_error) / self.cfg.slow_down_angle_rad)
        lin_vel = float(np.clip(alignment, 0.0, 1.0))

        return np.array([lin_vel, ang_vel], dtype=np.float32)

    def _advance_to_lookahead(
        self, env_id: int, robot_xy: np.ndarray, path: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Advance the waypoint cursor to the first point beyond the lookahead.

        The cursor only moves FORWARD, so the robot cannot latch onto an earlier
        waypoint after overshooting and end up oscillating along the path.
        """
        idx = self._waypoint_idx[env_id]
        while idx < len(path) - 1:
            distance = math.dist(robot_xy, path[idx])
            if distance >= self.cfg.lookahead_distance_m:
                break
            idx += 1
        self._waypoint_idx[env_id] = idx
        return path[idx]

    # ------------------------------------------------------------------
    # ROS 2 bridge mode
    # ------------------------------------------------------------------
    def _init_ros2(self) -> None:
        """Initialize the ROS 2 node and Nav2 action clients.

        VERIFY ON A100 -- this whole path is unverifiable locally:
          1. ROS 2 (Humble or Iron) sourced in the same shell as Isaac Sim, and
             `rclpy` importable from Isaac's bundled Python. Isaac ships its own
             interpreter; a system-installed rclpy will NOT import.
          2. The Isaac Sim ROS 2 bridge extension enabled
             (omni.isaac.ros2_bridge) with odom/scan/tf publishing.
          3. Nav2 brought up with a matching map frame and this robot's footprint.
          4. NavigateToPose action server reachable.

        See setup_notes.md for the full bring-up sequence. Expect this to be the
        single most time-consuming item on the A100 -- budget accordingly, and
        note that `inproc_planner` mode produces publishable results without it.
        """
        warnings.warn(
            "Nav2 ROS 2 bridge mode is wired but UNVERIFIED. It requires a live "
            "ROS 2 + Nav2 stack bridged to Isaac Sim and runs roughly one env at "
            "a time. Prefer mode='inproc_planner' for the full OOD grid; see the "
            "Nav2Policy docstring.",
            stacklevel=2,
        )

        try:
            import rclpy  # noqa: PLC0415
            from nav2_msgs.action import NavigateToPose  # noqa: PLC0415
            from rclpy.action import ActionClient  # noqa: PLC0415
            from rclpy.node import Node  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "ROS 2 packages unavailable. Either source the ROS 2 setup before "
                "launching Isaac Sim, or set nav2.mode='inproc_planner' in "
                "baselines/nav2_params.yaml."
            ) from exc

        if not rclpy.ok():
            rclpy.init()

        self._ros_node = Node("nav2_baseline_runner")
        self._nav_client = ActionClient(self._ros_node, NavigateToPose, "navigate_to_pose")

        # VERIFY ON A100: server name and readiness timeout.
        if not self._nav_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError(
                "Nav2 'navigate_to_pose' action server not found after 30 s. "
                "Confirm Nav2 is running and the lifecycle nodes are ACTIVE "
                "(ros2 lifecycle get /bt_navigator)."
            )

    def _act_ros2(self) -> Any:
        """Pull the latest cmd_vel from Nav2 and convert it to normalized actions.

        VERIFY ON A100: the full round trip.
          * Isaac publishes /scan + /odom + /tf at a rate Nav2 accepts
          * goals are sent as NavigateToPose in the correct frame
          * cmd_vel is subscribed and converted to [-1, 1] using the SAME
            action ranges as the RL policy (configs/train.yaml: action.*),
            otherwise the baseline is effectively driving a different robot
        """
        raise NotImplementedError(
            "ROS 2 bridge stepping is not implemented -- it depends on the "
            "specific Isaac Sim <-> ROS 2 bridge topics available on the A100. "
            "Implement _act_ros2 there, or use mode='inproc_planner'. "
            "See setup_notes.md for the bring-up checklist."
        )
