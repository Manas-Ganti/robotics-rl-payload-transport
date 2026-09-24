"""Robot ArticulationCfg factory referenced by ``configs/robot.yaml: robot.cfg_import_path``.

ISAAC-ONLY: imported lazily by ``env/nav_env.py::resolve_robot_cfg`` after
AppLauncher has started Isaac Sim. Never import this from pure-logic code.

Isaac Lab 2.x ships no ArticulationCfg for NVIDIA's mobile bases (Carter,
JetBot) in ``isaaclab_assets`` -- its docs define them inline in tutorials.
This module does the same, generically: every value comes from robot.yaml, so
switching robots is a config edit. The robot is still NVIDIA's built-in USD,
not a custom model (CLAUDE.md: "Use an Isaac Lab built-in mobile robot").

Pattern source: isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/add_new_robot.html
"""

from __future__ import annotations

from typing import Any, Mapping

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def make_diff_drive_cfg(robot: Mapping[str, Any]) -> ArticulationCfg:
    """Build a differential-drive ArticulationCfg from the ``robot`` YAML block.

    * USD: ``robot.usd_path`` (local copy) or ``ISAAC_NUCLEUS_DIR/robot.nucleus_asset``.
      Isaac Sim 4.5 content layout is ``Robots/<Name>/...`` -- the current
      Isaac Lab docs show ``Robots/NVIDIA/...``, which is the 5.x layout and
      404s on the 4.5 server (verified on ARC for the JetBot).
    * Only the TWO WHEEL joints are actuated. Casters (swivel + roll joints on
      Carter) must stay passive -- actuating ".*" would drive them toward a
      target and fight the robot's own motion.
    * Velocity control: stiffness 0, damping = velocity gain. ``null`` in YAML
      keeps the USD's own value for that field.
    * Torque cap: ``effort_limit_sim`` -- for implicit actuators, the limit
      PhysX actually enforces (plain ``effort_limit`` is not the sim cap).
    * ``activate_contact_sensors=True``: ContactSensor (our collision ground
      truth) only works on prims spawned with contact reporting enabled.
    """
    usd_path = robot.get("usd_path") or f"{ISAAC_NUCLEUS_DIR}/{robot['nucleus_asset']}"
    act = robot["actuator"]
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path), activate_contact_sensors=True),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[robot["left_wheel_joint"], robot["right_wheel_joint"]],
                stiffness=act["stiffness"],
                damping=act["damping"],
                effort_limit_sim=act["effort_limit_sim"],
            )
        },
    )
