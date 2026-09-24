"""Robot ArticulationCfgs referenced by ``configs/robot.yaml: robot.cfg_import_path``.

ISAAC-ONLY: imported lazily by ``env/nav_env.py::resolve_robot_cfg`` after
AppLauncher has started Isaac Sim. Never import this from pure-logic code.

Isaac Lab 2.x does NOT ship a JetBot config in ``isaaclab_assets`` -- the
official docs define one inline in the "Add a new robot" tutorial and the
technical-env-design walkthrough. This module is that documented definition,
using NVIDIA's own JetBot USD asset: it is Isaac's built-in robot, not a custom
model (CLAUDE.md: "Use an Isaac Lab built-in mobile robot").

Source pattern: isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/add_new_robot.html
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Wheel joints are "left_wheel_joint" / "right_wheel_joint" (per the Isaac Lab
# walkthrough's dof_names), matching configs/robot.yaml.
#
# stiffness=None / damping=None keeps the USD's own drive gains, exactly as the
# official tutorial does, where the JetBot is driven with
# set_joint_velocity_target -- the same call env/nav_env.py::_apply_action makes.
# VERIFY ON ARC: the smoke test's "mean displacement" is > 0. If the robot does
# not move, the USD drives are not velocity drives: set stiffness=0.0 and a
# positive damping (e.g. 10.0) here to make the implicit actuator a velocity
# controller.
#
# The USD is fetched from Isaac's Nucleus/S3 content server at spawn time.
# VERIFY ON ARC: compute nodes can reach it. If not, download the file once on
# the login node and set `robot.usd_path` in configs/robot.yaml to the local copy.
#
# activate_contact_sensors=True: Isaac Lab's ContactSensor (our collision
# ground truth -> the collision termination and metric) only works on prims
# spawned with contact reporting enabled. The tutorial omits it because it has
# no contact sensor; we need it.
JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Isaac Sim 4.5 layout (verified on ARC: HTTP 200). The current Isaac Lab
        # docs show Robots/NVIDIA/Jetbot/ -- that is the Isaac Sim 5.x layout
        # and does not exist on the 4.5 content server.
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd",
        activate_contact_sensors=True,
    ),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)
    },
)
