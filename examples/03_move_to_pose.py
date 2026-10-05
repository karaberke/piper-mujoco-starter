"""Step 3: inverse kinematics. Give a gripper pose, the arm goes there.

    uv run mjpython examples/03_move_to_pose.py --xyz 0.35 0.1 0.3 --rpy 0 90 0

--xyz is the TCP position in metres (world frame: x forward, y left, z up).
--rpy is roll, pitch, yaw in degrees (rotations about x, then y, then z of
the world). The gripper's approach axis is the TCP's z axis, so rpy 0 90 0
points the gripper straight forward, rpy 0 180 0 straight down.
"""

import argparse

import numpy as np

from piper_starter.poses import make_pose, pose_error, rot_x, rot_y, rot_z
from piper_starter.robot import PiperController
from piper_starter.runtime import add_common_args, session
from piper_starter.scene import build_model

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--xyz", type=float, nargs=3, default=[0.35, 0.0, 0.30])
parser.add_argument("--rpy", type=float, nargs=3, default=[0.0, 90.0, 0.0])
add_common_args(parser)
args = parser.parse_args()

r, p, y = np.radians(args.rpy)
T_goal = make_pose(rot_z(y) @ rot_y(p) @ rot_x(r), args.xyz)

model, data = build_model()
with session(model, data, args) as on_step:
    robot = PiperController(model, data, on_step=on_step)
    if robot.move_to_pose(T_goal, 2.0):
        dp, da = pose_error(robot.tcp_pose(), T_goal)
        print(f"reached: position error {dp * 1000:.1f} mm, orientation error {np.degrees(da):.2f} deg")
        print("joint angles (rad):", np.round(robot.arm_q(), 3))
