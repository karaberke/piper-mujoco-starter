"""Step 2: drive the joints directly, one at a time, then open and close the gripper.

Shows the core control idea: write a target angle into data.ctrl and the
position actuator moves the joint there. No IK yet.
"""

import argparse

import numpy as np

from piper_starter.robot import GRIPPER_CLOSED, GRIPPER_OPEN, PiperController
from piper_starter.runtime import add_common_args, session
from piper_starter.scene import build_model

parser = argparse.ArgumentParser(description=__doc__)
add_common_args(parser)
args = parser.parse_args()

model, data = build_model()
with session(model, data, args) as on_step:
    robot = PiperController(model, data, on_step=on_step)
    home = robot.arm_q()
    for j in range(6):
        q = home.copy()
        q[j] += 0.4 if j != 2 else -0.4  # joint3 only bends one way (range -2.7..0)
        print(f"joint{j + 1}: {home[j]:+.2f} -> {q[j]:+.2f} rad")
        robot.move_joints(q, 1.0)
        robot.move_joints(home, 1.0)
    print("gripper open / close")
    robot.gripper(GRIPPER_OPEN, 1.0)
    robot.gripper(GRIPPER_CLOSED, 1.0)
    print("final joint angles:", np.round(robot.arm_q(), 3))
