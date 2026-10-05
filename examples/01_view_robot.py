"""Step 1: open the scene and look around. Nothing moves except the plates settling.

    uv run mjpython examples/01_view_robot.py      (macOS)
    uv run python  examples/01_view_robot.py       (Linux / Windows)

In the viewer: left-drag rotates, right-drag pans, scroll zooms, double-click
selects a body, Ctrl + right-drag on a selected body pushes it. The panel on
the left (Tab) shows joints and lets you toggle collision shapes (group 3).
"""

import argparse

import mujoco

from piper_starter.runtime import add_common_args, session
from piper_starter.scene import build_model

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser)
args = parser.parse_args()

model, data = build_model()
print(f"bodies={model.nbody} joints={model.njnt} actuators={model.nu} qpos size={model.nq}")
for i in range(model.nu):
    print(f"  actuator {i}: {model.actuator(i).name:8s} ctrl range {model.actuator_ctrlrange[i]}")

with session(model, data, args) as on_step:
    for _ in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        on_step()
