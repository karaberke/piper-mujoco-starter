"""Step 4: which wall positions can the arm actually reach?

Prints a map of the wall: '#' = IK finds joint angles for both the
placement pose and the pre-place pose 8 cm out, '.' = it doesn't. Run it
before picking targets for step 5. No viewer; takes a few seconds.

    uv run python examples/04_reach_map.py --theta 30
"""

import argparse

import numpy as np

from piper_starter.config import SceneConfig
from piper_starter.ik import PiperIK
from piper_starter.poses import make_pose, plate_target_in_wall, rot_z, tcp_in_plate, wall_pose_in_world

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--theta", type=float, default=0.0, help="plate rotation on the wall, degrees")
parser.add_argument("--step", type=float, default=0.05, help="grid spacing in metres")
args = parser.parse_args()

cfg = SceneConfig()
ik = PiperIK(cfg)
plate, wall = cfg.plates[0], cfg.wall
T_world_wall = wall_pose_in_world(wall.surface_x, wall.left_y)
home = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0])


def reachable(u: float, v: float) -> bool:
    for flip in (0.0, np.pi):  # the tab can be held either way up
        T_pt = tcp_in_plate(plate.thickness, cfg.grasp_back_face_gap) @ make_pose(rot_z(flip), [0, 0, 0])
        T = T_world_wall @ plate_target_in_wall(u, v, np.radians(args.theta), plate.thickness) @ T_pt
        T_pre = T.copy()
        T_pre[:3, 3] += T_world_wall[:3, 2] * 0.08
        if ik.solve(T, home, restarts=15)[1] and ik.solve(T_pre, home, restarts=15)[1]:
            return True
    return False


us = np.arange(0.15, wall.width - 0.15 + 1e-9, args.step)
vs = np.arange(0.10, wall.height - 0.10 + 1e-9, args.step)
print(f"theta = {args.theta} deg    u = {us[0]:.2f} .. {us[-1]:.2f} m left to right")
for v in vs[::-1]:
    print(f"v={v:.2f} " + "".join("#" if reachable(u, v) else "." for u in us))
