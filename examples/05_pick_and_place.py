"""Step 5: the milestone. Pick plate <label> and put it at (u, v, theta) on the wall.

    uv run mjpython examples/05_pick_and_place.py --plate B --u 0.5 --v 0.3 --theta 15
    uv run python  examples/05_pick_and_place.py --plate A --u 0.35 --v 0.25 --video outputs/demo.mp4

u, v are metres from the wall's bottom-left corner (as seen from the robot),
matching the grid painted on the wall (labels are in cm). theta turns the
plate counter-clockwise, in degrees. The green ghost shows the target.
"""

import argparse

from piper_starter.robot import PiperController
from piper_starter.runtime import add_common_args, session
from piper_starter.scene import build_model
from piper_starter.task import pick_and_place

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--plate", default="B", help="plate label: A or B")
parser.add_argument("--u", type=float, default=0.5)
parser.add_argument("--v", type=float, default=0.3)
parser.add_argument("--theta", type=float, default=0.0)
add_common_args(parser)
args = parser.parse_args()

model, data = build_model()
with session(model, data, args) as on_step:
    robot = PiperController(model, data, on_step=on_step)
    result = pick_and_place(robot, args.plate, args.u, args.v, args.theta)
    au, av, at = result.achieved_uv_theta
    print()
    print(f"target   u={args.u:.3f} v={args.v:.3f} theta={args.theta:.1f}")
    print(f"achieved u={au:.3f} v={av:.3f} theta={at:.1f}")
    print(f"error    {result.position_error_mm:.1f} mm, {result.angle_error_deg:.2f} deg  ->  "
          f"{'SUCCESS' if result.success else 'MISSED'}")
