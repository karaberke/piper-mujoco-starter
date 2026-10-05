"""A small controller for the simulated Piper.

The Menagerie model drives every joint with a *position actuator*: you
write the angle you want into data.ctrl and a built-in PD controller
pushes the joint there. Everything here boils down to choosing a sequence
of those targets and stepping the physics.
"""

from typing import Callable

import mujoco
import numpy as np

from .config import SceneConfig
from .ik import ARM_JOINTS, PiperIK
from .poses import make_pose, matrix_from_quat_wxyz, offset_along_own_z, quat_wxyz_from_matrix

GRIPPER_OPEN = 0.035  # each finger travels 35 mm, so the jaws open 70 mm
GRIPPER_CLOSED = 0.0
CONTROL_EVERY = 10  # physics steps per control update (0.002 s x 10 = 50 Hz)


class PiperController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cfg: SceneConfig | None = None,
                 on_step: Callable[[], None] | None = None):
        self.model, self.data = model, data
        self.cfg = cfg or SceneConfig()
        self.ik = PiperIK(self.cfg)
        self.on_step = on_step or (lambda: None)  # viewer refresh / video capture
        self.qadr = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.arm_act = [model.actuator(j).id for j in ARM_JOINTS]
        self.grip_act = model.actuator("gripper").id
        self.tcp_site = model.site("tcp").id
        self.q_cmd = self.arm_q()  # the joint angles we *want* (what IK produced)
        # Extra offset added on top of q_cmd. Under load the joints settle a
        # little short of their targets (gravity on the plate, joint friction),
        # so settle() learns how far short and aims past by that much. It's
        # the same idea as the integral term in a PID controller.
        self.q_bias = np.zeros(len(ARM_JOINTS))

    # --- reading state ---------------------------------------------------
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def tcp_pose(self) -> np.ndarray:
        return make_pose(self.data.site_xmat[self.tcp_site].reshape(3, 3), self.data.site_xpos[self.tcp_site])

    def body_pose(self, name: str) -> np.ndarray:
        b = self.data.body(name)
        return make_pose(matrix_from_quat_wxyz(b.xquat), b.xpos)

    # --- low level ---------------------------------------------------------
    def step(self, n: int = 1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self.on_step()

    def command(self, q_arm: np.ndarray):
        self.q_cmd = np.asarray(q_arm, float).copy()
        self.data.ctrl[self.arm_act] = self.q_cmd + self.q_bias

    def settle(self, rounds: int = 4, seconds: float = 0.25):
        """Wait for the arm to stop, then correct for how far short each joint stopped."""
        self.wait(seconds)
        for _ in range(rounds):
            self.q_bias += self.q_cmd - self.arm_q()
            self.command(self.q_cmd)
            self.wait(seconds)

    def reset_bias(self):
        """Forget the learned offset (call when the load changes, e.g. after releasing)."""
        self.q_bias[:] = 0
        self.command(self.q_cmd)

    def wait(self, seconds: float):
        self.step(int(seconds / self.model.opt.timestep))

    def gripper(self, opening: float, seconds: float = 0.6):
        self.data.ctrl[self.grip_act] = opening
        self.wait(seconds)

    # --- motions -------------------------------------------------------------
    def move_joints(self, q_goal: np.ndarray, duration: float = 1.5):
        """Joint-space move: every joint goes from its current target to its goal.

        Uses a smooth S-curve (smoothstep) so the arm starts and stops gently.
        """
        q_start = self.q_cmd.copy()
        ticks = max(1, int(duration / (self.model.opt.timestep * CONTROL_EVERY)))
        for k in range(1, ticks + 1):
            s = k / ticks
            s = s * s * (3 - 2 * s)
            self.command(q_start + s * (q_goal - q_start))
            self.step(CONTROL_EVERY)
        self.settle()

    def move_to_pose(self, T_goal: np.ndarray, duration: float = 2.0) -> bool:
        """Solve IK once for the goal, then do a joint-space move.

        Fast and robust, but the gripper's path in between is a curve, not a
        straight line, so only use it in open space.
        """
        q_goal, ok = self.ik.solve(T_goal, self.q_cmd)
        if not ok:
            print("  ! IK found no solution for that pose; skipping the move")
            return False
        self.move_joints(q_goal, duration)
        return True

    def move_linear(self, T_goal: np.ndarray, duration: float = 1.5) -> bool:
        """Move the TCP along a straight line (and rotate smoothly) to T_goal.

        Each control tick asks IK for a pose a little further along the line,
        starting from the previous answer. Use this near objects: approaching
        a grasp, lifting, pressing a plate onto the wall.
        """
        T_start = self.ik.tcp_pose(self.q_cmd)
        q_start, q_goal = quat_wxyz_from_matrix(T_start[:3, :3]), quat_wxyz_from_matrix(T_goal[:3, :3])
        if np.dot(q_start, q_goal) < 0:  # take the short way round
            q_goal = -q_goal
        ticks = max(1, int(duration / (self.model.opt.timestep * CONTROL_EVERY)))
        ok_all = True
        for k in range(1, ticks + 1):
            s = k / ticks
            s = s * s * (3 - 2 * s)
            q = (1 - s) * q_start + s * q_goal  # normalised lerp, close enough to slerp here
            T = make_pose(matrix_from_quat_wxyz(q / np.linalg.norm(q)), (1 - s) * T_start[:3, 3] + s * T_goal[:3, 3])
            q_arm, ok = self.ik.solve_local(T, self.q_cmd, max_iters=50)
            ok_all &= ok
            self.command(q_arm)
            self.step(CONTROL_EVERY)
        self.settle()
        return ok_all

    def back_off(self, distance: float, duration: float = 1.0):
        """Retreat straight back along the gripper's approach axis."""
        self.move_linear(offset_along_own_z(self.ik.tcp_pose(self.q_cmd), -distance), duration)
