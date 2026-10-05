"""Inverse kinematics: "put the TCP at this pose" -> six joint angles.

Uses mink, which solves IK as a small optimisation problem: each iteration
finds the joint velocity that best reduces the pose error while respecting
joint limits, then integrates it. Repeating that converges on a solution.

IK runs on a robot-only copy of the model, so the plates' free joints and
the gripper fingers never enter the problem.
"""

import mink
import mujoco
import numpy as np

from .config import PIPER_DIR, SceneConfig
from .poses import quat_wxyz_from_matrix

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]


class PiperIK:
    def __init__(self, cfg: SceneConfig | None = None):
        cfg = cfg or SceneConfig()
        spec = mujoco.MjSpec.from_file(str(PIPER_DIR / "piper.xml"))
        spec.body("link6").add_site(name="tcp", pos=cfg.tcp_offset)
        self.model = spec.compile()
        self.config = mink.Configuration(self.model)
        self.tcp_task = mink.FrameTask("tcp", "site", position_cost=1.0, orientation_cost=1.0, lm_damping=1e-3)
        # A weak pull towards the current pose keeps solutions close to where
        # the arm already is (no sudden elbow flips).
        self.posture_task = mink.PostureTask(self.model, cost=1e-3)
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.qadr = [self.model.joint(j).qposadr[0] for j in ARM_JOINTS]

    def solve(self, T_target: np.ndarray, q_arm_init: np.ndarray, restarts: int = 40) -> tuple[np.ndarray, bool]:
        """Return (six joint angles, found?) that put the TCP at T_target.

        Tries from q_arm_init first. The Piper's joint limits create dead
        ends where the optimiser gets stuck, so if that fails it retries
        from random starting poses and keeps the answer closest to
        q_arm_init.
        """
        q, ok = self.solve_local(T_target, q_arm_init)
        if ok:
            return q, True
        rng = np.random.default_rng(0)  # fixed seed: same answer every run
        lo, hi = self.model.jnt_range[: len(ARM_JOINTS)].T
        best = None
        for _ in range(restarts):
            q, ok = self.solve_local(T_target, rng.uniform(lo, hi))
            if ok and (best is None or np.linalg.norm(q - q_arm_init) < np.linalg.norm(best - q_arm_init)):
                best = q
        if best is None:
            return q_arm_init.copy(), False
        return best, True

    def solve_local(self, T_target: np.ndarray, q_arm_init: np.ndarray, max_iters: int = 200,
              pos_tol: float = 5e-4, rot_tol: float = 2e-3) -> tuple[np.ndarray, bool]:
        """Return (six joint angles, converged?) that put the TCP at T_target."""
        q = self.model.qpos0.copy()
        q[self.qadr] = q_arm_init
        self.config.update(q)
        self.posture_task.set_target(q)
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3(quat_wxyz_from_matrix(T_target[:3, :3])), T_target[:3, 3]
        )
        self.tcp_task.set_target(target)
        dt = 0.01
        converged = False
        for _ in range(max_iters):
            vel = mink.solve_ik(self.config, [self.tcp_task, self.posture_task], dt, solver="daqp",
                                damping=1e-3, limits=self.limits)
            self.config.integrate_inplace(vel, dt)
            err = self.tcp_task.compute_error(self.config)
            if np.linalg.norm(err[:3]) < pos_tol and np.linalg.norm(err[3:]) < rot_tol:
                converged = True
                break
        return self.config.q[self.qadr].copy(), converged

    def tcp_pose(self, q_arm: np.ndarray) -> np.ndarray:
        """Forward kinematics: where the TCP is for given joint angles."""
        q = self.model.qpos0.copy()
        q[self.qadr] = q_arm
        self.config.update(q)
        return self.config.get_transform_frame_to_world("tcp", "site").as_matrix()
