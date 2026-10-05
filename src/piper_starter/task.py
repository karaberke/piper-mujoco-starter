"""The milestone task: pick plate <label>, place it at (u, v, theta) on the wall.

The sequence is a simple state machine written as straight-line code:

  1. find the plate      read its pose (stand-in for perception)
  2. pre-grasp           hover 8 cm behind the tab
  3. grasp               slide in, close the fingers
  4. lift + retreat      raise it out of the slot, pull back
  5. pre-place           hold the plate 8 cm in front of its target
     and correct        measure the plate, nudge until it lines up
  6. place               slide straight in until the plate touches the wall
  7. attach + release    switch on the magnet, open, back off
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import SceneConfig
from .poses import (
    invert,
    make_pose,
    rot_z,
    offset_along_own_z,
    plate_target_in_wall,
    pose_error,
    quat_wxyz_from_matrix,
    tcp_in_plate,
    wall_pose_in_world,
)
from .robot import GRIPPER_CLOSED, GRIPPER_OPEN, PiperController

HOME = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0])


@dataclass
class PlaceResult:
    label: str
    target_uv_theta: tuple
    achieved_uv_theta: tuple
    position_error_mm: float
    angle_error_deg: float
    success: bool


def plate_in_wall_coords(T_wall_plate: np.ndarray) -> tuple[float, float, float]:
    """Read (u, v, theta) back out of a plate pose in the wall frame."""
    u, v = T_wall_plate[0, 3], T_wall_plate[1, 3]
    # Plate +y is "up" at theta = 0; measure how far it has turned in the wall plane.
    y_axis = T_wall_plate[:3, 1]
    theta = np.arctan2(-y_axis[0], y_axis[1])
    return float(u), float(v), float(theta)


def show_target(model: mujoco.MjModel, data: mujoco.MjData, T: np.ndarray):
    mocap = model.body("target_ghost").mocapid[0]
    data.mocap_pos[mocap] = T[:3, 3]
    data.mocap_quat[mocap] = quat_wxyz_from_matrix(T[:3, :3])


def switch_magnet(model: mujoco.MjModel, data: mujoco.MjData, label: str, wall_body: str, on: bool):
    """Weld the plate to the wall where it currently is (or release it)."""
    eq = model.equality(f"magnet_{label}").id
    if on:
        b1, b2 = data.body(wall_body), data.body(f"plate_{label}")
        R1 = b1.xmat.reshape(3, 3)
        # A weld stores body2's pose relative to body1 and holds it there.
        rel_pos = R1.T @ (b2.xpos - b1.xpos)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, _quat_conj(b1.xquat), b2.xquat)
        model.eq_data[eq, 0:3] = 0  # anchor at body2's origin
        model.eq_data[eq, 3:6] = rel_pos
        model.eq_data[eq, 6:10] = rel_quat
        model.eq_data[eq, 10] = 1  # torque scale
    data.eq_active[eq] = int(on)


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def plan_grasp(robot: PiperController, T_world_plate: np.ndarray, T_target: np.ndarray,
               T_world_wall: np.ndarray, T_plate_tcp: np.ndarray, standoff: float):
    """Pick which way up to hold the tab, checking every key pose is reachable.

    The tab can be pinched with the gripper either way up (turned 180 degrees
    about the approach axis). Once grasped, that choice is locked in for the
    place, and with the Piper's limited wrist sometimes only one of the two
    works at the wall. So check both *before* moving: a tiny planner.
    Returns T_plate_tcp for the chosen grasp, or None if neither works.
    """
    ik = robot.ik
    for flip in (0.0, np.pi):
        T_pt = T_plate_tcp @ make_pose(rot_z(flip), [0, 0, 0])
        T_grasp = T_world_plate @ T_pt
        T_target_pre = T_target.copy()
        T_target_pre[:3, 3] += T_world_wall[:3, 2] * standoff
        q = robot.q_cmd
        ok_all = True
        for T in (offset_along_own_z(T_grasp, -0.08), T_grasp, T_target_pre @ T_pt, T_target @ T_pt):
            q, ok = ik.solve(T, q)
            ok_all &= ok
            if not ok:
                break
        if ok_all:
            return T_pt
    return None


def pick_and_place(robot: PiperController, label: str, u: float, v: float, theta_deg: float,
                   tol_mm: float = 5.0, tol_deg: float = 2.0) -> PlaceResult:
    cfg: SceneConfig = robot.cfg
    model, data = robot.model, robot.data
    plate = cfg.plate(label)
    wall_name = f"wall_{cfg.wall.name}"
    T_plate_tcp = tcp_in_plate(plate.thickness, cfg.grasp_back_face_gap)

    # 1. Find the plate. In simulation we can just ask where it is; on a real
    #    robot this would come from a camera (e.g. a tag or a detector).
    print(f"[1] locating plate {label}")
    robot.wait(0.5)  # let the plates settle into their slots before looking
    T_world_plate = robot.body_pose(f"plate_{label}")

    # Where should the plate go? wall frame -> world frame.
    T_world_wall = wall_pose_in_world(cfg.wall.surface_x, cfg.wall.left_y)
    T_target = T_world_wall @ plate_target_in_wall(u, v, np.radians(theta_deg), plate.thickness)
    show_target(model, data, T_target)
    standoff = 0.08

    T_plate_tcp = plan_grasp(robot, T_world_plate, T_target, T_world_wall, T_plate_tcp, standoff)
    if T_plate_tcp is None:
        raise RuntimeError(f"Target (u={u}, v={v}, theta={theta_deg}) is out of the arm's reach")
    T_grasp = T_world_plate @ T_plate_tcp

    print("[2] pre-grasp")
    robot.gripper(GRIPPER_OPEN, 0.3)
    robot.move_to_pose(offset_along_own_z(T_grasp, -0.08), 2.0)

    print("[3] grasp")
    robot.move_linear(T_grasp, 1.2)
    robot.gripper(GRIPPER_CLOSED, 0.8)

    print("[4] lift and retreat")
    T_lift = T_grasp.copy()
    T_lift[2, 3] += 0.06
    robot.move_linear(T_lift, 1.0)
    robot.back_off(0.10, 1.0)

    # How is the plate actually sitting in the hand? Use the measured offset
    # rather than the nominal one, so a slightly crooked grasp still lands
    # the plate in the right place.
    T_tcp_plate = invert(robot.tcp_pose()) @ robot.body_pose(f"plate_{label}")

    T_place = T_target @ invert(T_tcp_plate)

    # 5. Pre-place: hold the plate 8 cm out from its target, then measure
    #    where the plate really is and nudge until it lines up. Doing this
    #    in free space (not against the wall) lets the plate move freely.
    print("[5] pre-place")
    T_target_pre = T_target.copy()
    T_target_pre[:3, 3] += T_world_wall[:3, 2] * standoff  # wall z points out of the wall
    T_cmd_pre = T_target_pre @ invert(T_tcp_plate)
    if not robot.move_to_pose(T_cmd_pre, 2.5):
        raise RuntimeError(f"Target (u={u}, v={v}, theta={theta_deg}) is out of the arm's reach")
    T_cmd = T_cmd_pre
    for i in range(3):
        T_actual = robot.body_pose(f"plate_{label}")
        dp, da = pose_error(T_actual, T_target_pre)
        print(f"    check {i}: plate is {dp * 1000:.1f} mm / {np.degrees(da):.2f} deg off")
        if dp * 1000 < tol_mm / 2 and np.degrees(da) < tol_deg / 2:
            break
        T_cmd = (T_target_pre @ invert(T_actual)) @ T_cmd  # shift the command by the error
        robot.move_linear(T_cmd, 0.5)
    # The correction, expressed in the gripper's own frame, carries over to
    # the final pose because that's just 8 cm further along the same line.
    correction = invert(T_cmd_pre) @ T_cmd

    print("[6] place")
    robot.move_linear(T_place @ correction, 1.2)

    T_actual = robot.body_pose(f"plate_{label}")
    dp, da = pose_error(T_actual, T_target)

    print("[7] attach and release")
    switch_magnet(model, data, label, wall_name, True)
    robot.gripper(GRIPPER_OPEN, 0.5)
    robot.reset_bias()
    robot.back_off(0.08, 1.0)
    robot.move_joints(HOME, 2.0)

    achieved = plate_in_wall_coords(invert(T_world_wall) @ robot.body_pose(f"plate_{label}"))
    result = PlaceResult(
        label=label,
        target_uv_theta=(u, v, theta_deg),
        achieved_uv_theta=(achieved[0], achieved[1], float(np.degrees(achieved[2]))),
        position_error_mm=dp * 1000,
        angle_error_deg=float(np.degrees(da)),
        success=dp * 1000 <= tol_mm and np.degrees(da) <= tol_deg,
    )
    return result
