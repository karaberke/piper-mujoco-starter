"""Pose math: wall coordinates -> plate pose -> gripper pose.

This file is plain NumPy and knows nothing about MuJoCo, so the same code
works unchanged in the Isaac Sim project.

A pose is a 4x4 homogeneous transform T. T_a_b means "frame b expressed in
frame a": its top-left 3x3 block holds b's axes as columns (in a's
coordinates) and its last column holds b's origin. Chaining poses is matrix
multiplication: T_world_c = T_world_b @ T_b_c.

Frames used here:
  world  robot base at the origin, x forward, y left, z up.
  wall   origin at the bottom-left corner of the wall surface as seen from
         the robot. x = u (to the right), y = v (up), z = out of the wall,
         towards the robot.
  plate  origin at the plate centre. x along its width, y along its height,
         z out of its front face. The grab tab is on the back (-z) face.
  tcp    the gripper's tool centre point. z = approach direction (out of the
         gripper), y = the axis the fingers close along.
"""

import numpy as np


def make_pose(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def invert(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    return make_pose(R.T, -R.T @ p)


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat_wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), the order MuJoCo and Isaac use."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def matrix_from_quat_wxyz(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def wall_pose_in_world(surface_x: float, left_y: float) -> np.ndarray:
    """T_world_wall for a wall whose surface faces the robot (normal = -x).

    Seen from the robot, "right" is world -y and "up" is world +z, so
    u = -y, v = +z and the wall normal (towards the robot) = -x.
    """
    R = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    return make_pose(R, [surface_x, left_y, 0.0])


def plate_target_in_wall(u: float, v: float, theta: float, thickness: float, gap: float = 0.0005) -> np.ndarray:
    """T_wall_plate: the plate lying flat on the wall at (u, v), turned by theta.

    The plate's front face (+z) points into the wall, so its z axis is the
    wall's -z, and at theta = 0 the plate is upright (its y axis = wall v).
    rot_y(pi) does that flip (x -> -x, z -> -z, y unchanged), then rot_z(theta)
    turns it counter-clockwise as seen from the robot.
    """
    R = rot_z(theta) @ rot_y(np.pi)
    return make_pose(R, [u, v, thickness / 2 + gap])


def tcp_in_plate(thickness: float, back_face_gap: float) -> np.ndarray:
    """T_plate_tcp: where the gripper's TCP sits relative to a grasped plate.

    The gripper comes in from behind (approach = plate +z, through the back
    face towards the front), pinching the tab across the plate's y axis:
      tcp z = plate z, tcp y = plate y, tcp x = plate x.
    The TCP is back_face_gap behind the back face.
    """
    return make_pose(np.eye(3), [0.0, 0.0, -(thickness / 2 + back_face_gap)])


def plate_in_holder(radius: float, angle: float, bottom_z: float, height: float) -> np.ndarray:
    """T_world_plate for a plate standing upright, front face pointing away from the base.

    The plate's centre is radius from the base at the given angle; its bottom
    edge is at bottom_z. Facing straight ahead (angle 0) the plate axes are
    x = world y, y = world z (up), z = world x (front face away from the robot).
    """
    R0 = np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    p = [radius * np.cos(angle), radius * np.sin(angle), bottom_z + height / 2]
    return make_pose(rot_z(angle) @ R0, p)


def offset_along_own_z(T: np.ndarray, distance: float) -> np.ndarray:
    """Move a pose along its own z axis (e.g. back off along the approach axis)."""
    out = T.copy()
    out[:3, 3] = T[:3, 3] + T[:3, 2] * distance
    return out


def pose_error(T_actual: np.ndarray, T_target: np.ndarray) -> tuple[float, float]:
    """(position error in metres, rotation error in radians) between two poses."""
    dp = np.linalg.norm(T_actual[:3, 3] - T_target[:3, 3])
    R_err = T_target[:3, :3].T @ T_actual[:3, :3]
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    return float(dp), float(angle)
