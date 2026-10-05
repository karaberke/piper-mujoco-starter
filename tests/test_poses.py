"""Checks for the pose math. Run with: uv run pytest"""

import numpy as np
import pytest

from piper_starter.poses import (
    invert,
    make_pose,
    matrix_from_quat_wxyz,
    plate_target_in_wall,
    quat_wxyz_from_matrix,
    rot_x,
    rot_z,
    tcp_in_plate,
    wall_pose_in_world,
)
from piper_starter.task import plate_in_wall_coords


def is_rotation(R):
    return np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)


def test_wall_frame_axes():
    T = wall_pose_in_world(0.55, 0.5)
    assert is_rotation(T[:3, :3])
    assert np.allclose(T[:3, 0], [0, -1, 0])  # u: right as seen from the robot
    assert np.allclose(T[:3, 1], [0, 0, 1])  # v: up
    assert np.allclose(T[:3, 2], [-1, 0, 0])  # normal: towards the robot


def test_plate_faces_the_wall_and_is_upright():
    T = plate_target_in_wall(0.3, 0.2, 0.0, 0.002)
    assert is_rotation(T[:3, :3])
    assert np.allclose(T[:3, 2], [0, 0, -1])  # front face into the wall
    assert np.allclose(T[:3, 1], [0, 1, 0])  # plate "up" = wall "up"
    assert T[2, 3] > 0  # centre in front of the surface


@pytest.mark.parametrize("theta", [-90, -30, 0, 15, 45, 170])
def test_wall_coords_round_trip(theta):
    T = plate_target_in_wall(0.42, 0.31, np.radians(theta), 0.002)
    u, v, th = plate_in_wall_coords(T)
    assert np.allclose([u, v, np.degrees(th)], [0.42, 0.31, theta])


def test_quaternion_round_trip():
    R = rot_z(0.7) @ rot_x(-1.2)
    assert np.allclose(matrix_from_quat_wxyz(quat_wxyz_from_matrix(R)), R)


def test_invert():
    T = make_pose(rot_z(0.3) @ rot_x(0.5), [0.1, -0.2, 0.3])
    assert np.allclose(T @ invert(T), np.eye(4))


def test_tcp_sits_behind_the_back_face():
    T = tcp_in_plate(0.002, 0.02)
    assert np.isclose(T[2, 3], -0.021)  # 1 mm half-thickness + 20 mm gap
