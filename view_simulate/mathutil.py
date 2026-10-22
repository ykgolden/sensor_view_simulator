"""Small transform helpers. Rotations follow the URDF rpy convention."""

import numpy as np


def normalize(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    length = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / np.maximum(length, 1e-12)


def parse_vec(text, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if text is None or str(text).strip() == "":
        return np.array(default, dtype=float)
    values = [float(part) for part in str(text).replace(",", " ").split()]
    if len(values) != 3:
        raise ValueError(f"需要 3 个数，得到：{text}")
    return np.array(values, dtype=float)


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis rpy: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def xyz_rpy_matrix(xyz, rpy) -> np.ndarray:
    transform = np.eye(4)
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    transform[:3, :3] = rpy_matrix(float(roll), float(pitch), float(yaw))
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def rotation_about_axis(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    length = np.linalg.norm(axis)
    if length < 1e-12:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = axis / length
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    one_c = 1.0 - c
    rotation = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform
