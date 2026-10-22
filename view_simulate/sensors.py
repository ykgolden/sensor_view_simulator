"""Camera and lidar models.

The sensor frame uses +X as the viewing axis, +Y to the left and +Z up.
`xyz` / `rpy` are the mount pose relative to the parent link, with URDF rpy order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from view_simulate.mathutil import normalize, xyz_rpy_matrix

PALETTE = (
    (0.18, 0.55, 0.95),
    (0.95, 0.58, 0.12),
    (0.20, 0.72, 0.45),
    (0.72, 0.36, 0.86),
    (0.90, 0.32, 0.38),
)

MAX_RAYS = 25000


@dataclass
class Sensor:
    name: str
    kind: str
    parent: str
    xyz: np.ndarray
    rpy: np.ndarray
    hfov_deg: float = 90.0
    vfov_deg: float = 60.0
    hfov_min_deg: float | None = None
    hfov_max_deg: float | None = None
    vfov_min_deg: float | None = None
    vfov_max_deg: float | None = None
    max_range: float = 5.0
    h_rays: int = 32
    v_rays: int = 16
    color: tuple = (0.18, 0.55, 0.95)

    def __post_init__(self) -> None:
        if self.kind not in ("camera", "lidar"):
            raise ValueError(f"未知传感器类型 {self.kind}")
        self.xyz = np.array(self.xyz, dtype=float)
        self.rpy = np.array(self.rpy, dtype=float)
        self.color = tuple(float(channel) for channel in self.color)
        self.hfov_min_deg, self.hfov_max_deg = _limits_from_span(
            self.hfov_deg, self.hfov_min_deg, self.hfov_max_deg
        )
        self.vfov_min_deg, self.vfov_max_deg = _limits_from_span(
            self.vfov_deg, self.vfov_min_deg, self.vfov_max_deg
        )
        self.hfov_deg = float(self.hfov_max_deg - self.hfov_min_deg)
        self.vfov_deg = float(self.vfov_max_deg - self.vfov_min_deg)

    def fov_limits_deg(self) -> tuple[float, float, float, float]:
        """Horizontal then vertical limits, each ordered low to high, in degrees.

        Horizontal 0 looks along +X and positive angles point toward +Y (left).
        Vertical 0 is level and positive angles point toward +Z (up).
        """
        horizontal = sorted((float(self.hfov_min_deg), float(self.hfov_max_deg)))
        vertical = sorted((float(self.vfov_min_deg), float(self.vfov_max_deg)))
        return horizontal[0], horizontal[1], vertical[0], vertical[1]

    def to_dict(self) -> dict:
        h0, h1, v0, v1 = self.fov_limits_deg()
        return {
            "name": self.name,
            "kind": self.kind,
            "parent": self.parent,
            "xyz": [float(v) for v in self.xyz],
            "rpy_deg": [float(v) for v in np.rad2deg(self.rpy)],
            "hfov_deg": h1 - h0,
            "vfov_deg": v1 - v0,
            "hfov_min_deg": h0,
            "hfov_max_deg": h1,
            "vfov_min_deg": v0,
            "vfov_max_deg": v1,
            "max_range": float(self.max_range),
            "h_rays": int(self.h_rays),
            "v_rays": int(self.v_rays),
            "color": list(self.color),
        }

    @classmethod
    def from_dict(cls, data: dict, fallback_color=PALETTE[0]) -> "Sensor":
        kind = data.get("kind", "camera")
        color = data.get("color", fallback_color)
        if "rpy_deg" in data:
            rpy = np.deg2rad(np.array(data["rpy_deg"], dtype=float))
        else:
            rpy = np.array(data.get("rpy", [0.0, 0.0, 0.0]), dtype=float)
        return cls(
            name=str(data.get("name") or kind),
            kind=kind,
            parent=str(data.get("parent") or ""),
            xyz=np.array(data.get("xyz", [0.0, 0.0, 0.0]), dtype=float),
            rpy=rpy,
            hfov_deg=float(data.get("hfov_deg", 90.0)),
            vfov_deg=float(data.get("vfov_deg", 60.0)),
            hfov_min_deg=_optional_float(data, "hfov_min_deg"),
            hfov_max_deg=_optional_float(data, "hfov_max_deg"),
            vfov_min_deg=_optional_float(data, "vfov_min_deg"),
            vfov_max_deg=_optional_float(data, "vfov_max_deg"),
            max_range=float(data.get("max_range", 5.0)),
            h_rays=int(data.get("h_rays", 32)),
            v_rays=int(data.get("v_rays", 16)),
            color=tuple(color),
        )


def demo_sensors() -> list[Sensor]:
    camera = Sensor(
        name="头部相机",
        kind="camera",
        parent="head_link",
        xyz=[0.16, 0.0, 0.08],
        rpy=[0.0, 0.0, 0.0],
        hfov_deg=70.0,
        vfov_deg=55.0,
        max_range=3.5,
        h_rays=48,
        v_rays=32,
        color=PALETTE[0],
    )
    lidar = Sensor(
        name="顶部雷达",
        kind="lidar",
        parent="head_link",
        xyz=[0.02, 0.0, 0.16],
        rpy=[0.0, 0.0, 0.0],
        hfov_deg=360.0,
        vfov_deg=90.0,
        max_range=3.5,
        h_rays=120,
        v_rays=12,
        color=PALETTE[1],
    )
    return [camera, lidar]


def sensor_pose(robot, sensor: Sensor) -> np.ndarray:
    return robot.link_transform(sensor.parent) @ xyz_rpy_matrix(sensor.xyz, sensor.rpy)


def ray_directions(sensor: Sensor) -> np.ndarray:
    horizontal = max(int(sensor.h_rays), 2)
    vertical = max(int(sensor.v_rays), 1 if sensor.kind == "lidar" else 2)
    if horizontal * vertical > MAX_RAYS:
        raise ValueError(f"射线数量 {horizontal * vertical} 超过 {MAX_RAYS}，请降低采样")
    h0, h1, v0, v1 = _limit_radians(sensor)
    if sensor.kind == "camera":
        ys = np.linspace(np.tan(h1), np.tan(h0), horizontal)
        zs = np.linspace(np.tan(v1), np.tan(v0), vertical)
        yy, zz = np.meshgrid(ys, zs)
        directions = np.stack([np.ones_like(yy), yy, zz], axis=-1).reshape(-1, 3)
        return normalize(directions)
    if _full_circle(h0, h1):
        yaws = np.linspace(-np.pi, np.pi, horizontal, endpoint=False)
    else:
        yaws = np.linspace(h0, h1, horizontal)
    pitches = np.linspace(v0, v1, vertical)
    yaw, pitch = np.meshgrid(yaws, pitches)
    cosine = np.cos(pitch)
    directions = np.stack(
        [cosine * np.cos(yaw), cosine * np.sin(yaw), np.sin(pitch)],
        axis=-1,
    )
    return directions.reshape(-1, 3)


def points_in_fov(local_points: np.ndarray, sensor: Sensor):
    """Return (mask, distance) for points expressed in the sensor frame."""
    local_points = np.asarray(local_points, dtype=float)
    if len(local_points) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    distance = np.linalg.norm(local_points, axis=1)
    x = local_points[:, 0]
    y = local_points[:, 1]
    z = local_points[:, 2]
    h0, h1, v0, v1 = _limit_radians(sensor)
    finite = np.isfinite(distance) & (distance > 1e-5) & (distance <= sensor.max_range + 1e-6)
    if sensor.kind == "camera":
        ahead = x > 1e-4
        mask = (
            finite
            & ahead
            & (y <= x * np.tan(h1) + 1e-8)
            & (y >= x * np.tan(h0) - 1e-8)
            & (z <= x * np.tan(v1) + 1e-8)
            & (z >= x * np.tan(v0) - 1e-8)
        )
    else:
        yaw = np.arctan2(y, x)
        pitch = np.arctan2(z, np.hypot(x, y))
        if _full_circle(h0, h1):
            in_horizontal = np.ones(len(local_points), dtype=bool)
        else:
            in_horizontal = (yaw >= h0 - 1e-6) & (yaw <= h1 + 1e-6)
        in_vertical = (pitch >= v0 - 1e-6) & (pitch <= v1 + 1e-6)
        mask = finite & in_horizontal & in_vertical
    return mask, distance


def lidar_body_mesh() -> trimesh.Trimesh:
    """A short vertical puck at the sensor origin, independent of the scan shell."""
    return trimesh.creation.cylinder(radius=0.06, height=0.05, sections=24)


def frustum_mesh(sensor: Sensor) -> trimesh.Trimesh:
    h0, h1, v0, v1 = _limit_radians(sensor)
    if sensor.kind == "camera":
        return _camera_frustum(h0, h1, v0, v1, sensor.max_range)
    return _lidar_shell(h0, h1, v0, v1, sensor.max_range)


def _camera_frustum(h0: float, h1: float, v0: float, v1: float, distance: float) -> trimesh.Trimesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [distance, distance * np.tan(h1), distance * np.tan(v1)],
            [distance, distance * np.tan(h0), distance * np.tan(v1)],
            [distance, distance * np.tan(h0), distance * np.tan(v0)],
            [distance, distance * np.tan(h1), distance * np.tan(v0)],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [1, 4, 3],
            [1, 3, 2],
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices, faces, process=False)


def _lidar_shell(h0: float, h1: float, v0: float, v1: float, distance: float) -> trimesh.Trimesh:
    full = _full_circle(h0, h1)
    n_h = 48 if full else 36
    n_v = 10
    if full:
        yaws = np.linspace(-np.pi, np.pi, n_h, endpoint=False)
    else:
        yaws = np.linspace(h0, h1, n_h)
    pitches = np.linspace(v0, v1, n_v)
    yaw, pitch = np.meshgrid(yaws, pitches)
    cosine = np.cos(pitch)
    vertices = np.stack(
        [
            distance * cosine * np.cos(yaw),
            distance * cosine * np.sin(yaw),
            distance * np.sin(pitch),
        ],
        axis=-1,
    ).reshape(-1, 3)
    faces = []
    column_count = n_h if full else n_h - 1
    for row in range(n_v - 1):
        for col in range(column_count):
            nxt = (col + 1) % n_h if full else col + 1
            a = row * n_h + col
            b = row * n_h + nxt
            c = (row + 1) * n_h + nxt
            d = (row + 1) * n_h + col
            faces.append((a, b, c))
            faces.append((a, c, d))
    if not full:
        origin = len(vertices)
        vertices = np.vstack([vertices, np.zeros((1, 3))])
        for row in range(n_v - 1):
            a = row * n_h
            d = (row + 1) * n_h
            b = row * n_h + (n_h - 1)
            c = (row + 1) * n_h + (n_h - 1)
            faces.append((origin, d, a))
            faces.append((origin, b, c))
    if not faces:
        vertices = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0], [distance, 0.01, 0.0]])
        faces = [(0, 1, 2)]
    return trimesh.Trimesh(vertices, np.array(faces, dtype=np.int64), process=False)


def _limits_from_span(span: float, low, high) -> tuple[float, float]:
    if low is None and high is None:
        half = float(span) * 0.5
        return -half, half
    if low is None:
        low = -float(span) * 0.5
    if high is None:
        high = float(span) * 0.5
    return float(low), float(high)


def _optional_float(data: dict, key: str) -> float | None:
    if key not in data or data[key] is None:
        return None
    return float(data[key])


def _limit_radians(sensor: Sensor) -> tuple[float, float, float, float]:
    h0, h1, v0, v1 = sensor.fov_limits_deg()
    if sensor.kind == "camera":
        h0, h1 = np.clip(h0, -89.0, 89.0), np.clip(h1, -89.0, 89.0)
        v0, v1 = np.clip(v0, -89.0, 89.0), np.clip(v1, -89.0, 89.0)
    else:
        h0, h1 = np.clip(h0, -180.0, 180.0), np.clip(h1, -180.0, 180.0)
        v0, v1 = np.clip(v0, -90.0, 90.0), np.clip(v1, -90.0, 90.0)
    return np.deg2rad(h0), np.deg2rad(h1), np.deg2rad(v0), np.deg2rad(v1)


def _full_circle(h0: float, h1: float) -> bool:
    return float(h1 - h0) >= np.deg2rad(359.9)
