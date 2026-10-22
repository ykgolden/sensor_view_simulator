"""Draw a robot, sensor frustums and visibility samples in a PyVista plotter."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

LABEL_RGB = np.array(
    [
        [168, 174, 184],
        [230, 70, 70],
        [46, 204, 122],
    ],
    dtype=np.uint8,
)


@dataclass
class Frame:
    robot_mesh: object = None
    frustums: list = field(default_factory=list)
    axes: list = field(default_factory=list)
    coverage_points: np.ndarray | None = None
    coverage_colors: np.ndarray | None = None
    hit_points: np.ndarray | None = None
    hit_distance: np.ndarray | None = None
    reset_camera: bool = False
    grid_extent: float = 10.0
    walls: tuple | None = None
    voxel_centers: np.ndarray | None = None
    voxel_labels: np.ndarray | None = None
    voxel_size: float = 0.0
    voxel_opacity: float = 0.35


def prepare_plotter(plotter) -> None:
    plotter.set_background("#1e2228")
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    plotter.add_axes(line_width=2)


def render_frame(plotter, frame: Frame) -> None:
    _clear_dynamic(plotter)
    if frame.robot_mesh is not None and len(frame.robot_mesh.faces):
        poly = trimesh_to_polydata(frame.robot_mesh, use_face_color=True)
        plotter.add_mesh(
            poly,
            scalars="RGBA",
            rgb=True,
            preference="cell",
            smooth_shading=False,
            name="dyn_robot",
        )
    for index, (mesh, color, opacity, edges) in enumerate(frame.frustums):
        poly = trimesh_to_polydata(mesh, use_face_color=False)
        plotter.add_mesh(
            poly,
            color=color,
            opacity=float(opacity),
            lighting=False,
            show_edges=bool(edges),
            edge_color=color,
            line_width=2,
            name=f"dyn_frustum_{index}",
        )
    for index, (pose, length) in enumerate(frame.axes):
        _add_axes(plotter, pose, length, f"dyn_axis_{index}")
    if frame.coverage_points is not None and len(frame.coverage_points):
        cloud = pv.PolyData(np.ascontiguousarray(frame.coverage_points, dtype=np.float64))
        cloud["RGB"] = np.ascontiguousarray(frame.coverage_colors, dtype=np.uint8)
        plotter.add_mesh(
            cloud,
            scalars="RGB",
            rgb=True,
            point_size=12,
            render_points_as_spheres=True,
            name="dyn_coverage",
        )
    if frame.voxel_centers is not None and frame.voxel_labels is not None and frame.voxel_size > 0:
        _add_occupancy(plotter, frame.voxel_centers, frame.voxel_labels, frame.voxel_size, frame.voxel_opacity)
    if frame.hit_points is not None and len(frame.hit_points):
        cloud = pv.PolyData(np.ascontiguousarray(frame.hit_points, dtype=np.float64))
        cloud["m"] = np.ascontiguousarray(frame.hit_distance, dtype=np.float64)
        plotter.add_mesh(
            cloud,
            scalars="m",
            cmap="viridis",
            point_size=8,
            render_points_as_spheres=True,
            name="dyn_hits",
            scalar_bar_args={"title": "m"},
        )
    if frame.walls is not None:
        _add_walls(plotter, frame.walls)
    _show_reference_grid(plotter, frame.robot_mesh, frame.grid_extent)
    if frame.reset_camera:
        frame_camera(plotter, frame.robot_mesh, frame.grid_extent)
    plotter.render()
    if hasattr(plotter, "update"):
        plotter.update()


def _add_occupancy(plotter, centers, labels, size: float, opacity: float) -> None:
    centers = np.asarray(centers, dtype=float)
    labels = np.asarray(labels)
    edge = float(size) * 0.92
    cube = pv.Cube(x_length=edge, y_length=edge, z_length=edge)
    opacity = float(np.clip(opacity, 0.0, 1.0))
    for name, label, color in (
        ("dyn_voxel_visible", 2, "#2ecc71"),
        ("dyn_voxel_blocked", 1, "#e64646"),
    ):
        chosen = centers[labels == label]
        if len(chosen) == 0:
            continue
        cloud = pv.PolyData(np.ascontiguousarray(chosen, dtype=np.float64))
        glyphs = cloud.glyph(geom=cube, scale=False, orient=False)
        plotter.add_mesh(glyphs, color=color, opacity=opacity, lighting=False, name=name)


def _add_walls(plotter, walls) -> None:
    half, z0, height = walls
    z1 = float(z0) + float(height)
    half = float(half)
    quads = (
        ((half, -half, z0), (half, half, z0), (half, half, z1), (half, -half, z1)),
        ((-half, half, z0), (-half, -half, z0), (-half, -half, z1), (-half, half, z1)),
        ((-half, half, z0), (half, half, z0), (half, half, z1), (-half, half, z1)),
        ((half, -half, z0), (-half, -half, z0), (-half, -half, z1), (half, -half, z1)),
    )
    for index, corners in enumerate(quads):
        points = np.asarray(corners, dtype=float)
        poly = pv.PolyData(points, np.array([4, 0, 1, 2, 3], dtype=np.int64))
        plotter.add_mesh(
            poly,
            color="#9aa3b2",
            opacity=0.14,
            lighting=False,
            culling=False,
            name=f"dyn_wall_{index}",
        )


def _show_reference_grid(plotter, robot_mesh, extent: float) -> None:
    """Draw X/Y ticks from -extent to +extent metres, still covering the robot."""
    try:
        plotter.remove_bounds_axes()
    except Exception:
        pass
    reach = max(float(extent), 0.5)
    low = np.array([-reach, -reach, -0.05])
    high = np.array([reach, reach, 2.0])
    if robot_mesh is not None:
        robot_low, robot_high = np.asarray(robot_mesh.bounds, dtype=float)
        low = np.minimum(low, robot_low)
        high = np.maximum(high, robot_high)
    bounds = (
        float(low[0]),
        float(high[0]),
        float(low[1]),
        float(high[1]),
        float(min(low[2], -0.05)),
        float(high[2] + 0.3),
    )
    plotter.show_grid(bounds=bounds, color="#8b95a3", xtitle="X", ytitle="Y", ztitle="Z")


def frame_camera(plotter, robot_mesh, fallback_extent: float = 10.0) -> None:
    if robot_mesh is not None and len(robot_mesh.vertices):
        low, high = np.asarray(robot_mesh.bounds, dtype=float)
    else:
        reach = max(float(fallback_extent), 1.0)
        low = np.array([-reach, -reach, 0.0])
        high = np.array([reach, reach, 2.0])
    center = (low + high) / 2.0
    extent = max(float(np.max(high - low)), 0.5)
    position = center + np.array([0.45, -0.95, 0.55]) * extent
    plotter.camera_position = [position.tolist(), center.tolist(), [0.0, 0.0, 1.0]]
    plotter.reset_camera_clipping_range()


def coverage_colors(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    if len(labels) == 0:
        return colors
    known = (labels >= 0) & (labels < len(LABEL_RGB))
    colors[known] = LABEL_RGB[labels[known]]
    return colors


def trimesh_to_polydata(mesh, use_face_color: bool):
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return pv.PolyData(vertices)
    packed = np.hstack((np.full((len(faces), 1), 3, dtype=np.int64), faces)).ravel()
    poly = pv.PolyData(vertices, packed)
    if use_face_color:
        colors = np.asarray(mesh.visual.face_colors)
        if colors.ndim != 2 or len(colors) != len(faces):
            colors = np.tile(np.array([180, 185, 190, 255], dtype=np.uint8), (len(faces), 1))
        poly.cell_data["RGBA"] = np.ascontiguousarray(colors[:, :4], dtype=np.uint8)
    return poly


def _add_axes(plotter, pose: np.ndarray, length: float, prefix: str) -> None:
    origin = pose[:3, 3]
    colors = ("#e74c3c", "#2ecc71", "#3498db")
    for axis, color in enumerate(colors):
        end = origin + pose[:3, axis] * float(length)
        plotter.add_mesh(
            pv.Line(np.asarray(origin, dtype=float), np.asarray(end, dtype=float)),
            color=color,
            line_width=5 if axis == 0 else 2,
            name=f"{prefix}_{axis}",
        )


def _clear_dynamic(plotter) -> None:
    names = [name for name in list(plotter.actors) if str(name).startswith("dyn_")]
    for name in names:
        plotter.remove_actor(name)
    try:
        plotter.remove_scalar_bar()
    except Exception:
        pass
