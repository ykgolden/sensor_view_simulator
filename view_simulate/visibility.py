"""Ray-cast visibility. Green is seen, red is inside the field of view but blocked."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from view_simulate.gpu_rays import mesh_distances as gpu_mesh_distances
from view_simulate.mathutil import normalize
from view_simulate.sensors import points_in_fov, ray_directions, sensor_pose

GROUND_Z = 0.0
DISPLAY_OFFSET = 0.012


@dataclass
class VisibilityResult:
    hits: np.ndarray
    hit_distance: np.ndarray
    coverage_points: np.ndarray
    coverage_normals: np.ndarray
    coverage_labels: np.ndarray
    n_rays: int
    voxel_centers: np.ndarray | None = None
    voxel_labels: np.ndarray | None = None
    voxel_size: float = 0.0


class SceneQuery:
    """Closest intersection against the robot mesh and a horizontal ground plane."""

    def __init__(
        self,
        mesh: trimesh.Trimesh | None,
        *,
        ground: bool,
        clearance: float,
        ground_z: float = GROUND_Z,
        walls=None,
    ):
        self.ground = ground
        self.ground_z = float(ground_z)
        self.clearance = float(clearance)
        self._triangles = None
        self._v0 = None
        self._edge1 = None
        self._edge2 = None
        self._gpu_v0 = None
        self._gpu_edge1 = None
        self._gpu_edge2 = None
        self._gpu_failed = False
        self._wall_half = None
        self._wall_z0 = 0.0
        self._wall_z1 = 0.0
        if walls is not None:
            half, z0, height = walls
            self._wall_half = max(float(half), 0.0)
            self._wall_z0 = float(z0)
            self._wall_z1 = float(z0) + max(float(height), 0.0)
        if mesh is not None and len(mesh.faces) > 0:
            triangles = np.ascontiguousarray(mesh.triangles, dtype=np.float64)
            self._triangles = triangles
            self._v0 = np.ascontiguousarray(triangles[:, 0])
            self._edge1 = np.ascontiguousarray(triangles[:, 1] - self._v0)
            self._edge2 = np.ascontiguousarray(triangles[:, 2] - self._v0)
            self._gpu_v0 = np.ascontiguousarray(self._v0, dtype=np.float32)
            self._gpu_edge1 = np.ascontiguousarray(self._edge1, dtype=np.float32)
            self._gpu_edge2 = np.ascontiguousarray(self._edge2, dtype=np.float32)

    def distances(self, origins: np.ndarray, directions: np.ndarray, max_range: float) -> np.ndarray:
        mesh_t = self._mesh_distances(origins, directions)
        ground_t = self._ground_distances(origins, directions)
        wall_t = self._wall_distances(origins, directions)
        distance = np.minimum(np.minimum(mesh_t, ground_t), wall_t)
        distance[distance > max_range] = np.inf
        return distance

    def label_points(self, points: np.ndarray, pose: np.ndarray, sensor) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        labels = np.zeros(len(points), dtype=np.int8)
        if len(points) == 0:
            return labels
        rotation = pose[:3, :3]
        origin = pose[:3, 3]
        local = (points - origin) @ rotation
        mask, distance = points_in_fov(local, sensor)
        index = np.flatnonzero(mask)
        if len(index) == 0:
            return labels
        relative = points[index] - origin
        dirs = relative / np.maximum(distance[index, None], 1e-12)
        origins = np.repeat(origin.reshape(1, 3), len(index), axis=0)
        hit_t = self.distances(origins, dirs, float(sensor.max_range))
        tolerance = np.maximum(0.02, distance[index] * 0.025)
        visible = ~(hit_t < distance[index] - tolerance)
        labels[index] = np.where(visible, 2, 1).astype(np.int8)
        return labels

    def cast_rays(self, origin, directions, max_range: float):
        directions = normalize(np.asarray(directions, dtype=float))
        if len(directions) == 0:
            return np.zeros((0, 3)), np.zeros(0)
        origins = np.repeat(np.asarray(origin, dtype=float).reshape(1, 3), len(directions), axis=0)
        distance = self.distances(origins, directions, float(max_range))
        ok = np.isfinite(distance)
        points = origins[ok] + directions[ok] * distance[ok, None]
        return points, distance[ok]

    def _mesh_distances(self, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        count = len(origins)
        distances = np.full(count, np.inf)
        if self._triangles is None or count == 0:
            return distances
        origins = np.ascontiguousarray(origins, dtype=np.float64)
        directions = np.ascontiguousarray(directions, dtype=np.float64)
        if not self._gpu_failed:
            gpu = gpu_mesh_distances(
                origins,
                directions,
                self._gpu_v0,
                self._gpu_edge1,
                self._gpu_edge2,
                self.clearance,
            )
            if gpu is not None:
                return gpu
            self._gpu_failed = True
        face_count = len(self._v0)
        v0 = self._v0
        edge1 = self._edge1
        edge2 = self._edge2
        # Bound the broadcast so a dense mesh does not allocate a huge temporary.
        chunk = max(1, min(count, 400_000 // max(face_count, 1)))
        for start in range(0, count, chunk):
            end = min(start + chunk, count)
            distances[start:end] = _ray_triangle_distance(
                origins[start:end],
                directions[start:end],
                v0,
                edge1,
                edge2,
                self.clearance,
            )
        return distances

    def _ground_distances(self, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        count = len(origins)
        distances = np.full(count, np.inf)
        if not self.ground or count == 0:
            return distances
        dz = directions[:, 2]
        valid = np.abs(dz) > 1e-9
        t_hit = np.full(count, np.inf)
        t_hit[valid] = (self.ground_z - origins[valid, 2]) / dz[valid]
        ok = valid & np.isfinite(t_hit) & (t_hit >= self.clearance)
        distances[ok] = t_hit[ok]
        return distances

    def _wall_distances(self, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        count = len(origins)
        distances = np.full(count, np.inf)
        if self._wall_half is None or count == 0 or self._wall_z1 <= self._wall_z0:
            return distances
        half = self._wall_half
        for axis, sign in ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)):
            denom = directions[:, axis]
            valid = np.abs(denom) > 1e-9
            t_hit = np.full(count, np.inf)
            plane = sign * half
            t_hit[valid] = (plane - origins[valid, axis]) / denom[valid]
            other = 1 - axis
            point_other = np.zeros(count)
            point_z = np.zeros(count)
            point_other[valid] = origins[valid, other] + directions[valid, other] * t_hit[valid]
            point_z[valid] = origins[valid, 2] + directions[valid, 2] * t_hit[valid]
            ok = (
                valid
                & np.isfinite(t_hit)
                & (t_hit >= self.clearance)
                & (np.abs(point_other) <= half + 1e-6)
                & (point_z >= self._wall_z0 - 1e-6)
                & (point_z <= self._wall_z1 + 1e-6)
                & (t_hit < distances)
            )
            distances[ok] = t_hit[ok]
        return distances


def evaluate_sensor(
    robot,
    sensor,
    *,
    ground: bool = True,
    ground_z: float = GROUND_Z,
    clearance: float = 0.01,
    surface_spacing: float = 0.05,
    ground_spacing: float = 0.10,
    walls=None,
    slice_height: float = 0.0,
    cell_size: float = 0.5,
    volume_half: float | None = None,
    want_hits: bool = True,
    want_coverage: bool = True,
) -> VisibilityResult:
    return evaluate_sensors(
        robot,
        [sensor],
        ground=ground,
        ground_z=ground_z,
        clearance=clearance,
        surface_spacing=surface_spacing,
        ground_spacing=ground_spacing,
        walls=walls,
        slice_height=slice_height,
        cell_size=cell_size,
        volume_half=volume_half,
        want_hits=want_hits,
        want_coverage=want_coverage,
    )[0]


def evaluate_sensors(
    robot,
    sensors,
    *,
    ground: bool = True,
    ground_z: float = GROUND_Z,
    clearance: float = 0.01,
    surface_spacing: float = 0.05,
    ground_spacing: float = 0.10,
    walls=None,
    slice_height: float = 0.0,
    cell_size: float = 0.5,
    volume_half: float | None = None,
    want_hits: bool = True,
    want_coverage: bool = True,
) -> list[VisibilityResult]:
    """Occlusion for every sensor against one shared robot mesh."""
    sensors = list(sensors)
    query = SceneQuery(
        robot.world_mesh(),
        ground=ground,
        clearance=clearance,
        ground_z=ground_z,
        walls=walls,
    )
    sampled = _sample_robot(robot, surface_spacing, cap=10000) if want_coverage and sensors else None
    occupancy = float(slice_height) > 1e-6 and float(cell_size) > 1e-6
    half = float(walls[0]) if walls is not None else volume_half
    results = [
        _evaluate_one(
            robot,
            query,
            sampled,
            sensor,
            ground=ground and not occupancy,
            ground_z=ground_z,
            ground_spacing=ground_spacing,
            walls=walls,
            want_hits=want_hits,
            want_coverage=want_coverage,
            layer=index,
        )
        for index, sensor in enumerate(sensors)
    ]
    centers, used = occupancy_centers(half, ground_z, slice_height, cell_size)
    if want_coverage and len(centers) and sensors:
        merged = np.zeros(len(centers), dtype=np.int8)
        for sensor in sensors:
            pose = sensor_pose(robot, sensor)
            labels = query.label_points(centers, pose, sensor)
            merged = np.maximum(merged, labels)
        keep = merged > 0
        centers = centers[keep]
        merged = merged[keep]
    else:
        centers = np.zeros((0, 3))
        merged = np.zeros(0, dtype=np.int8)
        used = 0.0
    for result in results:
        result.voxel_centers = centers
        result.voxel_labels = merged
        result.voxel_size = float(used)
    return results


def _evaluate_one(
    robot,
    query: SceneQuery,
    sampled,
    sensor,
    *,
    ground: bool,
    ground_z: float,
    ground_spacing: float,
    walls,
    want_hits: bool,
    want_coverage: bool,
    layer: int,
) -> VisibilityResult:
    pose = sensor_pose(robot, sensor)
    hits = np.zeros((0, 3))
    hit_distance = np.zeros(0)
    n_rays = 0
    if want_hits:
        directions = ray_directions(sensor)
        n_rays = len(directions)
        world_dirs = directions @ pose[:3, :3].T
        hits, hit_distance = query.cast_rays(pose[:3, 3], world_dirs, sensor.max_range)

    coverage_points = np.zeros((0, 3))
    coverage_normals = np.zeros((0, 3))
    coverage_labels = np.zeros(0, dtype=np.int8)
    if want_coverage:
        origin = pose[:3, 3]
        robot_points, robot_normals = _keep_facing(sampled[0], sampled[1], origin)
        robot_labels = query.label_points(robot_points, pose, sensor)
        chunks = [(robot_points, robot_normals, robot_labels)]
        if ground:
            ground_points, ground_normals = _ground_samples(
                origin[:2],
                sensor.max_range,
                ground_spacing,
                ground_z,
                limit_half=query._wall_half if walls is not None else None,
            )
            ground_points, ground_normals = _keep_facing(ground_points, ground_normals, origin)
            ground_labels = query.label_points(ground_points, pose, sensor)
            keep = ground_labels > 0
            chunks.append((ground_points[keep], ground_normals[keep], ground_labels[keep]))
        if walls is not None and query._wall_half:
            wall_points, wall_normals = _wall_samples(
                query._wall_half, query._wall_z0, query._wall_z1 - query._wall_z0, ground_spacing
            )
            wall_points, wall_normals = _keep_facing(wall_points, wall_normals, origin)
            wall_labels = query.label_points(wall_points, pose, sensor)
            keep = wall_labels > 0
            chunks.append((wall_points[keep], wall_normals[keep], wall_labels[keep]))

        points = [item[0] for item in chunks if len(item[0])]
        normals = [item[1] for item in chunks if len(item[1])]
        labels = [item[2] for item in chunks if len(item[2])]
        if points:
            coverage_points = np.vstack(points)
            coverage_normals = np.vstack(normals)
            coverage_labels = np.concatenate(labels)
            offset = DISPLAY_OFFSET * (1.0 + 0.35 * layer)
            coverage_points = coverage_points + normalize(coverage_normals) * offset
    return VisibilityResult(
        hits=hits,
        hit_distance=hit_distance,
        coverage_points=coverage_points,
        coverage_normals=coverage_normals,
        coverage_labels=coverage_labels,
        n_rays=n_rays,
    )


def _ray_triangle_distance(origins, directions, v0, edge1, edge2, clearance):
    """Closest hit distance along unit rays. Misses are inf. Möller–Trumbore."""
    pvec = np.cross(directions[:, None, :], edge2[None, :, :])
    det = np.einsum("fj,rfj->rf", edge1, pvec)
    parallel = np.abs(det) < 1e-10
    inv_det = np.zeros_like(det)
    np.divide(1.0, det, out=inv_det, where=~parallel)
    tvec = origins[:, None, :] - v0[None, :, :]
    u = np.einsum("rfj,rfj->rf", tvec, pvec) * inv_det
    qvec = np.cross(tvec, edge1[None, :, :])
    v = np.einsum("rj,rfj->rf", directions, qvec) * inv_det
    distance = np.einsum("fj,rfj->rf", edge2, qvec) * inv_det
    hit = (~parallel) & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1.0 + 1e-8) & (distance >= clearance)
    distance = np.where(hit, distance, np.inf)
    return distance.min(axis=1)


def _keep_facing(points, normals, origin):
    if len(points) == 0:
        return points, normals
    toward = origin.reshape(1, 3) - points
    facing = np.einsum("ij,ij->i", normals, toward) > 0.0
    return points[facing], normals[facing]


def _sample_robot(robot, spacing: float, cap: int):
    point_chunks = []
    normal_chunks = []
    transforms = robot.transforms()
    for name, link in robot.links.items():
        if name not in transforms:
            continue
        for visual in link.visuals:
            mesh = visual.copy()
            if len(mesh.faces) == 0:
                continue
            mesh.apply_transform(transforms[name])
            points, normals = sample_mesh(mesh, spacing)
            if len(points):
                point_chunks.append(points)
                normal_chunks.append(normals)
    if not point_chunks:
        return np.zeros((0, 3)), np.zeros((0, 3))
    points = np.vstack(point_chunks)
    normals = np.vstack(normal_chunks)
    if len(points) > cap:
        index = np.linspace(0, len(points) - 1, cap).astype(int)
        points = points[index]
        normals = normals[index]
    return points, normals


def sample_mesh(mesh: trimesh.Trimesh, spacing: float, max_level: int = 12):
    if mesh is None or len(mesh.faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    areas = np.asarray(mesh.area_faces, dtype=float)
    valid = areas > 1e-12
    if not np.any(valid):
        return np.zeros((0, 3)), np.zeros((0, 3))
    triangles = np.asarray(mesh.triangles, dtype=float)[valid]
    normals = np.asarray(mesh.face_normals, dtype=float)[valid]
    areas = areas[valid]
    points = [triangles.mean(axis=1)]
    out_normals = [normals]
    cell = max(spacing, 1e-3) ** 2
    big = np.flatnonzero(areas > cell * 2.5)
    extra_points = []
    extra_normals = []
    for index in big:
        level = int(np.ceil(np.sqrt(areas[index] / cell)))
        level = max(2, min(level, max_level))
        triangle = triangles[index]
        normal = normals[index]
        for i in range(level):
            for j in range(level - i):
                a = (i + 1.0 / 3.0) / level
                b = (j + 1.0 / 3.0) / level
                c = 1.0 - a - b
                if c <= 1e-6:
                    continue
                extra_points.append(a * triangle[0] + b * triangle[1] + c * triangle[2])
                extra_normals.append(normal)
    if extra_points:
        points.append(np.asarray(extra_points, dtype=float))
        out_normals.append(np.asarray(extra_normals, dtype=float))
    return np.vstack(points), np.vstack(out_normals)


def _ground_samples(
    origin_xy,
    max_range: float,
    spacing: float,
    ground_z: float = GROUND_Z,
    limit_half: float | None = None,
):
    max_range = float(max_range)
    if max_range <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    spacing = max(float(spacing), (2.0 * max_range) / 80.0, 1e-3)
    half = np.arange(0.0, max_range + 1e-9, spacing)
    coords = np.concatenate((-half[:0:-1], half))
    xx, yy = np.meshgrid(coords, coords)
    points = np.column_stack(
        [
            xx.ravel() + float(origin_xy[0]),
            yy.ravel() + float(origin_xy[1]),
            np.full(xx.size, float(ground_z)),
        ]
    )
    offset = points[:, :2] - np.asarray(origin_xy, dtype=float).reshape(1, 2)
    keep = np.einsum("ij,ij->i", offset, offset) <= max_range**2 + 1e-8
    if limit_half is not None:
        wall = float(limit_half)
        keep &= np.abs(points[:, 0]) <= wall + 1e-8
        keep &= np.abs(points[:, 1]) <= wall + 1e-8
    points = points[keep]
    normals = np.zeros_like(points)
    normals[:, 2] = 1.0
    return points, normals


def occupancy_centers(half, ground_z: float, rise: float, cell: float, max_cells: int = 50000):
    """Centers of a square column of cubes, from the ground up to ``rise``."""
    rise = float(rise)
    cell = float(cell)
    if half is None or float(half) <= 0 or rise <= 1e-6 or cell <= 1e-6:
        return np.zeros((0, 3)), 0.0
    half = float(half)
    while True:
        xs = _axis_centers(-half, 2.0 * half, cell)
        zs = _axis_centers(float(ground_z), rise, cell)
        count = len(xs) * len(xs) * len(zs)
        if count == 0:
            return np.zeros((0, 3)), 0.0
        if count <= max_cells:
            break
        cell *= 1.5
    xx, yy, zz = np.meshgrid(xs, xs, zs, indexing="ij")
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())), float(cell)


def _axis_centers(start: float, length: float, cell: float) -> np.ndarray:
    count = int(np.floor(float(length) / float(cell) + 1e-9))
    if count < 1:
        return np.zeros(0)
    return float(start) + (np.arange(count) + 0.5) * float(cell)


def surrounding_half(mesh, requested: float) -> float:
    """Half-width of a square that stays outside the robot."""
    half = max(float(requested), 1.0)
    if mesh is None or len(getattr(mesh, "vertices", [])) == 0:
        return half
    low, high = np.asarray(mesh.bounds, dtype=float)
    reach = float(np.max(np.abs(np.concatenate([low[:2], high[:2]]))))
    return max(half, reach + 0.5)


def _wall_samples(half: float, z0: float, height: float, spacing: float):
    half = float(half)
    height = float(height)
    if half <= 0 or height <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    spacing = max(float(spacing), (2.0 * half) / 60.0, height / 30.0, 0.05)
    along = np.arange(-half, half + 1e-9, spacing)
    up = np.arange(float(z0), float(z0) + height + 1e-9, spacing)
    horizontal, vertical = np.meshgrid(along, up)
    point_chunks = []
    normal_chunks = []
    for axis, sign in ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)):
        points = np.zeros((horizontal.size, 3))
        points[:, axis] = sign * half
        points[:, 1 - axis] = horizontal.ravel()
        points[:, 2] = vertical.ravel()
        normals = np.zeros_like(points)
        normals[:, axis] = -sign
        point_chunks.append(points)
        normal_chunks.append(normals)
    return np.vstack(point_chunks), np.vstack(normal_chunks)
