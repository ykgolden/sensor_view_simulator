"""URDF loading and a small kinematic tree for manual robots."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
import trimesh

from view_simulate.mathutil import parse_vec, rotation_about_axis, xyz_rpy_matrix

_DEFAULT_RGBA = np.array([150, 156, 166, 255], dtype=np.uint8)
_LINK_COLORS = (
    (70, 110, 160, 255),
    (210, 120, 40, 255),
    (40, 150, 90, 255),
    (150, 70, 170, 255),
    (180, 60, 60, 255),
)


@dataclass
class Link:
    name: str
    visuals: list = field(default_factory=list)


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    type: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    value: float = 0.0

    def motion(self) -> np.ndarray:
        from view_simulate.mathutil import rotation_about_axis

        if self.type == "prismatic":
            axis = np.asarray(self.axis, dtype=float)
            length = np.linalg.norm(axis)
            axis = np.array([0.0, 0.0, 1.0]) if length < 1e-8 else axis / length
            transform = np.eye(4)
            transform[:3, 3] = axis * float(self.value)
            return transform
        if self.type in ("revolute", "continuous"):
            return rotation_about_axis(self.axis, float(self.value))
        return np.eye(4)

    def set_value(self, value: float) -> None:
        value = float(value)
        if self.type == "fixed":
            self.value = 0.0
        elif self.type == "continuous":
            self.value = (value + np.pi) % (2.0 * np.pi) - np.pi
        elif self.type in ("revolute", "prismatic"):
            lo, hi = self.lower, self.upper
            if hi < lo:
                lo, hi = hi, lo
            self.value = float(np.clip(value, lo, hi))
        else:
            self.value = 0.0


class RobotModel:
    def __init__(self, name: str = "robot"):
        self.name = name
        self.links: dict[str, Link] = {}
        self.joints: dict[str, Joint] = {}
        self.root: str | None = None
        self.warnings: list[str] = []
        self.source = ""

    def add_root(self, link: Link) -> None:
        self.links[link.name] = link
        self.root = link.name

    def transforms(self) -> dict[str, np.ndarray]:
        children: dict[str, list[Joint]] = {}
        for joint in self.joints.values():
            children.setdefault(joint.parent, []).append(joint)
        transforms: dict[str, np.ndarray] = {}
        if self.root is None:
            return transforms
        transforms[self.root] = np.eye(4)
        stack = [self.root]
        seen = {self.root}
        while stack:
            parent = stack.pop()
            for joint in children.get(parent, []):
                if joint.child in seen:
                    continue
                transforms[joint.child] = transforms[parent] @ joint.origin @ joint.motion()
                seen.add(joint.child)
                stack.append(joint.child)
        return transforms

    def link_transform(self, name: str) -> np.ndarray:
        transforms = self.transforms()
        if name not in transforms:
            raise KeyError(f"找不到连杆 {name}")
        return transforms[name]

    def link_names(self) -> list[str]:
        children: dict[str, list[str]] = {}
        for joint in self.joints.values():
            children.setdefault(joint.parent, []).append(joint.child)
        names: list[str] = []
        if self.root is not None:
            queue = [self.root]
            seen = set()
            while queue:
                name = queue.pop(0)
                if name in seen or name not in self.links:
                    continue
                seen.add(name)
                names.append(name)
                queue.extend(children.get(name, []))
        for name in self.links:
            if name not in names:
                names.append(name)
        return names

    def movable_joints(self) -> list[Joint]:
        return [
            joint
            for joint in self.joints.values()
            if joint.type in ("revolute", "continuous", "prismatic")
        ]

    def world_mesh(self) -> trimesh.Trimesh | None:
        parts = []
        transforms = self.transforms()
        for name, link in self.links.items():
            if name not in transforms:
                continue
            for visual in link.visuals:
                mesh = visual.copy()
                if len(mesh.faces) == 0:
                    continue
                if mesh.faces.shape[1] != 3:
                    mesh = mesh.triangulate()
                mesh.apply_transform(transforms[name])
                parts.append(mesh)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return trimesh.util.concatenate(parts)

    def attach_primitive(
        self,
        *,
        name: str,
        parent: str,
        joint_type: str,
        axis,
        origin_xyz,
        origin_rpy,
        shape: str,
        size,
        color,
    ) -> None:
        name = name.strip()
        if not name:
            raise ValueError("连杆名称是空的")
        if name in self.links:
            raise ValueError(f"连杆 {name} 已存在")
        if parent not in self.links:
            raise ValueError(f"父连杆 {parent} 不存在")
        size = np.asarray(size, dtype=float)
        if shape == "box":
            ok = size.size >= 3 and np.all(size[:3] > 0)
        elif shape == "cylinder":
            ok = size.size >= 2 and size[0] > 0 and size[1] > 0
        elif shape == "sphere":
            ok = size.size >= 1 and size[0] > 0
        else:
            ok = False
        if not ok:
            raise ValueError("尺寸必须大于 0")
        mesh = make_primitive(shape, size, color)
        link = Link(name, [mesh])
        if joint_type == "prismatic":
            lower, upper = -1.0, 1.0
        elif joint_type == "revolute":
            lower, upper = -np.pi, np.pi
        else:
            joint_type = "fixed"
            lower, upper = 0.0, 0.0
        joint_name = f"joint_{name}"
        suffix = 2
        while joint_name in self.joints:
            joint_name = f"joint_{name}_{suffix}"
            suffix += 1
        joint = Joint(
            name=joint_name,
            parent=parent,
            child=name,
            type=joint_type,
            origin=xyz_rpy_matrix(origin_xyz, origin_rpy),
            axis=np.asarray(axis, dtype=float),
            lower=lower,
            upper=upper,
        )
        self.links[name] = link
        self.joints[joint_name] = joint

    def remove_leaf(self, name: str) -> None:
        if name == self.root:
            raise ValueError("不能删除根连杆")
        if name not in self.links:
            raise ValueError("连杆不存在")
        if any(joint.parent == name for joint in self.joints.values()):
            raise ValueError("只能删除没有子连杆的末端")
        victims = [key for key, joint in self.joints.items() if joint.child == name]
        for key in victims:
            del self.joints[key]
        del self.links[name]


def make_primitive(shape: str, size, rgba) -> trimesh.Trimesh:
    size = np.asarray(size, dtype=float)
    if shape == "box":
        mesh = trimesh.creation.box(extents=(float(size[0]), float(size[1]), float(size[2])))
    elif shape == "cylinder":
        mesh = trimesh.creation.cylinder(
            radius=float(size[0]), height=float(size[1]), sections=24
        )
    elif shape == "sphere":
        mesh = trimesh.creation.icosphere(radius=float(size[0]), subdivisions=2)
    else:
        raise ValueError(f"未知形状 {shape}")
    color = np.array(rgba, dtype=np.uint8)
    if color.shape == (3,):
        color = np.append(color, 255)
    mesh.visual.face_colors = color
    return mesh


def manual_base() -> RobotModel:
    robot = RobotModel("manual")
    mesh = make_primitive("box", (0.50, 0.40, 0.22), (70, 110, 160, 255))
    mesh.apply_translation((0.0, 0.0, 0.13))
    robot.add_root(Link("base_link", [mesh]))
    robot.source = "手动底座"
    return robot


def next_link_color(robot: RobotModel) -> tuple[int, int, int, int]:
    return _LINK_COLORS[len(robot.links) % len(_LINK_COLORS)]


def bundled_urdf() -> Path:
    resource = files("view_simulate").joinpath("data/simple_robot.urdf")
    return Path(str(resource))


def load_urdf(path, package_paths=None) -> RobotModel:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件 {path}")
    return load_urdf_text(
        _read_text(path),
        base_dir=path.parent,
        package_paths=package_paths,
        source=str(path),
    )


def load_urdf_text(text: str, base_dir: Path, package_paths=None, source: str = "") -> RobotModel:
    if "${" in text or "<xacro:" in text or "xmlns:xacro" in text:
        raise ValueError("这是 xacro，请先展开成 URDF。例如：xacro model.xacro -o model.urdf")
    try:
        root_elem = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"URDF XML 无法解析：{exc}") from exc
    if _local(root_elem) != "robot":
        raise ValueError("根节点不是 robot")

    robot = RobotModel(root_elem.get("name") or "robot")
    robot.source = source
    base_dir = Path(base_dir)
    package_paths = [Path(item) for item in (package_paths or [])]
    materials = _material_library(root_elem)

    for elem in list(root_elem):
        if _local(elem) != "link":
            continue
        name = elem.get("name") or ""
        if not name:
            robot.warnings.append("有一个连杆缺少 name，已跳过")
            continue
        if name in robot.links:
            robot.warnings.append(f"连杆 {name} 重复定义，已忽略后一次")
            continue
        visuals = _link_visuals(elem, name, base_dir, package_paths, materials, robot.warnings)
        robot.links[name] = Link(name, visuals)

    claimed: set[str] = set()
    for elem in list(root_elem):
        if _local(elem) != "joint":
            continue
        joint = _parse_joint(elem, robot.warnings)
        if joint is None:
            continue
        if joint.child in claimed:
            robot.warnings.append(f"连杆 {joint.child} 有多个父关节，已忽略 {joint.name}")
            continue
        claimed.add(joint.child)
        for link_name in (joint.parent, joint.child):
            if link_name not in robot.links:
                robot.links[link_name] = Link(link_name, [])
                robot.warnings.append(f"关节引用了未定义连杆 {link_name}，已补一个空连杆")
        robot.joints[joint.name] = joint

    if not robot.links:
        raise ValueError("URDF 里没有连杆")
    children = {joint.child for joint in robot.joints.values()}
    roots = [name for name in robot.links if name not in children]
    if "base_link" in roots:
        robot.root = "base_link"
    elif roots:
        robot.root = roots[0]
    else:
        robot.root = next(iter(robot.links))
    if len(roots) > 1:
        robot.warnings.append(f"有多个根连杆，使用 {robot.root}")
    _add_kinematic_visuals(robot)
    return robot


def _add_kinematic_visuals(robot: RobotModel) -> None:
    """Give links that have no mesh a bar along each outgoing joint offset."""
    created = False
    for link in robot.links.values():
        if link.visuals:
            continue
        pieces = []
        for joint in robot.joints.values():
            if joint.parent != link.name:
                continue
            offset = np.asarray(joint.origin[:3, 3], dtype=float)
            length = float(np.linalg.norm(offset))
            if length < 0.02:
                continue
            radius = float(np.clip(length * 0.02, 0.03, 0.08))
            pieces.append(_cylinder_between(np.zeros(3), offset, radius))
        if not pieces:
            pieces.append(trimesh.creation.icosphere(radius=0.06, subdivisions=1))
        color = _placeholder_color(link.name)
        for mesh in pieces:
            mesh.visual.face_colors = color
        link.visuals.extend(pieces)
        created = True
    if created:
        robot.warnings.append("有连杆没有视觉网格，已按关节间距生成示意杆件")


def _placeholder_color(name: str) -> np.ndarray:
    lowered = name.lower()
    if "_l" in lowered or "left" in lowered:
        rgb = (70, 130, 180, 255)
    elif "_r" in lowered or "right" in lowered:
        rgb = (210, 130, 60, 255)
    else:
        rgb = (150, 156, 166, 255)
    return np.array(rgb, dtype=np.uint8)


def _cylinder_between(start, end, radius: float) -> trimesh.Trimesh:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
    mesh.apply_translation((0.0, 0.0, length / 2.0))
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        if float(np.dot(z_axis, direction)) < 0.0:
            mesh.apply_transform(rotation_about_axis([1.0, 0.0, 0.0], np.pi))
    else:
        angle = float(np.arctan2(axis_norm, np.dot(z_axis, direction)))
        mesh.apply_transform(rotation_about_axis(axis, angle))
    mesh.apply_translation(start)
    return mesh


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _local(elem) -> str:
    tag = elem.tag
    if not isinstance(tag, str):
        return ""
    return tag.split("}")[-1]


def _children(elem, name: str):
    for child in list(elem):
        if _local(child) == name:
            yield child


def _first(elem, name: str):
    for child in _children(elem, name):
        return child
    return None


def _material_library(root_elem) -> dict[str, np.ndarray]:
    library = {}
    for elem in _children(root_elem, "material"):
        color = _color_of(elem)
        name = elem.get("name")
        if name and color is not None:
            library[name] = color
    return library


def _color_of(material_elem) -> np.ndarray | None:
    color = _first(material_elem, "color")
    if color is None or not color.get("rgba"):
        return None
    values = [float(part) for part in color.get("rgba").split()]
    if len(values) == 3:
        values.append(1.0)
    if len(values) != 4:
        raise ValueError(f"颜色格式无法识别：{color.get('rgba')}")
    rgba = np.clip(np.array(values, dtype=float), 0.0, 1.0)
    return np.round(rgba * 255).astype(np.uint8)


def _link_visuals(link_elem, link_name, base_dir, package_paths, materials, warnings):
    visuals = list(_children(link_elem, "visual"))
    elements = visuals or list(_children(link_elem, "collision"))
    meshes = []
    for elem in elements:
        try:
            mesh = _visual_mesh(elem, base_dir, package_paths, materials)
        except (OSError, ValueError) as exc:
            warnings.append(f"{link_name}: {exc}")
            continue
        if mesh is None or len(mesh.faces) == 0:
            warnings.append(f"{link_name}: 几何体没有三角面")
            continue
        meshes.append(mesh)
    return meshes


def _visual_mesh(elem, base_dir, package_paths, materials) -> trimesh.Trimesh | None:
    geometry = _first(elem, "geometry")
    if geometry is None:
        raise ValueError("缺少 geometry")
    mesh = None
    for child in list(geometry):
        kind = _local(child)
        if kind == "box":
            size = parse_vec(child.get("size"), (1.0, 1.0, 1.0))
            mesh = trimesh.creation.box(extents=size)
        elif kind == "cylinder":
            radius = float(child.get("radius") or 0.05)
            length = float(child.get("length") or 0.1)
            mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=24)
        elif kind == "sphere":
            radius = float(child.get("radius") or 0.05)
            mesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        elif kind == "mesh":
            mesh = _load_mesh_file(child, base_dir, package_paths)
        else:
            raise ValueError(f"不支持的几何类型 {kind}")
        break
    if mesh is None:
        raise ValueError("geometry 是空的")
    origin = _origin_matrix(_first(elem, "origin"))
    mesh.apply_transform(origin)
    mesh.visual.face_colors = _material_color(_first(elem, "material"), materials)
    return mesh


def _material_color(material_elem, library) -> np.ndarray:
    if material_elem is None:
        return _DEFAULT_RGBA.copy()
    color = _color_of(material_elem)
    if color is not None:
        return color
    name = material_elem.get("name")
    if name and name in library:
        return library[name].copy()
    return _DEFAULT_RGBA.copy()


def _origin_matrix(origin_elem) -> np.ndarray:
    if origin_elem is None:
        return np.eye(4)
    xyz = parse_vec(origin_elem.get("xyz"), (0.0, 0.0, 0.0))
    rpy = parse_vec(origin_elem.get("rpy"), (0.0, 0.0, 0.0))
    return xyz_rpy_matrix(xyz, rpy)


def _load_mesh_file(mesh_elem, base_dir: Path, package_paths) -> trimesh.Trimesh:
    filename = mesh_elem.get("filename")
    if not filename:
        raise ValueError("mesh 缺少 filename")
    path = _resolve_mesh(filename.strip(), base_dir, package_paths)
    loaded = trimesh.load(str(path), force="mesh", process=False)
    mesh = _as_single_mesh(loaded)
    scale = parse_vec(mesh_elem.get("scale"), (1.0, 1.0, 1.0))
    if np.any(np.abs(scale - 1.0) > 1e-12):
        mesh.apply_scale(scale)
    return mesh


def _as_single_mesh(loaded) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        dumped = loaded.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            return dumped
        if isinstance(dumped, (list, tuple)) and dumped:
            return trimesh.util.concatenate(list(dumped))
    raise ValueError("网格里没有三角面")


def _resolve_mesh(filename: str, base_dir: Path, package_paths) -> Path:
    if filename.startswith("package://"):
        rest = filename[len("package://") :]
        package, separator, relative = rest.partition("/")
        if not separator:
            raise FileNotFoundError(f"找不到网格 {filename}")
        for root in package_paths:
            candidates = (
                root / package / relative,
                root / "share" / package / relative,
                root / package / "share" / package / relative,
            )
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(f"找不到网格 {filename}")
    if filename.startswith("file://"):
        filename = filename[len("file://") :]
    path = Path(filename)
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"找不到网格 {path}")
    return path


def _parse_joint(elem, warnings) -> Joint | None:
    name = elem.get("name") or "joint"
    joint_type = elem.get("type") or "fixed"
    parent_elem = _first(elem, "parent")
    child_elem = _first(elem, "child")
    if parent_elem is None or child_elem is None:
        warnings.append(f"关节 {name} 缺少 parent 或 child")
        return None
    parent = parent_elem.get("link")
    child = child_elem.get("link")
    if not parent or not child or parent == child:
        warnings.append(f"关节 {name} 的连杆引用无效")
        return None
    axis_elem = _first(elem, "axis")
    axis = parse_vec(axis_elem.get("xyz") if axis_elem is not None else None, (0.0, 0.0, 1.0))
    limit_elem = _first(elem, "limit")
    if joint_type == "continuous":
        lower, upper = -np.pi, np.pi
    elif joint_type == "revolute":
        lower = _attr_float(limit_elem, "lower", -np.pi)
        upper = _attr_float(limit_elem, "upper", np.pi)
    elif joint_type == "prismatic":
        lower = _attr_float(limit_elem, "lower", -1.0)
        upper = _attr_float(limit_elem, "upper", 1.0)
    elif joint_type == "fixed":
        lower, upper = 0.0, 0.0
    else:
        warnings.append(f"关节 {name} 类型 {joint_type} 按固定关节处理")
        joint_type = "fixed"
        lower, upper = 0.0, 0.0
    return Joint(
        name=name,
        parent=parent,
        child=child,
        type=joint_type,
        origin=_origin_matrix(_first(elem, "origin")),
        axis=axis,
        lower=lower,
        upper=upper,
    )


def _attr_float(elem, name: str, default: float) -> float:
    if elem is None or elem.get(name) is None:
        return default
    return float(elem.get(name))
