"""Desktop window for sensor field-of-view and occlusion."""

from __future__ import annotations

import json
import os
import sys
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")


def _select_qt_binding() -> None:
    if os.environ.get("QT_API"):
        return
    for module_name, api_name in (
        ("PySide6", "pyside6"),
        ("PyQt6", "pyqt6"),
        ("PyQt5", "pyqt5"),
        ("PySide2", "pyside2"),
    ):
        try:
            __import__(module_name)
        except ImportError:
            continue
        os.environ["QT_API"] = api_name
        return


_select_qt_binding()

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QFont
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from view_simulate.draw import Frame, coverage_colors, prepare_plotter, render_frame
from view_simulate.robot import (
    bundled_urdf,
    load_urdf,
    manual_base,
    next_link_color,
)
from view_simulate.sensors import PALETTE, Sensor, demo_sensors, frustum_mesh, lidar_body_mesh, sensor_pose
from view_simulate.gpu_rays import device_label
from view_simulate.visibility import evaluate_sensors, surrounding_half


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("传感器视野仿真")
        self.robot = load_urdf(bundled_urdf())
        self.sensors = demo_sensors()
        self._block = False
        self._busy = False
        self._dirty = False
        self._frame = True
        self._solve_occlusion = False
        self._occlusion = {}
        self._occlusion_cache_key = None
        self._last_dir = str(Path.home())
        self._build_ui()
        prepare_plotter(self.plotter)
        self._set_robot(self.robot)

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(central)
        layout.addWidget(self.plotter, 1)
        layout.addWidget(self._build_side())
        self.setCentralWidget(central)
        self.statusBar().showMessage("左键旋转，中键平移，滚轮缩放")
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.refresh)
        self._connect_sensor_form()

    def _build_side(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(430)
        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(10, 10, 10, 10)
        column.addWidget(self._robot_group())
        column.addWidget(self._joint_group())
        column.addWidget(self._sensor_group())
        column.addWidget(self._display_group())
        column.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _robot_group(self) -> QGroupBox:
        box = QGroupBox("机器人")
        layout = QVBoxLayout(box)
        self.robot_info = QLabel("")
        self.robot_info.setWordWrap(True)
        layout.addWidget(self.robot_info)
        grid = QGridLayout()
        buttons = (
            ("载入 URDF…", self._load_urdf),
            ("示例机器人", self._load_demo),
            ("新建底座", self._new_base),
            ("添加连杆…", self._add_link),
            ("删除末端", self._delete_link),
            ("复位视角", self._reset_view),
        )
        for index, (text, slot) in enumerate(buttons):
            button = QPushButton(text)
            button.clicked.connect(slot)
            grid.addWidget(button, index // 2, index % 2)
        layout.addLayout(grid)
        path_row = QHBoxLayout()
        self.package_edit = QLineEdit()
        self.package_edit.setPlaceholderText("package:// 搜索目录，用分号分隔")
        self.package_edit.setToolTip("URDF 里的 package://包名/相对路径 会在这些目录下查找")
        browse = QPushButton("添加目录")
        browse.clicked.connect(self._add_package_dir)
        path_row.addWidget(self.package_edit, 1)
        path_row.addWidget(browse)
        layout.addLayout(path_row)
        link_row = QHBoxLayout()
        self.link_combo = QComboBox()
        link_row.addWidget(self.link_combo, 1)
        layout.addLayout(link_row)
        return box

    def _joint_group(self) -> QGroupBox:
        box = QGroupBox("关节")
        self.joint_layout = QFormLayout(box)
        return box

    def _sensor_group(self) -> QGroupBox:
        box = QGroupBox("传感器")
        layout = QVBoxLayout(box)
        self.sensor_list = QListWidget()
        self.sensor_list.setMaximumHeight(110)
        self.sensor_list.currentRowChanged.connect(self._on_sensor_selected)
        layout.addWidget(self.sensor_list)
        grid = QGridLayout()
        actions = (
            ("添加相机", lambda _checked=False: self._add_sensor("camera")),
            ("添加雷达", lambda _checked=False: self._add_sensor("lidar")),
            ("删除传感器", self._delete_sensor),
            ("保存布置…", self._save_layout),
            ("载入布置…", self._load_layout),
        )
        for index, (text, slot) in enumerate(actions):
            button = QPushButton(text)
            button.clicked.connect(slot)
            grid.addWidget(button, index // 2, index % 2)
        layout.addLayout(grid)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.parent_combo = QComboBox()
        self.pos_widget, self.pos_spins = _triple(("X", "Y", "Z"), (-20, 20), 0.01, 3)
        self.rpy_widget, self.rpy_spins = _triple(("R", "P", "Y"), (-360, 360), 1.0, 2)
        self.hfov_widget, self.hfov_spins = _pair()
        self.vfov_widget, self.vfov_spins = _pair()
        self.range_spin = _double(0.1, 200, 0.1, 2)
        self.h_rays = QSpinBox()
        self.v_rays = QSpinBox()
        self.h_rays.setRange(2, 720)
        self.v_rays.setRange(1, 80)
        form.addRow("名称", self.name_edit)
        form.addRow("安装连杆", self.parent_combo)
        form.addRow("位置 m", self.pos_widget)
        form.addRow("姿态 °", self.rpy_widget)
        form.addRow("水平范围 °", self.hfov_widget)
        form.addRow("垂直范围 °", self.vfov_widget)
        form.addRow("最远距离 m", self.range_spin)
        form.addRow("水平采样", self.h_rays)
        form.addRow("垂直采样", self.v_rays)
        hint = QLabel(
            "姿态是相对父连杆的 rpy，顺序与 URDF 相同。红色轴是视线（+X）。"
            "水平角 0 朝前、正方向朝左；垂直角 0 水平、正方向朝上。"
            "雷达水平 -180 到 180 是一整圈，垂直 0 到 90 是上半球。"
        )
        hint.setWordWrap(True)
        form.addRow(hint)
        layout.addLayout(form)
        self._form_widgets = [
            self.name_edit,
            self.parent_combo,
            self.pos_widget,
            self.rpy_widget,
            self.hfov_widget,
            self.vfov_widget,
            self.range_spin,
            self.h_rays,
            self.v_rays,
        ]
        return box

    def _display_group(self) -> QGroupBox:
        box = QGroupBox("显示")
        layout = QVBoxLayout(box)
        self.solve_button = QPushButton("计算遮挡")
        self.solve_button.setToolTip("一次计算当前要显示的全部传感器。勾选「只显示当前传感器视场」时只算选中的那个。")
        self.solve_button.clicked.connect(self._solve_occlusion_clicked)
        layout.addWidget(self.solve_button)
        self.chk_frustum = QCheckBox("视场体")
        self.chk_coverage = QCheckBox("遮挡着色")
        self.chk_hits = QCheckBox("传感器回波")
        self.chk_ground = QCheckBox("地面参与遮挡")
        self.chk_walls = QCheckBox("四周围墙")
        self.chk_only = QCheckBox("只显示当前传感器视场")
        for box_widget in (
            self.chk_frustum,
            self.chk_coverage,
            self.chk_ground,
            self.chk_walls,
        ):
            box_widget.setChecked(True)
        self.chk_hits.setChecked(False)
        for box_widget in (
            self.chk_frustum,
            self.chk_coverage,
            self.chk_hits,
            self.chk_ground,
            self.chk_walls,
            self.chk_only,
        ):
            box_widget.stateChanged.connect(self.schedule)
            layout.addWidget(box_widget)
        form = QFormLayout()
        self.clearance = _double(0.0, 0.5, 0.005, 3, 0.01)
        self.surface_spacing = _double(0.02, 0.2, 0.01, 3, 0.05)
        self.ground_z = _double(-20, 20, 0.01, 3, -1.28)
        self.ground_spacing = _double(0.05, 0.4, 0.01, 3, 0.10)
        self.cell_size = _double(0.1, 5, 0.1, 2, 0.5)
        self.voxel_opacity = _double(0.05, 1, 0.05, 2, 0.35)
        self.slice_height = _double(0, 30, 0.1, 2, 0)
        self.wall_distance = _double(1, 80, 0.5, 1, 8)
        self.wall_height = _double(0.5, 30, 0.1, 2, 3)
        self.grid_extent = _double(1, 200, 1, 1, 10)
        self.clearance.setToolTip("忽略传感器原点附近这么远的命中，避免装进壳体内部时全部被挡住")
        self.grid_extent.setToolTip("X、Y 刻度从 -N 画到 N，单位米。机器人比这个范围大时会自动再扩大。")
        self.ground_z.setToolTip("遮挡用的水平面高度，相对机器人根连杆，单位米")
        self.cell_size.setToolTip("占用网格每个立方体的边长。格子太多时会自动加大")
        self.voxel_opacity.setToolTip("1 是不透明，数值越小格子越透。改这个不用重新计算")
        self.slice_height.setToolTip("占用网格从地面向上铺到这个高度。0 表示不画占用网格")
        self.wall_distance.setToolTip("四面墙到原点的水平距离。机器人比这个更大时，墙会自动退到机体外侧")
        self.wall_height.setToolTip("墙从地面向上的高度，用来看竖直方向的遮挡")
        form.addRow("近处忽略 m", self.clearance)
        form.addRow("表面间距 m", self.surface_spacing)
        form.addRow("地面高度 m", self.ground_z)
        form.addRow("地面间距 m", self.ground_spacing)
        form.addRow("网格大小 m", self.cell_size)
        form.addRow("网格不透明度", self.voxel_opacity)
        form.addRow("最高高度 m", self.slice_height)
        form.addRow("围墙距离 m", self.wall_distance)
        form.addRow("围墙高度 m", self.wall_height)
        form.addRow("刻度范围 m", self.grid_extent)
        layout.addLayout(form)
        for spin in (
            self.clearance,
            self.surface_spacing,
            self.ground_z,
            self.ground_spacing,
            self.cell_size,
            self.voxel_opacity,
            self.slice_height,
            self.wall_distance,
            self.wall_height,
            self.grid_extent,
        ):
            spin.valueChanged.connect(self.schedule)
        self.legend = QLabel()
        self.legend.setTextFormat(Qt.RichText)
        self.legend.setWordWrap(True)
        layout.addWidget(self.legend)
        self._refresh_legend()
        return box

    def _connect_sensor_form(self) -> None:
        self.name_edit.textChanged.connect(self._on_name)
        self.parent_combo.currentTextChanged.connect(self._on_sensor_edited)
        for spin in (
            *self.pos_spins,
            *self.rpy_spins,
            *self.hfov_spins,
            *self.vfov_spins,
            self.range_spin,
        ):
            spin.valueChanged.connect(self._on_sensor_edited)
        self.h_rays.valueChanged.connect(self._on_sensor_edited)
        self.v_rays.valueChanged.connect(self._on_sensor_edited)

    def schedule(self, *_args) -> None:
        self._timer.start(80)

    def refresh(self, *_args) -> None:
        if self._busy:
            self._dirty = True
            return
        self._busy = True
        try:
            frame, message = self._compute()
            render_frame(self.plotter, frame)
            if frame.reset_camera:
                self._frame = False
            self.statusBar().showMessage(message)
        except Exception as exc:
            traceback.print_exc()
            self.statusBar().showMessage(f"计算失败：{exc}")
        finally:
            self._busy = False
            if self._dirty:
                self._dirty = False
                QTimer.singleShot(0, self.refresh)

    def _compute(self):
        import time

        started = time.perf_counter()
        solve = self._solve_occlusion
        self._solve_occlusion = False
        show_coverage = self.chk_coverage.isChecked()
        show_hits = self.chk_hits.isChecked()
        selected = self.sensor_list.currentRow()
        frustums = []
        axes = []
        only_current = self.chk_only.isChecked()
        targets = []
        for index, sensor in enumerate(self.sensors):
            try:
                pose = sensor_pose(self.robot, sensor)
            except KeyError:
                continue
            chosen = index == selected
            axes.append((pose, 0.28 if chosen else 0.16))
            if sensor.kind == "lidar" and (chosen or not only_current):
                body = lidar_body_mesh()
                body.apply_transform(pose)
                frustums.append((body, sensor.color, 1.0, False))
            if self.chk_frustum.isChecked() and (chosen or not only_current):
                mesh = frustum_mesh(sensor)
                mesh.apply_transform(pose)
                if sensor.kind == "camera":
                    frustums.append((mesh, sensor.color, 0.18 if chosen else 0.06, True))
                else:
                    # Fill stays faint so a full shell does not hide the robot. Edges carry the color.
                    frustums.append((mesh, sensor.color, 0.10 if chosen else 0.05, True))
            if (show_coverage or show_hits) and (chosen or not only_current):
                targets.append(sensor)
        robot_mesh = self.robot.world_mesh()
        walls = self._wall_spec(robot_mesh)
        key = self._occlusion_key(targets)
        results = []
        if targets and solve:
            self.statusBar().showMessage(f"正在计算 {len(targets)} 个传感器的遮挡…")
            self.statusBar().repaint()
            results = evaluate_sensors(
                self.robot,
                targets,
                ground=self.chk_ground.isChecked(),
                ground_z=float(self.ground_z.value()),
                clearance=float(self.clearance.value()),
                surface_spacing=float(self.surface_spacing.value()),
                ground_spacing=float(self.ground_spacing.value()),
                walls=walls,
                slice_height=float(self.slice_height.value()),
                cell_size=float(self.cell_size.value()),
                volume_half=surrounding_half(robot_mesh, float(self.wall_distance.value())),
                want_hits=show_hits,
                want_coverage=show_coverage,
            )
            self._occlusion = {id(sensor): result for sensor, result in zip(targets, results)}
            self._occlusion_cache_key = key
        elif targets and self._occlusion_cache_key == key:
            results = [self._occlusion[id(sensor)] for sensor in targets]
        coverage_points = None
        coverage_colors_arr = None
        voxel_centers = None
        voxel_labels = None
        voxel_size = 0.0
        hit_points = None
        hit_distance = None
        if results and show_coverage:
            point_chunks = [item.coverage_points for item in results if len(item.coverage_points)]
            color_chunks = [coverage_colors(item.coverage_labels) for item in results if len(item.coverage_labels)]
            if point_chunks:
                coverage_points = np.vstack(point_chunks)
                coverage_colors_arr = np.vstack(color_chunks)
            if results[0].voxel_centers is not None and len(results[0].voxel_centers):
                voxel_centers = results[0].voxel_centers
                voxel_labels = results[0].voxel_labels
                voxel_size = float(results[0].voxel_size)
        if results and show_hits:
            hit_chunks = [item.hits for item in results if len(item.hits)]
            distance_chunks = [item.hit_distance for item in results if len(item.hit_distance)]
            if hit_chunks:
                hit_points = np.vstack(hit_chunks)
                hit_distance = np.concatenate(distance_chunks)
        frame = Frame(
            robot_mesh=robot_mesh,
            frustums=frustums,
            axes=axes,
            coverage_points=coverage_points,
            coverage_colors=coverage_colors_arr,
            hit_points=hit_points,
            hit_distance=hit_distance,
            reset_camera=self._frame,
            grid_extent=float(self.grid_extent.value()),
            walls=walls,
            voxel_centers=voxel_centers,
            voxel_labels=voxel_labels,
            voxel_size=voxel_size,
            voxel_opacity=float(self.voxel_opacity.value()),
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        message = self._status_text(targets, results, show_coverage, show_hits, elapsed)
        return frame, message

    def _wall_spec(self, robot_mesh):
        if not self.chk_walls.isChecked():
            return None
        half = surrounding_half(robot_mesh, float(self.wall_distance.value()))
        return (half, float(self.ground_z.value()), float(self.wall_height.value()))

    def _occlusion_key(self, sensors) -> tuple:
        joint_state = tuple(
            (name, round(float(joint.value), 6)) for name, joint in self.robot.joints.items()
        )
        links = tuple((name, len(link.visuals)) for name, link in self.robot.links.items())
        sensor_keys = tuple(
            (
                id(sensor),
                sensor.kind,
                sensor.parent,
                tuple(round(float(value), 5) for value in sensor.xyz),
                tuple(round(float(value), 5) for value in sensor.rpy),
                tuple(round(float(value), 4) for value in sensor.fov_limits_deg()),
                round(float(sensor.max_range), 4),
                int(sensor.h_rays),
                int(sensor.v_rays),
            )
            for sensor in sensors
        )
        return (
            links,
            joint_state,
            sensor_keys,
            bool(self.chk_ground.isChecked()),
            round(float(self.ground_z.value()), 5),
            round(float(self.clearance.value()), 5),
            round(float(self.surface_spacing.value()), 5),
            round(float(self.ground_spacing.value()), 5),
            round(float(self.cell_size.value()), 3),
            round(float(self.slice_height.value()), 3),
            bool(self.chk_walls.isChecked()),
            round(float(self.wall_distance.value()), 3),
            round(float(self.wall_height.value()), 3),
            bool(self.chk_coverage.isChecked()),
            bool(self.chk_hits.isChecked()),
        )

    def _solve_occlusion_clicked(self, *_args) -> None:
        if not (self.chk_coverage.isChecked() or self.chk_hits.isChecked()):
            self.statusBar().showMessage("请先勾选「遮挡着色」或「传感器回波」")
            return
        if not self.sensors:
            self.statusBar().showMessage("先添加一个传感器")
            return
        if self.chk_only.isChecked() and self._current() is None:
            self.statusBar().showMessage("先选择一个传感器")
            return
        self._solve_occlusion = True
        self.refresh()

    def _status_text(self, sensors, results, show_coverage, show_hits, elapsed) -> str:
        head = f"{self.robot.name}，{len(self.robot.links)} 个连杆"
        if sensors and not results and (show_coverage or show_hits):
            count = len(sensors)
            return f"{head}  |  {count} 个传感器  |  点「计算遮挡」更新  |  {elapsed:.0f} ms"
        if not results:
            return f"{head}  |  {elapsed:.0f} ms"
        parts = [head, f"{len(results)} 个传感器"]
        if show_hits:
            echoes = sum(len(item.hits) for item in results)
            rays = sum(item.n_rays for item in results)
            parts.append(f"回波 {echoes}/{rays}")
        if show_coverage:
            label_chunks = [item.coverage_labels for item in results if len(item.coverage_labels)]
            labels = np.concatenate(label_chunks) if label_chunks else np.zeros(0, dtype=np.int8)
            parts.append(f"可见 {int(np.sum(labels == 2))}")
            parts.append(f"被挡 {int(np.sum(labels == 1))}")
            parts.append(f"视场外 {int(np.sum(labels == 0))}")
        parts.append(device_label())
        parts.append(f"{elapsed:.0f} ms")
        return "  |  ".join(parts)

    def _set_robot(self, robot, frame: bool = True) -> None:
        self.robot = robot
        names = set(robot.link_names())
        for sensor in self.sensors:
            if sensor.parent not in names and robot.root:
                sensor.parent = robot.root
        self.robot_info.setText(
            f"{robot.name}，{len(robot.links)} 个连杆，{len(robot.movable_joints())} 个活动关节"
        )
        self._reload_parents()
        self._reload_link_combo()
        self._rebuild_joints()
        self._reload_sensor_list()
        self._load_form()
        self._frame = frame
        self.refresh()

    def _reload_parents(self) -> None:
        current = self.parent_combo.currentText()
        self.parent_combo.blockSignals(True)
        self.parent_combo.clear()
        self.parent_combo.addItems(self.robot.link_names())
        index = self.parent_combo.findText(current)
        if index >= 0:
            self.parent_combo.setCurrentIndex(index)
        self.parent_combo.blockSignals(False)

    def _reload_link_combo(self) -> None:
        current = self.link_combo.currentText()
        self.link_combo.blockSignals(True)
        self.link_combo.clear()
        self.link_combo.addItems(self.robot.link_names())
        index = self.link_combo.findText(current)
        if index >= 0:
            self.link_combo.setCurrentIndex(index)
        self.link_combo.blockSignals(False)

    def _rebuild_joints(self) -> None:
        _clear_layout(self.joint_layout)
        joints = self.robot.movable_joints()
        if not joints:
            self.joint_layout.addRow(QLabel("没有可动关节"))
            return
        for joint in joints:
            if joint.type == "prismatic":
                lo = min(joint.lower, joint.upper)
                hi = max(joint.lower, joint.upper)
                spin = _double(lo, hi, 0.01, 3, joint.value)
                spin.setSuffix(" m")
                spin.valueChanged.connect(
                    lambda value, name=joint.name: self._set_joint(name, float(value), False)
                )
            else:
                lo = np.rad2deg(min(joint.lower, joint.upper))
                hi = np.rad2deg(max(joint.lower, joint.upper))
                spin = _double(lo, hi, 1.0, 2, np.rad2deg(joint.value))
                spin.setSuffix(" °")
                spin.valueChanged.connect(
                    lambda value, name=joint.name: self._set_joint(name, float(value), True)
                )
            self.joint_layout.addRow(joint.name, spin)

    def _set_joint(self, name: str, value: float, degrees: bool) -> None:
        joint = self.robot.joints[name]
        joint.set_value(np.deg2rad(value) if degrees else value)
        self.schedule()

    def _refresh_legend(self) -> None:
        shells = []
        for sensor in self.sensors:
            red, green, blue = (int(round(channel * 255)) for channel in sensor.color[:3])
            shells.append(f"<span style='color:rgb({red},{green},{blue})'>■ {sensor.name}</span>")
        shell_line = "视场体：" + "&nbsp;&nbsp;".join(shells) if shells else ""
        self.legend.setText(
            "<span style='color:#2ecc71'>● 可见</span>&nbsp;&nbsp;"
            "<span style='color:#e64646'>● 被遮挡</span>&nbsp;&nbsp;"
            "<span style='color:#8b919a'>● 视场外</span>"
            + (f"<br>{shell_line}" if shell_line else "")
        )

    def _reload_sensor_list(self) -> None:
        row = self.sensor_list.currentRow()
        self.sensor_list.blockSignals(True)
        self.sensor_list.clear()
        for sensor in self.sensors:
            kind = "相机" if sensor.kind == "camera" else "雷达"
            item = QListWidgetItem(f"{sensor.name}    {kind}")
            item.setForeground(QColor.fromRgbF(*sensor.color[:3]))
            self.sensor_list.addItem(item)
        self._refresh_legend()
        if self.sensors:
            self.sensor_list.setCurrentRow(max(0, min(row, len(self.sensors) - 1)))
        self.sensor_list.blockSignals(False)

    def _on_sensor_selected(self, _row: int) -> None:
        self._load_form()
        self.schedule()

    def _load_form(self) -> None:
        sensor = self._current()
        self._block = True
        try:
            enabled = sensor is not None
            for widget in self._form_widgets:
                widget.setEnabled(enabled)
            if sensor is None:
                return
            self.name_edit.setText(sensor.name)
            self._select_parent(sensor.parent)
            for spin, value in zip(self.pos_spins, sensor.xyz):
                spin.setValue(float(value))
            for spin, value in zip(self.rpy_spins, np.rad2deg(sensor.rpy)):
                spin.setValue(float(value))
            if sensor.kind == "camera":
                for spin in (*self.hfov_spins, *self.vfov_spins):
                    spin.setRange(-89, 89)
                self.h_rays.setRange(4, 120)
                self.v_rays.setRange(4, 80)
            else:
                for spin in self.hfov_spins:
                    spin.setRange(-180, 180)
                for spin in self.vfov_spins:
                    spin.setRange(-90, 90)
                self.h_rays.setRange(4, 360)
                self.v_rays.setRange(1, 40)
            h0, h1, v0, v1 = sensor.fov_limits_deg()
            self.hfov_spins[0].setValue(h0)
            self.hfov_spins[1].setValue(h1)
            self.vfov_spins[0].setValue(v0)
            self.vfov_spins[1].setValue(v1)
            self.range_spin.setValue(float(sensor.max_range))
            self.h_rays.setValue(int(sensor.h_rays))
            self.v_rays.setValue(int(sensor.v_rays))
        finally:
            self._block = False

    def _select_parent(self, name: str) -> None:
        index = self.parent_combo.findText(name)
        if index < 0:
            self.parent_combo.addItem(name)
            index = self.parent_combo.count() - 1
        self.parent_combo.setCurrentIndex(index)

    def _current(self) -> Sensor | None:
        row = self.sensor_list.currentRow()
        if row < 0 or row >= len(self.sensors):
            return None
        return self.sensors[row]

    def _on_name(self, text: str) -> None:
        if self._block:
            return
        sensor = self._current()
        if sensor is None:
            return
        sensor.name = text.strip() or sensor.name
        item = self.sensor_list.currentItem()
        if item is not None:
            kind = "相机" if sensor.kind == "camera" else "雷达"
            item.setText(f"{sensor.name}    {kind}")
            item.setForeground(QColor.fromRgbF(*sensor.color[:3]))
        self._refresh_legend()

    def _on_sensor_edited(self, *_args) -> None:
        if self._block:
            return
        sensor = self._current()
        if sensor is None:
            return
        sensor.parent = self.parent_combo.currentText() or sensor.parent
        sensor.xyz = np.array([spin.value() for spin in self.pos_spins], dtype=float)
        sensor.rpy = np.deg2rad([spin.value() for spin in self.rpy_spins])
        sensor.hfov_min_deg = float(self.hfov_spins[0].value())
        sensor.hfov_max_deg = float(self.hfov_spins[1].value())
        sensor.vfov_min_deg = float(self.vfov_spins[0].value())
        sensor.vfov_max_deg = float(self.vfov_spins[1].value())
        sensor.hfov_deg = abs(sensor.hfov_max_deg - sensor.hfov_min_deg)
        sensor.vfov_deg = abs(sensor.vfov_max_deg - sensor.vfov_min_deg)
        sensor.max_range = float(self.range_spin.value())
        sensor.h_rays = int(self.h_rays.value())
        sensor.v_rays = int(self.v_rays.value())
        self.schedule()

    def _add_sensor(self, kind: str) -> None:
        parent = self.robot.root or ""
        index = len(self.sensors)
        if kind == "camera":
            sensor = Sensor(
                name=f"相机{index + 1}",
                kind="camera",
                parent=parent,
                xyz=[0.30, 0.0, 0.20],
                rpy=[0.0, 0.0, 0.0],
                hfov_deg=70,
                vfov_deg=50,
                max_range=4,
                h_rays=48,
                v_rays=32,
                color=PALETTE[index % len(PALETTE)],
            )
        else:
            sensor = Sensor(
                name=f"雷达{index + 1}",
                kind="lidar",
                parent=parent,
                xyz=[0.0, 0.0, 0.30],
                rpy=[0.0, 0.0, 0.0],
                hfov_deg=360,
                vfov_deg=60,
                max_range=4,
                h_rays=120,
                v_rays=12,
                color=PALETTE[index % len(PALETTE)],
            )
        self.sensors.append(sensor)
        self._reload_sensor_list()
        self.sensor_list.setCurrentRow(len(self.sensors) - 1)
        self._load_form()
        self.schedule()

    def _delete_sensor(self, *_args) -> None:
        row = self.sensor_list.currentRow()
        if row < 0:
            return
        del self.sensors[row]
        self._reload_sensor_list()
        self._load_form()
        self.schedule()

    def _save_layout(self, *_args) -> None:
        path, _selected = QFileDialog.getSaveFileName(
            self, "保存传感器布置", str(Path(self._last_dir) / "sensors.json"), "JSON (*.json)"
        )
        if not path:
            return
        self._last_dir = str(Path(path).parent)
        payload = {"sensors": [sensor.to_dict() for sensor in self.sensors]}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.statusBar().showMessage(f"已保存 {path}")

    def _load_layout(self, *_args) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "载入传感器布置", self._last_dir, "JSON (*.json)"
        )
        if not path:
            return
        self._last_dir = str(Path(path).parent)
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            sensors = [Sensor.from_dict(item, PALETTE[i % len(PALETTE)]) for i, item in enumerate(payload["sensors"])]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "无法载入布置", str(exc))
            return
        self.sensors = sensors
        self._set_robot(self.robot, frame=False)

    def _load_urdf(self, *_args) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "载入 URDF", self._last_dir, "URDF (*.urdf *.xml)"
        )
        if not path:
            return
        self._last_dir = str(Path(path).parent)
        try:
            robot = load_urdf(path, self._package_paths())
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "无法载入 URDF", str(exc))
            return
        self._set_robot(robot, frame=True)
        if robot.warnings:
            QMessageBox.warning(self, "URDF 警告", "\n".join(robot.warnings[:15]))

    def _load_demo(self, *_args) -> None:
        self.sensors = demo_sensors()
        self._set_robot(load_urdf(bundled_urdf()), frame=True)

    def _new_base(self, *_args) -> None:
        self.sensors = [
            Sensor(
                name="前视相机",
                kind="camera",
                parent="base_link",
                xyz=[0.32, 0.0, 0.20],
                rpy=[0.0, 0.0, 0.0],
                hfov_deg=80,
                vfov_deg=55,
                max_range=4,
                h_rays=48,
                v_rays=32,
                color=PALETTE[0],
            )
        ]
        self._set_robot(manual_base(), frame=True)

    def _add_link(self, *_args) -> None:
        dialog = AddLinkDialog(self, self.robot)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            self.robot.attach_primitive(**dialog.values())
        except ValueError as exc:
            QMessageBox.warning(self, "添加连杆", str(exc))
            return
        self._set_robot(self.robot, frame=False)

    def _delete_link(self, *_args) -> None:
        name = self.link_combo.currentText()
        try:
            self.robot.remove_leaf(name)
        except ValueError as exc:
            QMessageBox.information(self, "无法删除", str(exc))
            return
        for sensor in self.sensors:
            if sensor.parent == name and self.robot.root:
                sensor.parent = self.robot.root
        self._set_robot(self.robot, frame=False)

    def _reset_view(self, *_args) -> None:
        self._frame = True
        self.refresh()

    def _add_package_dir(self, *_args) -> None:
        directory = QFileDialog.getExistingDirectory(self, "添加 ROS 包搜索目录", self._last_dir)
        if not directory:
            return
        self._last_dir = directory
        current = self.package_edit.text().strip()
        self.package_edit.setText(directory if not current else f"{current};{directory}")

    def _package_paths(self) -> list[str]:
        return [part.strip() for part in self.package_edit.text().split(";") if part.strip()]

    def closeEvent(self, event) -> None:
        self.plotter.close()
        super().closeEvent(event)


class AddLinkDialog(QDialog):
    def __init__(self, parent, robot):
        super().__init__(parent)
        self.robot = robot
        self.setWindowTitle("添加连杆")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit()
        self.parent_combo = QComboBox()
        self.parent_combo.addItems(robot.link_names())
        self.joint_type = QComboBox()
        self.joint_type.addItem("固定", "fixed")
        self.joint_type.addItem("旋转", "revolute")
        self.joint_type.addItem("直线", "prismatic")
        self.axis_widget, self.axis_spins = _triple(("X", "Y", "Z"), (-1, 1), 0.1, 2)
        self.axis_spins[2].setValue(1.0)
        self.origin_widget, self.origin_spins = _triple(("X", "Y", "Z"), (-20, 20), 0.01, 3)
        self.origin_spins[0].setValue(0.4)
        self.origin_spins[2].setValue(0.2)
        self.rpy_widget, self.rpy_spins = _triple(("R", "P", "Y"), (-180, 180), 1, 1)
        self.shape = QComboBox()
        self.shape.addItem("盒子", "box")
        self.shape.addItem("圆柱（沿连杆 Z）", "cylinder")
        self.shape.addItem("球", "sphere")
        self.stack = QStackedWidget()
        self.box_widget, self.box_spins = _triple(("X", "Y", "Z"), (0.01, 5), 0.01, 3)
        for spin in self.box_spins:
            spin.setValue(0.2)
        cyl = QWidget()
        cyl_form = QFormLayout(cyl)
        self.cyl_radius = _double(0.005, 2, 0.01, 3, 0.05)
        self.cyl_height = _double(0.01, 5, 0.01, 3, 0.25)
        cyl_form.addRow("半径 m", self.cyl_radius)
        cyl_form.addRow("高度 m", self.cyl_height)
        sph = QWidget()
        sph_form = QFormLayout(sph)
        self.sph_radius = _double(0.005, 2, 0.01, 3, 0.08)
        sph_form.addRow("半径 m", self.sph_radius)
        self.stack.addWidget(self.box_widget)
        self.stack.addWidget(cyl)
        self.stack.addWidget(sph)
        self.shape.currentIndexChanged.connect(self.stack.setCurrentIndex)
        form.addRow("名称", self.name)
        form.addRow("父连杆", self.parent_combo)
        form.addRow("关节", self.joint_type)
        form.addRow("关节轴", self.axis_widget)
        form.addRow("关节原点 m", self.origin_widget)
        form.addRow("原点姿态 °", self.rpy_widget)
        form.addRow("形状", self.shape)
        layout.addLayout(form)
        layout.addWidget(self.stack)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        name = self.name.text().strip()
        if not name:
            QMessageBox.warning(self, "添加连杆", "请填写连杆名称")
            return
        if name in self.robot.links:
            QMessageBox.warning(self, "添加连杆", "连杆名称已存在")
            return
        if self.shape.currentData() == "box" and min(spin.value() for spin in self.box_spins) <= 0:
            QMessageBox.warning(self, "添加连杆", "尺寸必须大于 0")
            return
        self.accept()

    def values(self) -> dict:
        shape = self.shape.currentData()
        if shape == "box":
            size = [spin.value() for spin in self.box_spins]
        elif shape == "cylinder":
            size = [self.cyl_radius.value(), self.cyl_height.value(), 0.0]
        else:
            size = [self.sph_radius.value(), 0.0, 0.0]
        return {
            "name": self.name.text().strip(),
            "parent": self.parent_combo.currentText(),
            "joint_type": self.joint_type.currentData(),
            "axis": [spin.value() for spin in self.axis_spins],
            "origin_xyz": [spin.value() for spin in self.origin_spins],
            "origin_rpy": np.deg2rad([spin.value() for spin in self.rpy_spins]),
            "shape": shape,
            "size": size,
            "color": next_link_color(self.robot),
        }


def _double(lo, hi, step, decimals, value=0.0) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(float(lo), float(hi))
    spin.setSingleStep(float(step))
    spin.setDecimals(int(decimals))
    spin.setValue(float(value))
    return spin


def _pair():
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    spins = []
    for title in ("起", "止"):
        layout.addWidget(QLabel(title))
        spin = _double(-180, 180, 1, 1)
        spin.setToolTip(title)
        layout.addWidget(spin, 1)
        spins.append(spin)
    return widget, spins


def _triple(titles, bounds, step, decimals):
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    spins = []
    for title in titles:
        layout.addWidget(QLabel(title))
        spin = _double(bounds[0], bounds[1], step, decimals)
        spin.setToolTip(title)
        layout.addWidget(spin, 1)
        spins.append(spin)
    return widget, spins


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        child = item.layout()
        if child is not None:
            _clear_layout(child)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI" if os.name == "nt" else "Noto Sans CJK SC", 10))
    window = MainWindow()
    window.resize(1440, 900)
    window.show()
    sys.exit(app.exec())
