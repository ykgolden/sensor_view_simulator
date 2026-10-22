import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import trimesh

from view_simulate.mathutil import xyz_rpy_matrix
from view_simulate.robot import Joint, Link, RobotModel, load_urdf, load_urdf_text, make_primitive
from view_simulate.sensors import (
    Sensor,
    demo_sensors,
    frustum_mesh,
    lidar_body_mesh,
    points_in_fov,
    ray_directions,
    sensor_pose,
)
from view_simulate.draw import coverage_colors
from view_simulate.visibility import SceneQuery, _ground_samples, evaluate_sensor, evaluate_sensors, occupancy_centers


def _box_at(center, size):
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(center)
    return mesh


class MathTests(unittest.TestCase):
    def test_urdf_yaw(self):
        transform = xyz_rpy_matrix([0, 0, 0], [0, 0, np.pi / 2])
        point = transform @ np.array([1.0, 0.0, 0.0, 1.0])
        self.assertTrue(np.allclose(point[:3], [0.0, 1.0, 0.0], atol=1e-8))

    def test_revolute_motion(self):
        joint = Joint(
            name="j",
            parent="a",
            child="b",
            type="revolute",
            origin=np.eye(4),
            axis=np.array([0.0, 0.0, 1.0]),
            lower=-np.pi,
            upper=np.pi,
            value=np.pi / 2,
        )
        point = joint.motion() @ np.array([1.0, 0.0, 0.0, 1.0])
        self.assertTrue(np.allclose(point[:3], [0.0, 1.0, 0.0], atol=1e-8))


class SensorTests(unittest.TestCase):
    def test_camera_rays_stay_inside_fov(self):
        sensor = Sensor("c", "camera", "base", [0, 0, 0], [0, 0, 0], hfov_deg=60, vfov_deg=40, max_range=5)
        directions = ray_directions(sensor)
        mask, _distance = points_in_fov(directions, sensor)
        self.assertTrue(mask.all())
        behind, _distance = points_in_fov(np.array([[-1.0, 0.0, 0.0]]), sensor)
        self.assertFalse(behind[0])

    def test_lidar_wraps_when_full_circle(self):
        sensor = Sensor(
            "l", "lidar", "base", [0, 0, 0], [0, 0, 0], hfov_deg=360, vfov_deg=20, max_range=5, h_rays=8, v_rays=1
        )
        mask, _distance = points_in_fov(np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), sensor)
        self.assertTrue(mask.all())
        narrow = Sensor(
            "l2", "lidar", "base", [0, 0, 0], [0, 0, 0], hfov_deg=90, vfov_deg=20, max_range=5
        )
        mask, _distance = points_in_fov(np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), narrow)
        self.assertEqual(list(mask), [False, True])

    def test_yawed_rays_point_left(self):
        sensor = Sensor("c", "camera", "base", [0, 0, 0], [0, 0, 0], hfov_deg=10, vfov_deg=10, h_rays=3, v_rays=3)
        pose = xyz_rpy_matrix([0, 0, 0], [0, 0, np.pi / 2])
        world = ray_directions(sensor) @ pose[:3, :3].T
        mean = world.mean(axis=0)
        self.assertGreater(mean[1], 0.9)
        self.assertLess(abs(mean[0]), 0.2)

    def test_lidar_upper_hemisphere(self):
        sensor = Sensor(
            "l",
            "lidar",
            "base",
            [0, 0, 0],
            [0, 0, 0],
            hfov_min_deg=-180,
            hfov_max_deg=180,
            vfov_min_deg=0,
            vfov_max_deg=90,
            max_range=5,
        )
        mask, _distance = points_in_fov(
            np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.2]]),
            sensor,
        )
        self.assertEqual(list(mask), [True, True, False, True])

    def test_camera_asymmetric_range(self):
        sensor = Sensor(
            "c",
            "camera",
            "base",
            [0, 0, 0],
            [0, 0, 0],
            hfov_min_deg=-10,
            hfov_max_deg=40,
            vfov_min_deg=-5,
            vfov_max_deg=20,
            max_range=5,
        )
        left = np.array([[1.0, np.tan(np.deg2rad(30)), 0.0]])
        right = np.array([[1.0, np.tan(np.deg2rad(-30)), 0.0]])
        mask, _distance = points_in_fov(np.vstack([left, right]), sensor)
        self.assertEqual(list(mask), [True, False])

    def test_old_layout_stays_symmetric(self):
        sensor = Sensor.from_dict({"name": "c", "kind": "camera", "hfov_deg": 60, "vfov_deg": 40})
        self.assertEqual(sensor.fov_limits_deg(), (-30.0, 30.0, -20.0, 20.0))

    def test_frustum_reaches_declared_range(self):
        camera = Sensor("c", "camera", "base", [0, 0, 0], [0, 0, 0], hfov_deg=60, vfov_deg=40, max_range=2.5)
        mesh = frustum_mesh(camera)
        self.assertTrue(np.isclose(np.max(mesh.vertices[:, 0]), 2.5))
        lidar = Sensor("l", "lidar", "base", [0, 0, 0], [0, 0, 0], hfov_deg=120, vfov_deg=30, max_range=4)
        shell = frustum_mesh(lidar)
        self.assertTrue(np.isclose(np.max(np.linalg.norm(shell.vertices, axis=1)), 4.0, atol=1e-6))
        body = lidar_body_mesh()
        self.assertLess(np.max(np.linalg.norm(body.vertices, axis=1)), 0.1)
        self.assertGreater(len(body.faces), 0)


class OcclusionTests(unittest.TestCase):
    def test_near_wall_hides_far_wall(self):
        mesh = trimesh.util.concatenate(
            [
                _box_at((1.0, 0.0, 0.0), (0.2, 0.8, 0.8)),
                _box_at((2.0, 0.0, 0.0), (0.2, 0.8, 0.8)),
            ]
        )
        sensor = Sensor("c", "camera", "base", [0, 0, 0], [0, 0, 0], hfov_deg=80, vfov_deg=80, max_range=8)
        labels = SceneQuery(mesh, ground=False, clearance=0.001).label_points(
            np.array(
                [
                    [0.9, 0.0, 0.0],
                    [1.9, 0.0, 0.0],
                    [1.9, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            ),
            np.eye(4),
            sensor,
        )
        self.assertEqual(list(labels), [2, 1, 2, 0])

    def test_wall_casts_ground_shadow(self):
        wall = _box_at((1.0, 0.0, 0.5), (0.1, 1.0, 1.0))
        sensor = Sensor("c", "camera", "base", [0, 0, 0], [0, 0, 0], hfov_deg=140, vfov_deg=80, max_range=8)
        pose = np.eye(4)
        pose[2, 3] = 0.6
        labels = SceneQuery(wall, ground=True, clearance=0.001).label_points(
            np.array([[2.0, 0.0, 0.0], [2.0, 1.2, 0.0]]),
            pose,
            sensor,
        )
        self.assertEqual(list(labels), [1, 2], msg=str(labels))

    def test_ground_height_moves_the_plane(self):
        query = SceneQuery(None, ground=True, clearance=0.0, ground_z=-0.5)
        points, distance = query.cast_rays(np.array([0.0, 0.0, 1.0]), np.array([[0.0, 0.0, -1.0]]), 5.0)
        self.assertTrue(np.allclose(points[0], [0.0, 0.0, -0.5], atol=1e-6))
        self.assertTrue(np.isclose(distance[0], 1.5, atol=1e-6))

    def test_wall_blocks_the_center_and_leaves_the_side_open(self):
        mesh = _box_at((1.0, 0.0, 0.5), (0.2, 0.4, 1.0))
        sensor = Sensor("c", "camera", "base", [0, 0, 0], [0, 0, 0], hfov_deg=90, vfov_deg=80, max_range=6)
        query = SceneQuery(mesh, ground=False, clearance=0.001, walls=(2.0, 0.0, 3.0))
        labels = query.label_points(np.array([[2.0, 0.0, 0.5], [2.0, 1.2, 0.5]]), np.eye(4), sensor)
        self.assertEqual(list(labels), [1, 2], msg=str(labels))
        points, distance = SceneQuery(None, ground=False, clearance=0.0, walls=(2.0, 0.0, 3.0)).cast_rays(
            np.zeros(3), np.array([[1.0, 0.0, 0.0]]), 6.0
        )
        self.assertTrue(np.allclose(points[0], [2.0, 0.0, 0.0], atol=1e-6))
        self.assertTrue(np.isclose(distance[0], 2.0, atol=1e-6))

    def test_ground_samples_stop_at_the_walls(self):
        limited, _normals = _ground_samples((0.5, -0.2), 20.0, 0.5, limit_half=2.0)
        open_disk, _normals = _ground_samples((0.5, -0.2), 20.0, 0.5)
        self.assertGreater(len(limited), 0)
        self.assertTrue(np.all(np.abs(limited[:, 0]) <= 2.0 + 1e-6))
        self.assertTrue(np.all(np.abs(limited[:, 1]) <= 2.0 + 1e-6))
        self.assertTrue(np.any(np.abs(open_disk[:, 0]) > 2.0))

    def test_coverage_stays_red_and_green(self):
        colors = coverage_colors(np.array([0, 1, 2]))
        self.assertEqual(tuple(colors[0]), (168, 174, 184))
        self.assertEqual(tuple(colors[1]), (230, 70, 70))
        self.assertEqual(tuple(colors[2]), (46, 204, 122))

    def test_occupancy_grid_fills_the_column_above_the_ground(self):
        centers, size = occupancy_centers(1.0, -1.0, 1.0, 0.5)
        self.assertTrue(np.isclose(size, 0.5))
        self.assertGreater(len(centers), 0)
        self.assertTrue(np.all(np.abs(centers[:, 0]) <= 1.0))
        self.assertTrue(np.all(np.abs(centers[:, 1]) <= 1.0))
        self.assertTrue(np.allclose(np.unique(np.round(centers[:, 2], 5)), [-0.75, -0.25]))
        self.assertEqual(len(occupancy_centers(1.0, 0.0, 0.0, 0.5)[0]), 0)
        mesh = _box_at((1.0, 0.0, 0.5), (0.2, 2.0, 1.0))
        sensor = Sensor("c", "camera", "base", [0, 0, 0], [0, 0, 0], hfov_deg=90, vfov_deg=80, max_range=6)
        pose = np.eye(4)
        pose[2, 3] = 0.4
        labels = SceneQuery(mesh, ground=True, clearance=0.001, ground_z=0.0).label_points(
            np.array([[2.0, 0.0, 0.5], [2.0, 0.0, 2.0]]),
            pose,
            sensor,
        )
        self.assertEqual(list(labels), [1, 2], msg=str(labels))

    def test_ray_stops_on_first_surface(self):
        mesh = _box_at((1.0, 0.0, 0.0), (0.2, 0.8, 0.8))
        _points, distance = SceneQuery(mesh, ground=False, clearance=0.001).cast_rays(
            np.zeros(3), np.array([[1.0, 0.0, 0.0]]), 5.0
        )
        self.assertTrue(np.isclose(distance[0], 0.9, atol=0.02))


class RobotTests(unittest.TestCase):
    def test_manual_chain(self):
        robot = RobotModel("arm")
        robot.add_root(Link("base", [make_primitive("box", (0.2, 0.2, 0.2), (10, 10, 10, 255))]))
        robot.attach_primitive(
            name="link1",
            parent="base",
            joint_type="revolute",
            axis=[0, 0, 1],
            origin_xyz=[1, 0, 0],
            origin_rpy=[0, 0, 0],
            shape="box",
            size=[0.2, 0.2, 0.2],
            color=(1, 2, 3, 255),
        )
        robot.joints["joint_link1"].set_value(np.pi / 2)
        point = robot.link_transform("link1") @ np.array([1.0, 0.0, 0.0, 1.0])
        self.assertTrue(np.allclose(point[:3], [1.0, 1.0, 0.0], atol=1e-6))
        robot.remove_leaf("link1")
        self.assertNotIn("link1", robot.links)

    def test_bundled_robot_and_head_motion(self):
        robot = load_urdf(Path(__file__).resolve().parents[1] / "view_simulate" / "data" / "simple_robot.urdf")
        camera = demo_sensors()[0]
        before = sensor_pose(robot, camera)
        robot.joints["head_yaw"].set_value(np.pi / 2)
        after = sensor_pose(robot, camera)
        self.assertGreater(abs(after[1, 3] - before[1, 3]), 0.05)

    def test_demo_panel_occludes_ground(self):
        robot = load_urdf(Path(__file__).resolve().parents[1] / "view_simulate" / "data" / "simple_robot.urdf")
        camera = demo_sensors()[0]
        pose = sensor_pose(robot, camera)
        labels = SceneQuery(robot.world_mesh(), ground=True, clearance=0.01).label_points(
            np.array(
                [
                    [2.2, 0.0, 0.0],
                    [2.2, 1.0, 0.0],
                    [1.185, 0.1, 0.40],
                ]
            ),
            pose,
            camera,
        )
        self.assertEqual(list(labels), [1, 2, 2], msg=f"pose={pose[:3, 3]} labels={labels}")

    def test_package_mesh_and_missing_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            box = trimesh.creation.box((0.2, 0.2, 0.2))
            package = root / "mybot"
            package.mkdir()
            box.export(package / "a.stl")
            text = """
            <robot name="m">
              <link name="base_link">
                <visual>
                  <geometry><mesh filename="package://mybot/a.stl"/></geometry>
                  <material name="blue"><color rgba="0 0 1 1"/></material>
                </visual>
              </link>
              <joint name="j" type="fixed">
                <parent link="base_link"/>
                <child link="child"/>
                <origin xyz="0.5 0 0"/>
              </joint>
              <link name="child">
                <visual>
                  <geometry><mesh filename="missing.stl"/></geometry>
                </visual>
              </link>
            </robot>
            """
            robot = load_urdf_text(text, base_dir=root, package_paths=[root])
            self.assertTrue(robot.warnings)
            world = robot.world_mesh()
            self.assertIsNotNone(world)
            self.assertGreater(world.bounds[1, 0], 0.45)
            self.assertLess(world.bounds[1, 0], 0.7)
            self.assertTrue(np.any(np.asarray(world.visual.face_colors)[:, 2] > 200))

    def test_spray_boom_without_visuals_is_still_drawn(self):
        robot = load_urdf(
            Path(__file__).resolve().parents[1] / "view_simulate" / "data" / "spray_boom.urdf"
        )
        mesh = robot.world_mesh()
        self.assertIsNotNone(mesh)
        self.assertGreater(len(mesh.faces), 0)
        self.assertLess(mesh.bounds[0, 0], -8.0)
        self.assertGreater(mesh.bounds[1, 0], 8.0)
        self.assertTrue(any("示意杆件" in warning for warning in robot.warnings))

    def test_xacro_is_rejected(self):
        with self.assertRaises(ValueError):
            load_urdf_text('<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="a"/>', Path("."))

    def test_demo_coverage_has_both_labels_and_is_fast_enough(self):
        robot = load_urdf(Path(__file__).resolve().parents[1] / "view_simulate" / "data" / "simple_robot.urdf")
        started = time.perf_counter()
        result = evaluate_sensor(
            robot,
            demo_sensors()[0],
            ground=True,
            clearance=0.01,
            surface_spacing=0.05,
            ground_spacing=0.12,
            want_hits=True,
            want_coverage=True,
        )
        elapsed = time.perf_counter() - started
        self.assertIn(1, result.coverage_labels)
        self.assertIn(2, result.coverage_labels)
        self.assertGreater(len(result.hits), 10)
        self.assertTrue(np.any(np.abs(result.hits[:, 0] - 1.2) < 0.08))
        self.assertLess(elapsed, 8.0, msg=f"visibility took {elapsed:.2f}s")

    def test_two_sensors_are_both_labeled(self):
        robot = load_urdf(Path(__file__).resolve().parents[1] / "view_simulate" / "data" / "simple_robot.urdf")
        results = evaluate_sensors(
            robot,
            demo_sensors(),
            ground=False,
            surface_spacing=0.08,
            want_hits=False,
            want_coverage=True,
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(len(item.coverage_labels) > 0 for item in results))


if __name__ == "__main__":
    unittest.main()
