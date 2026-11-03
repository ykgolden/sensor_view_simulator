import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from view_simulate.live import LiveServer, occlusion_summary, query, select_cells
from view_simulate.mcp_server import handle


def _result(labels, voxels, size, hits=0, rays=0):
    return SimpleNamespace(
        coverage_labels=np.asarray(labels, dtype=np.int8),
        voxel_centers=np.zeros((len(voxels), 3)),
        voxel_labels=np.asarray(voxels, dtype=np.int8),
        voxel_size=size,
        hits=np.zeros((hits, 3)),
        n_rays=rays,
    )


class SummaryTests(unittest.TestCase):
    def test_stale_result_reports_an_empty_picture(self):
        summary = occlusion_summary(
            [_result([2, 1], [2, 2, 1], 0.5, hits=3, rays=4)],
            stale=True,
            status="点计算遮挡更新",
            sensor_names=["顶部雷达"],
            requested_cell_size=0.5,
            slice_height=2,
            show_coverage=True,
            show_hits=True,
            only_current=False,
        )
        self.assertTrue(summary["stale"])
        self.assertEqual(summary["green"], 0)
        self.assertEqual(summary["red"], 0)
        self.assertEqual(summary["gray"], 0)
        self.assertEqual(summary["per_sensor"], [])
        self.assertIn("重新点", summary["notes"][0])

    def test_fresh_counts_match_drawn_cells(self):
        summary = occlusion_summary(
            [_result([2, 1, 0], [2, 1, 1], 0.75, hits=2, rays=8)],
            stale=False,
            status="绿色占用 1",
            sensor_names=["顶部雷达"],
            requested_cell_size=0.5,
            slice_height=2,
            show_coverage=True,
            show_hits=True,
            only_current=True,
        )
        self.assertEqual(summary["green"], 1)
        self.assertEqual(summary["red"], 2)
        self.assertEqual(summary["gray"], 0)
        self.assertEqual(summary["drawn"], 3)
        self.assertEqual(summary["used_cell_size_m"], 0.75)
        self.assertEqual(summary["per_sensor"][0]["surface_visible"], 1)
        self.assertEqual(summary["per_sensor"][0]["surface_blocked"], 1)
        self.assertEqual(summary["per_sensor"][0]["surface_outside"], 1)
        self.assertEqual(summary["echo_hits"], 2)
        self.assertEqual(summary["echo_rays"], 8)
        self.assertTrue(any("加大" in note for note in summary["notes"]))

    def test_cell_query_keeps_only_the_box(self):
        centers = np.array([[0, 0, 0], [2, 0, 0], [0.5, 0.5, 0.5]])
        labels = np.array([2, 1, 2])
        chosen = select_cells(centers, labels, [0, 0, 0], [1, 1, 1], limit=1)
        self.assertEqual(chosen["total"], 2)
        self.assertTrue(chosen["truncated"])
        self.assertEqual(chosen["cells"][0]["color"], "green")
        self.assertEqual(len(chosen["cells"]), 1)


class ServerTests(unittest.TestCase):
    def test_round_trip_uses_the_handler_on_the_caller_side(self):
        def handler(request):
            self.assertEqual(request["method"], "get_live_state")
            return {"ok": True, "result": {"occlusion": {"green": 4, "red": 1}}}

        server = LiveServer(handler, port=0)
        self.addCleanup(server.stop)
        server.start()
        previous = os.environ.get("VIEW_SIMULATE_LIVE")
        os.environ["VIEW_SIMULATE_LIVE"] = f"127.0.0.1:{server.port}"
        self.addCleanup(self._restore_env, previous)
        result = query("get_live_state", timeout=5)
        self.assertEqual(result["occlusion"]["green"], 4)
        self.assertEqual(result["occlusion"]["red"], 1)

    def test_endpoint_file_records_the_bound_port(self):
        server = LiveServer(lambda _request: {"ok": True, "result": {}}, port=0)
        self.addCleanup(server.stop)
        server.start()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.json"
            previous = os.environ.get("VIEW_SIMULATE_ENDPOINT_FILE")
            os.environ["VIEW_SIMULATE_ENDPOINT_FILE"] = str(path)
            try:
                server.write_endpoint()
                data = json.loads(path.read_text(encoding="utf-8"))
            finally:
                if previous is None:
                    os.environ.pop("VIEW_SIMULATE_ENDPOINT_FILE", None)
                else:
                    os.environ["VIEW_SIMULATE_ENDPOINT_FILE"] = previous
        self.assertEqual(data["port"], server.port)
        self.assertEqual(data["pid"], os.getpid())

    def _restore_env(self, previous):
        if previous is None:
            os.environ.pop("VIEW_SIMULATE_LIVE", None)
        else:
            os.environ["VIEW_SIMULATE_LIVE"] = previous


class McpTests(unittest.TestCase):
    def test_initialize_and_tool_list(self):
        started = handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            }
        )
        self.assertEqual(started["result"]["protocolVersion"], "2025-06-18")
        self.assertIsNone(handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        listed = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = [tool["name"] for tool in listed["result"]["tools"]]
        self.assertEqual(names, ["get_live_state", "get_occlusion_cells"])

    def test_tool_call_reports_a_closed_window(self):
        previous = os.environ.get("VIEW_SIMULATE_LIVE")
        os.environ["VIEW_SIMULATE_LIVE"] = "127.0.0.1:1"
        self.addCleanup(self._restore, previous)
        called = handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "get_live_state", "arguments": {}},
            }
        )
        self.assertTrue(called["result"]["isError"])
        self.assertIn("没有在运行", called["result"]["content"][0]["text"])

    def _restore(self, previous):
        if previous is None:
            os.environ.pop("VIEW_SIMULATE_LIVE", None)
        else:
            os.environ["VIEW_SIMULATE_LIVE"] = previous
