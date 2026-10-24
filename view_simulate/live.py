"""Local snapshot server for the running view window.

The window listens on 127.0.0.1. A separate process asks it for the parameters
and occlusion result currently on screen. One JSON object per connection.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path

import numpy as np

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
PORT_SPAN = 10
MAX_CELLS = 2000
DEFAULT_CELL_LIMIT = 200


def endpoint_path() -> Path:
    override = os.environ.get("VIEW_SIMULATE_ENDPOINT_FILE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "view_simulate" / "live.json"


def read_endpoint() -> tuple[str, int]:
    direct = os.environ.get("VIEW_SIMULATE_LIVE")
    if direct:
        host, _, port = direct.rpartition(":")
        return host or HOST, int(port)
    path = endpoint_path()
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data.get("host") or HOST), int(data["port"])
    return HOST, DEFAULT_PORT


def occlusion_summary(
    results,
    *,
    stale: bool,
    status: str,
    sensor_names: list[str],
    requested_cell_size: float,
    slice_height: float,
    show_coverage: bool,
    show_hits: bool,
    only_current: bool,
) -> dict:
    """Describe the occlusion numbers that match the current picture."""
    requested_cell_size = float(requested_cell_size)
    slice_height = float(slice_height)
    summary = {
        "stale": bool(stale),
        "status": status,
        "sensors": list(sensor_names),
        "only_current": bool(only_current),
        "show_coverage": bool(show_coverage),
        "show_hits": bool(show_hits),
        "requested_cell_size_m": requested_cell_size,
        "used_cell_size_m": 0.0,
        "slice_height_m": slice_height,
        "green": 0,
        "red": 0,
        "drawn": 0,
        "echo_hits": 0,
        "echo_rays": 0,
        "per_sensor": [],
        "notes": [],
    }
    if stale:
        summary["notes"].append(
            "参数已经变化，画面上的遮挡着色和占用网格已清掉。绿格和红格按当前画面计为 0，不是上一次的结果。需要在程序里重新点「计算遮挡」。"
        )
        return summary
    if not results:
        if not (show_coverage or show_hits):
            summary["notes"].append("没有勾选「遮挡着色」或「传感器回波」，所以没有遮挡结果。")
        return summary

    per_sensor = []
    echo_hits = 0
    echo_rays = 0
    for result, name in zip(results, sensor_names):
        labels = np.asarray(getattr(result, "coverage_labels", []), dtype=int)
        hits = getattr(result, "hits", ())
        echo_hits += int(len(hits))
        echo_rays += int(getattr(result, "n_rays", 0))
        per_sensor.append(
            {
                "name": name,
                "surface_visible": int(np.sum(labels == 2)),
                "surface_blocked": int(np.sum(labels == 1)),
                "surface_outside": int(np.sum(labels == 0)),
                "echo_hits": int(len(hits)),
                "echo_rays": int(getattr(result, "n_rays", 0)),
            }
        )
    summary["per_sensor"] = per_sensor
    summary["echo_hits"] = echo_hits
    summary["echo_rays"] = echo_rays

    voxels = np.zeros(0, dtype=int)
    used = 0.0
    if show_coverage and results[0].voxel_labels is not None:
        voxels = np.asarray(results[0].voxel_labels, dtype=int)
        used = float(results[0].voxel_size or 0.0)
    green = int(np.sum(voxels == 2))
    red = int(np.sum(voxels == 1))
    summary["green"] = green
    summary["red"] = red
    summary["drawn"] = green + red
    summary["used_cell_size_m"] = used

    if show_coverage and slice_height <= 1e-6:
        summary["notes"].append("最高高度为 0，没有占用网格，绿格和红格都是 0。")
    elif show_coverage and used > requested_cell_size + 1e-6:
        summary["notes"].append(
            f"格子数量超过上限，实际边长从 {requested_cell_size:g} m 加大到 {used:g} m。画面上的立方体用的是加大后的边长。"
        )
    if show_coverage and len(sensor_names) > 1:
        summary["notes"].append("多只传感器合并计数：任一传感器能看见的格子算绿色，只有全部被挡住才算红色。")
    if show_coverage and summary["drawn"]:
        summary["notes"].append(
            "绿色格子是视场里没有被挡住的位置，包含空气，不只是打到表面上的点。"
        )
    return summary


def select_cells(centers, labels, minimum, maximum, limit: int) -> dict:
    """Return drawn occupancy cells inside an axis-aligned box."""
    limit = int(limit)
    if limit < 1 or limit > MAX_CELLS:
        raise ValueError(f"limit 要在 1 到 {MAX_CELLS} 之间")
    if minimum is None or maximum is None:
        raise ValueError("需要 min 和 max，各是三个米制坐标")
    low = np.asarray(minimum, dtype=float).reshape(3)
    high = np.asarray(maximum, dtype=float).reshape(3)
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("范围要是有限的米制坐标")
    low, high = np.minimum(low, high), np.maximum(low, high)
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if len(centers) != len(labels):
        raise ValueError("格子坐标和标签数量不一致")
    if len(centers) == 0:
        return {"total": 0, "truncated": False, "limit": limit, "cells": []}
    inside = np.all((centers >= low) & (centers <= high), axis=1)
    index = np.flatnonzero(inside)
    chosen = index[:limit]
    cells = []
    for item in chosen:
        label = int(labels[item])
        cells.append(
            {
                "xyz": [float(value) for value in centers[item]],
                "label": label,
                "color": "green" if label == 2 else "red" if label == 1 else "other",
            }
        )
    return {
        "total": int(len(index)),
        "truncated": bool(len(index) > limit),
        "limit": limit,
        "cells": cells,
    }


class LiveServer:
    """Serve one JSON request at a time. ``handler`` runs the caller's threading rules."""

    def __init__(self, handler, host: str = HOST, port: int = DEFAULT_PORT):
        self.handler = handler
        self.host = host
        self.port = int(port)
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._endpoint_written = False

    def start(self) -> None:
        if self._thread is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        first = self.port
        bound = False
        for offset in range(PORT_SPAN if first else 1):
            candidate = first + offset
            try:
                sock.bind((self.host, candidate))
            except OSError:
                continue
            bound = True
            break
        if not bound:
            sock.close()
            raise OSError(f"端口 {first} 到 {first + PORT_SPAN - 1} 都已被占用")
        sock.listen(4)
        sock.settimeout(0.5)
        self._socket = sock
        self.port = int(sock.getsockname()[1])
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="view-simulate-live", daemon=True)
        self._thread.start()

    def write_endpoint(self) -> None:
        path = endpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"host": self.host, "port": self.port, "pid": os.getpid()}
        path.write_text(json.dumps(payload), encoding="utf-8")
        self._endpoint_written = True

    def stop(self) -> None:
        self._stop.set()
        sock = self._socket
        self._socket = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        self._thread = None
        if self._endpoint_written:
            path = endpoint_path()
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            if int(data.get("pid") or 0) == os.getpid() and int(data.get("port") or 0) == self.port:
                path.unlink(missing_ok=True)
            self._endpoint_written = False

    def _serve(self) -> None:
        sock = self._socket
        if sock is None:
            return
        while not self._stop.is_set():
            try:
                connection, _address = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._handle(connection)
            finally:
                try:
                    connection.close()
                except OSError:
                    pass

    def _handle(self, connection: socket.socket) -> None:
        connection.settimeout(120)
        try:
            raw = _read_line(connection, limit=65536)
            request = json.loads(raw)
            if not isinstance(request, dict):
                response = {"ok": False, "error": "请求必须是 JSON 对象"}
            else:
                response = self.handler(request)
                if not isinstance(response, dict):
                    response = {"ok": False, "error": "内部返回不是 JSON 对象"}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        payload = json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n"
        try:
            connection.sendall(payload)
        except OSError:
            pass


def query(method: str, params: dict | None = None, timeout: float = 120) -> dict:
    """Ask the running window. Raises RuntimeError when it is not available."""
    host, port = read_endpoint()
    request = {"method": method}
    if params:
        request.update(params)
    try:
        connection = socket.create_connection((host, port), timeout=3)
    except OSError as exc:
        raise RuntimeError(
            "视场程序没有在运行，或实时端口没打开。先在这个项目里执行 python3 -m view_simulate。"
        ) from exc
    try:
        connection.settimeout(timeout)
        connection.sendall(json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n")
        raw = _read_line(connection, limit=8_000_000)
    except socket.timeout as exc:
        raise RuntimeError("视场程序 120 秒内没有返回。它可能正在计算遮挡。") from exc
    finally:
        connection.close()
    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"视场程序返回的不是 JSON：{raw[:200]}") from exc
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "视场程序拒绝了这次请求"))
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("视场程序没有返回结果对象")
    return result


def _read_line(connection: socket.socket, limit: int) -> str:
    chunks = []
    size = 0
    while True:
        chunk = connection.recv(4096)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise ValueError("请求或回复超过长度限制")
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    raw = b"".join(chunks)
    if not raw.strip():
        raise ValueError("空请求")
    line = raw.split(b"\n", 1)[0]
    return line.decode("utf-8")
