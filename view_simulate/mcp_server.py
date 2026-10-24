"""MCP stdio server that reads the running view window.

Cursor starts this process. It does not draw anything; it asks the already
open program on 127.0.0.1 for the live parameters and occlusion result.
"""

from __future__ import annotations

import json
import sys

from view_simulate import __version__
from view_simulate.live import DEFAULT_CELL_LIMIT, query

PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")

TOOLS = (
    {
        "name": "get_live_state",
        "description": (
            "读取正在运行的视场程序的当前参数和遮挡计算。"
            "包含关节角度、传感器布置、显示选项、状态栏原文，以及绿色和红色占用网格个数。"
            "stale 为 true 时，画面上的遮挡已经清掉，绿格和红格是 0，需要用户重新点「计算遮挡」。"
            "绿色格子包含视场里的空气，不只是表面上的点。多只传感器一起算时，任一只能看见就算绿色。"
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_occlusion_cells",
        "description": (
            "读取当前画面上、落在给定长方体内的占用网格中心。"
            "坐标相对机器人根连杆，单位米。只返回已经画出来的绿格和红格。"
            "结果过期或最高高度为 0 时格子为空，并在 notes 里说明原因。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "min_x": {"type": "number"},
                "min_y": {"type": "number"},
                "min_z": {"type": "number"},
                "max_x": {"type": "number"},
                "max_y": {"type": "number"},
                "max_z": {"type": "number"},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 2000,
                    "description": f"最多返回多少个格子，默认 {DEFAULT_CELL_LIMIT}",
                },
            },
            "required": ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z"],
            "additionalProperties": False,
        },
    },
)


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle(message)
        if response is None:
            continue
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def handle(message: dict) -> dict | None:
    method = message.get("method")
    if not isinstance(method, str):
        return None
    message_id = message.get("id")
    if method.startswith("notifications/"):
        return None
    if message_id is None:
        return None
    try:
        result = _dispatch(method, message.get("params") or {})
    except Exception as exc:
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "error": {"code": -32603, "message": str(exc)},
        }
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _dispatch(method: str, params: dict) -> dict:
    if method == "initialize":
        requested = str(params.get("protocolVersion") or "")
        version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[-1]
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "view-simulate", "version": __version__},
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": list(TOOLS)}
    if method == "tools/call":
        return _call_tool(str(params.get("name") or ""), params.get("arguments") or {})
    raise RuntimeError(f"不支持的方法 {method}")


def _call_tool(name: str, arguments: dict) -> dict:
    try:
        if name == "get_live_state":
            payload = query("get_live_state")
        elif name == "get_occlusion_cells":
            payload = query(
                "get_occlusion_cells",
                {
                    "min": [arguments["min_x"], arguments["min_y"], arguments["min_z"]],
                    "max": [arguments["max_x"], arguments["max_y"], arguments["max_z"]],
                    "limit": int(arguments.get("limit") or DEFAULT_CELL_LIMIT),
                },
            )
        else:
            raise RuntimeError(f"未知工具 {name}")
    except Exception as exc:
        return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    return {"content": [{"type": "text", "text": text}], "isError": False}


if __name__ == "__main__":
    main()
