"""Metal ray-mesh distances. Returns None when this Mac cannot use the GPU."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

_session = None
_unavailable = False
_device_name = "GPU"
_UNIFORM_DTYPE = np.dtype(
    [("n_rays", "<i4"), ("n_faces", "<i4"), ("clearance", "<f4"), ("pad", "<f4")]
)


def device_name() -> str:
    return _device_name


def compute(origins, directions, v0, e1, e2, clearance: float):
    """Float32 hit distance per ray. None means the caller should use the CPU path."""
    session = _engine()
    if session is None:
        return None
    try:
        return session.distances(origins, directions, v0, e1, e2, clearance)
    except (RuntimeError, MemoryError, TypeError, ctypes.ArgumentError) as exc:
        print(f"Metal 射线求交失败，改用 CPU：{exc}", file=sys.stderr)
        return None


def _engine():
    global _session, _unavailable, _device_name
    if sys.platform != "darwin" or _unavailable:
        return None
    if _session is not None:
        return _session
    try:
        session = _MetalRays()
    except Exception as exc:
        _unavailable = True
        print(f"Metal 射线求交不可用，改用 CPU：{exc}", file=sys.stderr)
        return None
    _session = session
    _device_name = session.name
    return session


class _MetalRays:
    def __init__(self):
        import Metal

        self.Metal = Metal
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            raise RuntimeError("没有可用的 Metal 设备")
        source = Path(__file__).with_name("ray_mesh.metal").read_text(encoding="utf-8")
        compiled = device.newLibraryWithSource_options_error_(source, None, None)
        library, error = compiled if isinstance(compiled, tuple) else (compiled, None)
        if library is None:
            raise RuntimeError(error or "Metal 着色器编译失败")
        function = library.newFunctionWithName_("ray_kernel")
        if function is None:
            raise RuntimeError("着色器里没有 ray_kernel")
        pipeline = device.newComputePipelineStateWithFunction_error_(function, None)
        state, error = pipeline if isinstance(pipeline, tuple) else (pipeline, None)
        if state is None:
            raise RuntimeError(error or "Metal 计算管线创建失败")
        self.device = device
        self.pipeline = state
        self.queue = device.newCommandQueue()
        if self.queue is None:
            raise RuntimeError("Metal 命令队列创建失败")
        self.name = str(device.name() or "GPU").strip() or "GPU"
        self._face_key = None
        self._faces = None
        width = max(int(state.threadExecutionWidth()), 1)
        limit = max(int(state.maxTotalThreadsPerThreadgroup()), width)
        threads = min(256, limit)
        self._threads = max(width, threads - (threads % width))

    def distances(self, origins, directions, v0, e1, e2, clearance: float):
        origins = np.ascontiguousarray(origins, dtype=np.float32)
        directions = np.ascontiguousarray(directions, dtype=np.float32)
        if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[1] != 3:
            raise RuntimeError("射线数组形状应为 (N, 3)")
        n_rays = int(origins.shape[0])
        if n_rays == 0:
            return np.zeros(0, dtype=np.float32)
        faces = self._face_buffers(v0, e1, e2)
        ray_origins = self._shared(origins)
        ray_dirs = self._shared(directions)
        out = self._alloc(n_rays * 4)
        uniforms = np.array(
            [(n_rays, int(faces[0]), np.float32(clearance), np.float32(0.0))],
            dtype=_UNIFORM_DTYPE,
        )
        uniform_buffer = self._shared(uniforms, raw=True)
        command = self.queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipeline)
        encoder.setBuffer_offset_atIndex_(ray_origins, 0, 0)
        encoder.setBuffer_offset_atIndex_(ray_dirs, 0, 1)
        encoder.setBuffer_offset_atIndex_(faces[1], 0, 2)
        encoder.setBuffer_offset_atIndex_(faces[2], 0, 3)
        encoder.setBuffer_offset_atIndex_(faces[3], 0, 4)
        encoder.setBuffer_offset_atIndex_(uniform_buffer, 0, 5)
        encoder.setBuffer_offset_atIndex_(out, 0, 6)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self.Metal.MTLSizeMake(n_rays, 1, 1),
            self.Metal.MTLSizeMake(self._threads, 1, 1),
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if command.status() != self.Metal.MTLCommandBufferStatusCompleted:
            raise RuntimeError(command.error() or "Metal 计算失败")
        return _read_floats(out, n_rays)

    def _face_buffers(self, v0, e1, e2):
        v0 = np.ascontiguousarray(v0, dtype=np.float32)
        e1 = np.ascontiguousarray(e1, dtype=np.float32)
        e2 = np.ascontiguousarray(e2, dtype=np.float32)
        if v0.shape != e1.shape or v0.shape != e2.shape:
            raise RuntimeError("三角面数组形状不一致")
        key = (v0.ctypes.data, e1.ctypes.data, e2.ctypes.data, v0.shape, v0.nbytes)
        if key == self._face_key and self._faces is not None:
            return self._faces
        n_faces = int(v0.shape[0]) if v0.ndim == 2 else int(v0.size // 3)
        self._faces = (n_faces, self._shared(v0), self._shared(e1), self._shared(e2))
        self._face_key = key
        return self._faces

    def _shared(self, array: np.ndarray, *, raw: bool = False):
        payload = np.ascontiguousarray(array) if raw else np.ascontiguousarray(array, dtype=np.float32)
        buffer = self._alloc(payload.nbytes)
        _write_buffer(buffer, payload)
        return buffer

    def _alloc(self, nbytes: int):
        size = max(16, int(nbytes))
        limit = int(self.device.maxBufferLength())
        if size > limit:
            raise MemoryError(f"Metal 缓冲 {size} 字节超过设备上限 {limit}")
        buffer = self.device.newBufferWithLength_options_(size, self.Metal.MTLResourceStorageModeShared)
        if buffer is None:
            raise MemoryError(f"Metal 缓冲分配失败（{size} 字节）")
        return buffer


def _write_buffer(buffer, array: np.ndarray) -> None:
    payload = np.ascontiguousarray(array)
    nbytes = int(payload.nbytes)
    if nbytes == 0:
        return
    contents = buffer.contents()
    try:
        ctypes.memmove(contents, payload.ctypes.data, nbytes)
        return
    except (TypeError, ctypes.ArgumentError):
        view = _contents_view(contents, nbytes)
    target = np.frombuffer(view, dtype=payload.dtype, count=payload.size)
    target[...] = payload.reshape(-1)


def _read_floats(buffer, count: int) -> np.ndarray:
    nbytes = count * 4
    contents = buffer.contents()
    out = np.empty(count, dtype=np.float32)
    try:
        ctypes.memmove(out.ctypes.data, contents, nbytes)
        return out
    except (TypeError, ctypes.ArgumentError):
        view = _contents_view(contents, nbytes)
    return np.array(np.frombuffer(view, dtype=np.float32, count=count), copy=True)


def _contents_view(contents, nbytes: int):
    if hasattr(contents, "as_buffer"):
        return contents.as_buffer(nbytes)
    return memoryview(contents)[:nbytes]
