"""CUDA ray-mesh distances. Falls back to None when the GPU cannot be used."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

_lib = None
_unavailable = False
_device = "CPU"


def device_label() -> str:
    return _device


def mesh_distances(origins, directions, v0, e1, e2, clearance: float):
    """Closest hit distance per ray. None means the caller should use the CPU path."""
    global _device
    lib = _library()
    if lib is None:
        return None
    origins = np.ascontiguousarray(origins, dtype=np.float32)
    directions = np.ascontiguousarray(directions, dtype=np.float32)
    if len(origins) == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.empty(len(origins), dtype=np.float32)
    code = lib.ray_mesh_distances(
        _ptr(origins),
        _ptr(directions),
        _ptr(v0),
        _ptr(e1),
        _ptr(e2),
        int(len(origins)),
        int(len(v0)),
        ctypes.c_float(float(clearance)),
        _ptr(out),
    )
    if code != 0:
        _device = "CPU"
        return None
    _device = _read_name(lib)
    distances = out.astype(np.float64)
    distances[~np.isfinite(distances)] = np.inf
    return distances


def _library():
    global _lib, _unavailable
    if _unavailable:
        return None
    if _lib is not None:
        return _lib
    try:
        binary = _compile()
        lib = _load(binary)
    except (OSError, subprocess.CalledProcessError, FileNotFoundError):
        _unavailable = True
        return None
    lib.ray_mesh_distances.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ray_mesh_distances.restype = ctypes.c_int
    lib.gpu_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.gpu_device_name.restype = ctypes.c_int
    _lib = lib
    return lib


def _compile() -> Path:
    source = Path(__file__).with_name("ray_mesh.cu")
    binary = source.with_name("_raymesh.dll" if os.name == "nt" else "_raymesh.so")
    if binary.is_file() and binary.stat().st_mtime >= source.stat().st_mtime:
        return binary
    nvcc = _nvcc()
    if os.name == "nt":
        command = [nvcc, "-O3", "-shared", "-o", str(binary), str(source)]
    else:
        command = [nvcc, "-O3", "-shared", "-Xcompiler", "-fPIC", "-o", str(binary), str(source)]
    subprocess.check_call(command, stdout=subprocess.DEVNULL)
    return binary


def _nvcc() -> str:
    env = os.environ.get("CUDA_NVCC")
    if env:
        if Path(env).is_file():
            return env
        raise FileNotFoundError(env)
    found = shutil.which("nvcc")
    if found:
        return found
    if os.name == "nt":
        cuda_path = os.environ.get("CUDA_PATH")
        candidates = []
        if cuda_path:
            candidates.append(Path(cuda_path) / "bin" / "nvcc.exe")
        toolkit = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if toolkit.is_dir():
            candidates.extend(
                path / "bin" / "nvcc.exe" for path in sorted(toolkit.glob("v*"), reverse=True)
            )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError("nvcc.exe")
    default = "/usr/local/cuda/bin/nvcc"
    if Path(default).is_file():
        return default
    raise FileNotFoundError(default)


def _load(binary: Path):
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for directory in _cuda_bin_dirs():
            os.add_dll_directory(str(directory))
    return ctypes.CDLL(str(binary))


def _cuda_bin_dirs():
    found = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        found.append(Path(cuda_path) / "bin")
    nvcc = os.environ.get("CUDA_NVCC") or shutil.which("nvcc")
    if nvcc:
        found.append(Path(nvcc).resolve().parent)
    toolkit = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit.is_dir():
        found.extend(path / "bin" for path in sorted(toolkit.glob("v*"), reverse=True))
    return [path for path in found if path.is_dir()]


def _read_name(lib) -> str:
    buffer = ctypes.create_string_buffer(128)
    if lib.gpu_device_name(buffer, len(buffer)) != 0:
        return "GPU"
    name = buffer.value.decode("utf-8", errors="ignore").strip()
    return name or "GPU"


def _ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
