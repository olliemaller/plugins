from __future__ import annotations

import ctypes
import platform

_CUDA_SUCCESS = 0


def _load_driver() -> ctypes.CDLL | None:
    names = ["nvcuda.dll"] if platform.system() == "Windows" else ["libcuda.so.1", "libcuda.so"]
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def gpu_total_memory(device_id: int) -> int | None:
    """Total memory of a CUDA device in bytes, read from the driver API. None when it can't be read."""
    driver = _load_driver()
    if driver is None:
        return None
    try:
        if driver.cuInit(0) != _CUDA_SUCCESS:
            return None
        device = ctypes.c_int()
        if driver.cuDeviceGet(ctypes.byref(device), device_id) != _CUDA_SUCCESS:
            return None
        total = ctypes.c_size_t()
        if driver.cuDeviceTotalMem_v2(ctypes.byref(total), device) != _CUDA_SUCCESS:
            return None
        return int(total.value)
    except (AttributeError, OSError):
        return None
