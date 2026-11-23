"""Loads every ONNX model in a folder on CUDA with one set of CUDA options, then runs them in turn.

Usage: python onnx_arena_bench.py <models-dir> <today|arena|workspace|both> [seconds]
Prints LOADED when all sessions exist, RUN_DONE after the timed run, then holds for 15 s
so the GPU memory can be read from outside, and exits.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

CONFIGS = {
    "today": {"cudnn_conv_use_max_workspace": "1"},
    "arena": {"cudnn_conv_use_max_workspace": "1", "arena_extend_strategy": "kSameAsRequested"},
    "workspace": {"cudnn_conv_use_max_workspace": "0"},
    "both": {"cudnn_conv_use_max_workspace": "0", "arena_extend_strategy": "kSameAsRequested"},
}

DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(uint8)": np.uint8,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def feed_for(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    feeds = {}
    for inp in session.get_inputs():
        dtype = DTYPES.get(inp.type, np.float32)
        shape = []
        for i, dim in enumerate(inp.shape):
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            elif i == 0:
                shape.append(1)
            else:
                shape.append(77 if dtype in (np.int64, np.int32) else 256)
        if dtype in (np.int64, np.int32):
            feeds[inp.name] = np.ones(shape, dtype=dtype)
        elif dtype == np.uint8:
            feeds[inp.name] = np.random.randint(0, 255, shape, dtype=np.uint8)
        else:
            feeds[inp.name] = np.random.rand(*shape).astype(dtype)
    return feeds


def main() -> None:
    models_dir, config = Path(sys.argv[1]), sys.argv[2]
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:  # noqa: BLE001
            print(f"preload_dlls: {e}")
    cuda = {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC", **CONFIGS[config]}
    print(f"PID {os.getpid()} config={config} options={cuda} ort={ort.__version__}", flush=True)

    sessions = []
    for path in sorted(models_dir.rglob("*.onnx")):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        s = ort.InferenceSession(str(path), so, providers=[("CUDAExecutionProvider", cuda), "CPUExecutionProvider"])
        if s.get_providers()[0] != "CUDAExecutionProvider":
            sys.exit(f"ABORT {path.name} did not load on CUDA: {s.get_providers()}")
        feeds = feed_for(s)
        try:
            s.run(None, feeds)
        except Exception as e:  # noqa: BLE001
            print(f"skip {path.name}: {e}", flush=True)
            continue
        sessions.append((path.name, s, feeds))
    print(f"LOADED {len(sessions)} models", flush=True)
    time.sleep(10)

    times: dict[str, list[float]] = {name: [] for name, _, _ in sessions}
    end = time.perf_counter() + seconds
    runs = 0
    while time.perf_counter() < end:
        for name, s, feeds in sessions:
            t = time.perf_counter()
            s.run(None, feeds)
            times[name].append((time.perf_counter() - t) * 1000)
            runs += 1
    for name, ts in times.items():
        ts = ts[5:] or ts
        print(f"LAT {name} mean={np.mean(ts):.2f}ms p50={np.percentile(ts, 50):.2f}ms n={len(ts)}", flush=True)
    print(f"THROUGHPUT {runs / seconds:.1f} runs/s", flush=True)
    print("RUN_DONE", flush=True)
    time.sleep(15)


if __name__ == "__main__":
    main()
