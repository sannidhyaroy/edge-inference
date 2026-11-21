"""ONNX Runtime sessions, and running a graph: everything a device needs, without torch.

A device only runs exported graphs. Keeping this apart from the export code,
which needs torch, lets a small board run the device side with nothing but
NumPy and ONNX Runtime installed.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Tensor names in every exported graph.
INPUT_NAME = "input"
OUTPUT_NAME = "logits"
# A stage graph with an exit after it also hands on its feature map.
FEATURES_NAME = "features"


def run_stage(session: ort.InferenceSession, x: np.ndarray) -> dict[str, np.ndarray]:
    """Run one stage graph and return its outputs by name."""
    names = [output.name for output in session.get_outputs()]
    return dict(zip(names, session.run(names, {INPUT_NAME: x}), strict=True))


def build_session(path: Path, *, threads: int = 1) -> ort.InferenceSession:
    """Open an ONNX Runtime session with the thread count pinned.

    Thread count is set explicitly for the same reason it is in the PyTorch
    harness: it is how a laptop stands in for a constrained device, and a
    default that varies by machine would make results incomparable.

    inter_op is fixed at one because it parallelises across independent
    branches of a graph, and a sequential backbone has none. Leaving it free
    would add scheduling overhead for no benefit.

    Once `build_stage_sessions` has created the process's shared pool, ONNX
    Runtime refuses sessions with a pool of their own, so this joins the shared
    pool instead.
    """
    if _shared_pool_threads is not None:
        return build_stage_sessions([path], threads=threads)[0]

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1

    return ort.InferenceSession(
        str(path),
        options,
        providers=["CPUExecutionProvider"],
    )


# Size of this process's shared ONNX Runtime thread pool, once it is set.
_shared_pool_threads: int | None = None


def build_stage_sessions(paths: Sequence[Path], *, threads: int) -> list[ort.InferenceSession]:
    """Open chained stage graphs on one thread pool shared between them.

    Use this, not `build_session`, whenever chained stages are timed. Each
    session normally owns its own pool, and a pool's threads keep spinning for
    a moment after a run in case more work arrives. Chained, the next stage
    then competes with the previous stage's spinning threads. Measured on the
    reference laptop, the three stages took 10.40 ms at 2 threads against
    8.62 ms for the same network as one graph, and 21.8 ms against 4.56 ms at 6
    threads. On one shared pool they run within 3% of the single graph at
    both.

    The shared pool belongs to the process and its size can be set once, so a
    process times its stage graphs at one thread count. Asking for another
    raises rather than silently running at the first.
    """
    global _shared_pool_threads
    if _shared_pool_threads is None:
        # Not re-exported by onnxruntime's public module, but the only way to
        # size the shared pool from Python.
        from onnxruntime.capi._pybind_state import set_global_thread_pool_sizes

        set_global_thread_pool_sizes(threads, 1)
        _shared_pool_threads = threads
    elif _shared_pool_threads != threads:
        raise RuntimeError(
            f"this process's shared thread pool already has {_shared_pool_threads} "
            f"thread(s); stage graphs at {threads} need a separate process"
        )

    sessions = []
    for path in paths:
        options = ort.SessionOptions()
        options.use_per_session_threads = False
        sessions.append(
            ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        )
    return sessions
