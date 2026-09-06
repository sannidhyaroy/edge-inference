"""Latency measurement.

Every latency number this project reports comes from here, so the methodology
is fixed in one place rather than reinvented per experiment:

* **Batch size one.** Edge inference handles a single input at a time. Larger
  batches would report throughput, which is a different and easier problem.

* **Warmup runs are discarded.** The first few inferences pay one-off costs:
  the memory allocator grows its pools, the backend picks kernels for the
  shapes it has just seen, and caches are cold. Including them would inflate
  the result by an amount that has nothing to do with steady-state latency.

* **Median and p95 are both reported.** The median is the typical case and
  shrugs off a scheduler hiccup. The p95 is the tail, which is what matters
  when inference has a deadline to meet. A mean on its own would hide both.

* **Thread count is pinned and recorded on every row.** This is what lets a
  laptop stand in for a constrained device, and it makes it impossible to
  compare two results from different thread budgets by accident.

* **The CPU and host are recorded too.** Latency is a property of the model and
  the hardware together, so a row without its hardware is not interpretable.
"""

import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from edge_inference.config import IMAGE_SIZE, SEED


def cpu_model() -> str:
    """Return a human readable CPU name, best effort, on any platform."""
    system = platform.system()
    if system == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
        except OSError, subprocess.SubprocessError:
            pass
        else:
            return result.stdout.strip()
    # platform.processor() is descriptive on Windows and vague elsewhere, so
    # fall back to the architecture string rather than returning nothing.
    return platform.processor() or platform.machine()


def machine_info() -> dict[str, str]:
    """Describe the machine well enough to interpret a latency row later."""
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "arch": platform.machine(),
        "cpu": cpu_model(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }


def set_thread_count(threads: int) -> int:
    """Pin the intra-op thread count and report what actually took effect.

    PyTorch parallelises a single operation across a pool of threads. Capping
    that pool is how a twelve thread laptop imitates a device with one or two
    cores.

    The return value is read back from PyTorch rather than echoed, because the
    request is not always honoured: the pool size can be fixed by the time the
    first operation has run, and an environment variable such as
    OMP_NUM_THREADS takes precedence. Recording the effective value keeps a
    silently ignored request from being written into results as though it
    worked.
    """
    torch.set_num_threads(threads)
    return torch.get_num_threads()


def synchronize(device: str) -> None:
    """Block until queued work on `device` has actually finished.

    GPU backends are asynchronous: the Python call returns once the work is
    queued, not once it is done. Timing without a synchronisation point would
    measure how fast work can be enqueued, which is meaningless. CPU execution
    is synchronous already, so this is a no-op there.
    """
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def time_callable(fn: Callable[[], object], *, warmup: int, runs: int) -> list[float]:
    """Call `fn` repeatedly and return the timed durations in milliseconds."""
    for _ in range(warmup):
        fn()

    times_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return times_ms


def summarise(times_ms: Sequence[float]) -> dict[str, float]:
    """Reduce raw durations to the statistics reported in results tables."""
    values = np.asarray(times_ms, dtype=np.float64)
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(values.mean()),
        # ddof=1 gives the sample standard deviation, which is the right
        # estimator when these runs are a sample of possible runs.
        "std_ms": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def benchmark_model(
    model: torch.nn.Module,
    *,
    threads: int,
    warmup: int,
    runs: int,
    batch_size: int = 1,
    image_size: int = IMAGE_SIZE,
    device: str = "cpu",
) -> dict[str, object]:
    """Time a single forward pass and return one fully labelled result row.

    The input is random noise rather than a real photograph. Convolution cost
    depends on tensor shapes, not pixel values, so this measures exactly what a
    real image would while keeping the benchmark independent of the dataset. It
    is generated from a fixed seed so every run sees identical input.
    """
    effective_threads = set_thread_count(threads)

    model = model.to(device).eval()

    generator = torch.Generator().manual_seed(SEED)
    inputs = torch.randn(batch_size, 3, image_size, image_size, generator=generator).to(device)

    def run_once() -> None:
        # inference_mode is a stricter no-grad: it skips building the autograd
        # graph entirely, which is both faster and closer to how a deployed
        # model runs.
        with torch.inference_mode():
            model(inputs)
        synchronize(device)

    times_ms = time_callable(run_once, warmup=warmup, runs=runs)

    return {
        **machine_info(),
        "device": device,
        "threads_requested": threads,
        "threads_effective": effective_threads,
        "batch_size": batch_size,
        "image_size": image_size,
        "warmup": warmup,
        "runs": runs,
        **summarise(times_ms),
    }
