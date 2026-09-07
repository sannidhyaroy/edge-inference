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

* **The hardware is recorded too.** Latency is a property of the model and the
  machine together, so a row without its hardware is not interpretable. Rows
  carry the machine model, CPU, and physical and logical core counts, which is
  what a results table or plot legend needs. The hostname is deliberately left
  out: it identifies the operator rather than the hardware.
"""

import os
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from edge_inference.config import IMAGE_SIZE, SEED


def _sysctl(key: str) -> str | None:
    """Read one macOS sysctl value, or None if it is unavailable."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError, subprocess.SubprocessError:
        return None
    return result.stdout.strip() or None


def _powershell(expression: str) -> str | None:
    """Evaluate one PowerShell expression on Windows, or None if unavailable.

    Windows exposes no /proc or sysctl, so hardware details come from CIM. This
    is the same best-effort shape as the macOS sysctl path: never raise, just
    report nothing when the information cannot be had.
    """
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", expression],
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError, subprocess.SubprocessError:
        return None
    return result.stdout.strip() or None


def _proc_field(path: Path, prefix: str, separator: str = ":") -> str | None:
    """Return the value of the first line in `path` starting with `prefix`."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith(prefix):
            return line.split(separator, 1)[1].strip()
    return None


def cpu_model() -> str:
    """Return a human readable CPU name, best effort, on any platform."""
    system = platform.system()
    if system == "Linux":
        name = _proc_field(Path("/proc/cpuinfo"), "model name")
        if name:
            return name
    elif system == "Darwin":
        name = _sysctl("machdep.cpu.brand_string")
        if name:
            return name
    # platform.processor() is descriptive on Windows and vague elsewhere, so
    # fall back to the architecture string rather than returning nothing.
    return platform.processor() or platform.machine()


def machine_model() -> str:
    """Return the hardware model, for labelling result tables and plots.

    A hostname identifies a machine but describes nothing. "HP ProBook 445 G10"
    tells a reader what produced a number, which is what a chart legend or a
    results table actually needs.
    """
    system = platform.system()
    if system == "Linux":
        # DMI is the firmware's own description of the machine, exposed by the
        # kernel. Some systems report placeholders here, hence the filter.
        product = Path("/sys/devices/virtual/dmi/id/product_name")
        if product.exists():
            name = product.read_text().strip()
            if name and "unknown" not in name.lower() and "o.e.m." not in name.lower():
                return name
    elif system == "Darwin":
        # Returns an identifier such as MacBookPro17,1 rather than a marketing
        # name, which is still far more useful than a hostname.
        model = _sysctl("hw.model")
        if model:
            return model
    elif system == "Windows":
        model = _powershell("(Get-CimInstance Win32_ComputerSystem).Model")
        if model:
            return model
    return platform.machine()


def physical_cores() -> int | None:
    """Count physical cores, which is not the same as logical threads.

    The distinction is central to these measurements: on a chip with SMT, the
    logical count is double the physical one, and oversubscribing past the
    physical cores changes latency behaviour markedly.
    """
    system = platform.system()
    if system == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            # A core is identified by (socket, core) together, since core ids
            # restart from zero on each socket.
            cores: set[tuple[str, str]] = set()
            socket_id: str | None = None
            for line in cpuinfo.read_text().splitlines():
                if line.startswith("physical id"):
                    socket_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id") and socket_id is not None:
                    cores.add((socket_id, line.split(":", 1)[1].strip()))
            if cores:
                return len(cores)
    elif system == "Darwin":
        value = _sysctl("hw.physicalcpu")
        if value and value.isdigit():
            return int(value)
    elif system == "Windows":
        # Sums across sockets, so it stays correct on a multi-socket machine.
        value = _powershell(
            "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"
        )
        if value and value.isdigit():
            return int(value)
    return None


def machine_info() -> dict[str, object]:
    """Describe the machine well enough to interpret a latency row later.

    `machine` is the field to label plots and result tables with: it says what
    produced a number, which a hostname does not. The hostname is deliberately
    not recorded, since the model, CPU, and platform already identify a machine
    and it would only add a personal detail to a public results file.
    """
    return {
        "machine": machine_model(),
        "cpu": cpu_model(),
        "cores_physical": physical_cores(),
        "cores_logical": os.cpu_count(),
        "system": platform.system(),
        "arch": platform.machine(),
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


def resolve_device(name: str) -> str:
    """Validate a device string, failing loudly rather than silently downgrading.

    A silent fallback to CPU would be the worst outcome here: training would
    still finish, just far slower, and nothing would say why. Equally, a
    benchmark that quietly ran on a GPU would produce numbers that look like
    edge latency and are not.
    """
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "cuda requested but torch reports no CUDA device. On a GPU machine, "
            "check the environment was installed with `uv sync --extra cuda`."
        )
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("mps requested but this build of torch has no MPS backend")
    return name


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


def benchmark_session(
    session,
    *,
    threads: int,
    warmup: int,
    runs: int,
    batch_size: int = 1,
    image_size: int = IMAGE_SIZE,
) -> dict[str, object]:
    """Time an ONNX Runtime session using the same protocol as the torch path.

    Deliberately mirrors `benchmark_model`: same warmup, same statistics, same
    fixed input. A float32 model measured one way and a quantized model
    measured another would produce a speedup figure that describes the two
    harnesses rather than the two models.

    The session's own thread count is set when it is built, since ONNX Runtime
    fixes it at session creation rather than per call. It is recorded here so
    the row carries the setting it was measured under.
    """
    generator = torch.Generator().manual_seed(SEED)
    inputs = torch.randn(batch_size, 3, image_size, image_size, generator=generator).numpy()
    input_name = session.get_inputs()[0].name

    def run_once() -> None:
        session.run(None, {input_name: inputs})

    times_ms = time_callable(run_once, warmup=warmup, runs=runs)

    return {
        **machine_info(),
        "device": "cpu",
        "runtime": "onnxruntime",
        "threads_requested": threads,
        "threads_effective": threads,
        "batch_size": batch_size,
        "image_size": image_size,
        "warmup": warmup,
        "runs": runs,
        **summarise(times_ms),
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
