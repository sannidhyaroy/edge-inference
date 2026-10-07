"""ONNX export and numerical parity checking.

ONNX is an interchange format: a frozen description of a network's operations
and weights, independent of the framework that trained it. Exporting to it is
what lets a model trained in PyTorch run under a different engine.

That engine here is ONNX Runtime, chosen because it is the deployment path on
both x86 and ARM, and because its static quantization tooling is what the next
stage of this project uses. Measuring PyTorch latency and then quantizing with
some other stack would compare two different things.

**Export is not a lossless operation**, which is why nothing here is trusted
without checking. Tracing runs the model once and records the operations it
performs, so anything the graph does conditionally can be silently baked in,
and operator implementations differ slightly between engines. If the exported
graph disagrees with the original, every measurement taken afterwards describes
a model that was never evaluated for accuracy. `verify_parity` exists to make
that failure loud rather than silent.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from edge_inference.config import IMAGE_SIZE, SEED
from edge_inference.models import EXIT_NAMES, EarlyExitResNet

# What torch's dynamo exporter emits natively for these models. Requesting an
# older opset makes it export at 18 and then down-convert, which fails for this
# graph and silently leaves the file at 18 anyway. Better to name the version
# actually produced and verify it than to request one and be quietly ignored.
DEFAULT_OPSET = 18

INPUT_NAME = "input"
OUTPUT_NAME = "logits"
# What an early-exit stage hands to the next stage, or across the network when
# the model is split.
FEATURES_NAME = "features"


def export_onnx(
    model: nn.Module,
    path: Path,
    *,
    image_size: int = IMAGE_SIZE,
    opset: int = DEFAULT_OPSET,
    example: torch.Tensor | None = None,
    output_names: Sequence[str] = (OUTPUT_NAME,),
) -> Path:
    """Export `model` to ONNX at `path` and return the path.

    `example` defaults to one image. A model whose input is not an image, such
    as a later stage of an early-exit network, passes a tensor of its own input
    shape instead.

    The batch dimension is marked dynamic. Inference here is always batch size
    one, but calibrating a quantized model feeds batches through the same file,
    and a graph frozen at batch one would reject them.

    Weights are written into the file rather than beside it. torch defaults to
    external data, which splits a model into a small graph plus a `.onnx.data`
    blob. That is necessary past protobuf's 2 GB ceiling and a liability below
    it: the size measured on disk becomes the graph alone, which would turn the
    quantization size comparison into nonsense. These models are tens of
    megabytes, so a single self-contained file is both simpler and honest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()

    if example is None:
        example = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=[INPUT_NAME],
        output_names=list(output_names),
        # `dynamic_shapes` rather than the older `dynamic_axes`, which is
        # deprecated. Dim.DYNAMIC lets the exporter infer the bounds instead of
        # a named Dim, which historically defaults to a minimum of 2 and would
        # reject the batch size of one every measurement here uses.
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        opset_version=opset,
        external_data=False,
    )

    # Read the opset back rather than trusting the request. Asking for a
    # version the exporter cannot down-convert to leaves the file at whatever
    # it produced, with the failure buried in the log, and every later claim
    # about the graph's version would be wrong.
    actual = exported_opset(path)
    if actual != opset:
        raise RuntimeError(
            f"requested opset {opset} but the exported graph is opset {actual}. "
            "The exporter emits its native version and down-converts afterwards, "
            "and that conversion can fail without raising."
        )
    return path


def exported_opset(path: Path) -> int:
    """Return the default-domain opset version recorded in an ONNX file."""
    model = onnx.load(str(path), load_external_data=False)
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return entry.version
    raise ValueError(f"{path} declares no default-domain opset")


class ExitStage(nn.Module):
    """One stage of an early-exit network and the exit head after it.

    Exported as one graph per stage because a single graph cannot stop early:
    ONNX Runtime runs a graph to the end, so an exit decision has to happen
    between graphs. The application runs stage one, reads the exit's
    confidence, and only then decides whether to run stage two on the features.

    The same boundary is where split computing cuts the network, so the
    `features` output is also the tensor that would cross the network.
    """

    def __init__(self, stage: nn.Module, head: nn.Module, *, last: bool) -> None:
        super().__init__()
        self.stage = stage
        self.head = head
        self.last = last

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.stage(x)
        logits = self.head(features)
        # Nothing runs after the last stage, so its features have nowhere to go.
        if self.last:
            return logits
        return features, logits


def stage_paths(base: Path, *, suffix: str = ".onnx") -> list[Path]:
    """Where each stage graph of the model at `base` (no suffix) is written.

    `suffix` distinguishes precisions, as `.int8.onnx` does for quantized stages.
    """
    return [
        base.with_name(f"{base.name}.stage{index}{suffix}")
        for index in range(1, len(EXIT_NAMES) + 1)
    ]


def export_exit_stages(
    model: EarlyExitResNet,
    base: Path,
    *,
    image_size: int = IMAGE_SIZE,
    opset: int = DEFAULT_OPSET,
) -> list[Path]:
    """Export each stage of an early-exit model as its own graph.

    Each later stage is traced with the previous stage's real output as its
    example, so its input shape is whatever that stage actually produces.
    """
    model = model.to("cpu").eval()
    paths = stage_paths(base)
    x = torch.randn(1, 3, image_size, image_size)

    for index, (stage, head, path) in enumerate(zip(model.stages, model.heads, paths, strict=True)):
        last = index == len(paths) - 1
        export_onnx(
            ExitStage(stage, head, last=last),
            path,
            opset=opset,
            example=x,
            output_names=(OUTPUT_NAME,) if last else (FEATURES_NAME, OUTPUT_NAME),
        )
        # no_grad rather than inference_mode: a tensor made under inference
        # mode cannot be traced, and the exporter's default non-strict tracing
        # fails on it before silently retrying in strict mode.
        with torch.no_grad():
            x = stage(x)

    return paths


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


def evaluate_session(session: ort.InferenceSession, loader) -> dict[str, float]:
    """Measure top-1 accuracy of an ONNX model over a dataloader.

    Float32 and quantized models are evaluated through this same function, so a
    reported accuracy drop is a difference between the models rather than a
    difference between two evaluation paths.
    """
    correct = 0
    seen = 0

    for images, targets in loader:
        outputs = session.run(None, {INPUT_NAME: images.numpy()})[0]
        correct += int((outputs.argmax(axis=1) == targets.numpy()).sum())
        seen += int(targets.shape[0])

    return {"accuracy": correct / seen, "images": seen}


def evaluate_stages(sessions: Sequence[ort.InferenceSession], loader) -> list[float]:
    """Top-1 accuracy at every exit of chained stage graphs over a dataloader.

    The early-exit counterpart of `evaluate_session`, with the stages chained
    as they run when deployed.
    """
    correct = [0] * len(sessions)
    seen = 0

    for images, targets in loader:
        x = images.numpy()
        labels = targets.numpy()
        for index, session in enumerate(sessions):
            outputs = run_stage(session, x)
            correct[index] += int((outputs[OUTPUT_NAME].argmax(axis=1) == labels).sum())
            x = outputs.get(FEATURES_NAME)
        seen += int(labels.shape[0])

    return [hits / seen for hits in correct]


def verify_parity(
    model: nn.Module,
    path: Path,
    *,
    image_size: int = IMAGE_SIZE,
    samples: int = 8,
    atol: float = 1e-4,
) -> dict[str, float]:
    """Compare PyTorch and ONNX Runtime outputs on identical inputs.

    Two things are checked, and the second matters more.

    The raw outputs are compared numerically, which catches an operator that
    was exported with different semantics. Exact equality is not expected:
    the two engines use different kernels and different accumulation orders, so
    tiny floating point differences are normal and harmless.

    The predicted classes are then compared, which is what actually decides
    accuracy. A large numerical difference that never changes a prediction is a
    curiosity; a small one that flips a prediction is a bug. Agreement here is
    required, not merely expected.
    """
    model = model.to("cpu").eval()
    session = build_session(path)

    generator = torch.Generator().manual_seed(SEED)
    inputs = torch.randn(samples, 3, image_size, image_size, generator=generator)

    with torch.inference_mode():
        torch_output = model(inputs).numpy()

    onnx_output = session.run(None, {INPUT_NAME: inputs.numpy()})[0]

    return _compare(torch_output, onnx_output, samples=samples, atol=atol)


def verify_stage_parity(
    model: EarlyExitResNet,
    paths: Sequence[Path],
    *,
    image_size: int = IMAGE_SIZE,
    samples: int = 8,
    atol: float = 1e-4,
) -> list[dict[str, float]]:
    """Compare every exit of the chained stage graphs against PyTorch.

    The graphs are chained the way they run when deployed: each stage receives
    the previous ONNX stage's features, not PyTorch's. Any error a stage
    introduces therefore reaches every later exit, and is caught there if it
    grows enough to matter. One row is returned per exit, checked as in
    `verify_parity`.
    """
    model = model.to("cpu").eval()
    sessions = [build_session(path) for path in paths]

    generator = torch.Generator().manual_seed(SEED)
    inputs = torch.randn(samples, 3, image_size, image_size, generator=generator)

    with torch.inference_mode():
        torch_outputs = [output.numpy() for output in model(inputs)]

    rows = []
    x = inputs.numpy()
    for name, session, expected in zip(EXIT_NAMES, sessions, torch_outputs, strict=True):
        outputs = run_stage(session, x)
        row = _compare(expected, outputs[OUTPUT_NAME], samples=samples, atol=atol)
        rows.append({"exit_name": name, **row})
        x = outputs.get(FEATURES_NAME)
    return rows


def _compare(
    torch_output: np.ndarray,
    onnx_output: np.ndarray,
    *,
    samples: int,
    atol: float,
) -> dict[str, float]:
    """Numerical and prediction agreement between two sets of class scores."""
    absolute_error = np.abs(torch_output - onnx_output)
    torch_predictions = torch_output.argmax(axis=1)
    onnx_predictions = onnx_output.argmax(axis=1)
    agreement = float((torch_predictions == onnx_predictions).mean())

    return {
        "samples": samples,
        "max_abs_error": float(absolute_error.max()),
        "mean_abs_error": float(absolute_error.mean()),
        "prediction_agreement": agreement,
        "tolerance": atol,
        "within_tolerance": bool(absolute_error.max() <= atol),
        "predictions_match": agreement == 1.0,
    }
