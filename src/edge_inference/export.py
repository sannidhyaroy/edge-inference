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

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from edge_inference.config import IMAGE_SIZE, SEED

# Opset 17 is old enough to be supported everywhere that matters and new enough
# for the quantization tooling. Pinning it keeps an exported graph reproducible
# rather than dependent on whatever the installed torch defaults to.
DEFAULT_OPSET = 17

INPUT_NAME = "input"
OUTPUT_NAME = "logits"


def export_onnx(
    model: nn.Module,
    path: Path,
    *,
    image_size: int = IMAGE_SIZE,
    opset: int = DEFAULT_OPSET,
) -> Path:
    """Export `model` to ONNX at `path` and return the path.

    The batch dimension is marked dynamic. Inference here is always batch size
    one, but calibrating a quantized model feeds batches through the same file,
    and a graph frozen at batch one would reject them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()

    example = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=opset,
    )
    return path


def build_session(path: Path, *, threads: int = 1) -> ort.InferenceSession:
    """Open an ONNX Runtime session with the thread count pinned.

    Thread count is set explicitly for the same reason it is in the PyTorch
    harness: it is how a laptop stands in for a constrained device, and a
    default that varies by machine would make results incomparable.

    inter_op is fixed at one because it parallelises across independent
    branches of a graph, and a sequential backbone has none. Leaving it free
    would add scheduling overhead for no benefit.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1

    return ort.InferenceSession(
        str(path),
        options,
        providers=["CPUExecutionProvider"],
    )


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
