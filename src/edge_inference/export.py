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
import onnx
import onnxruntime as ort
import torch
from torch import nn

from edge_inference.config import IMAGE_SIZE, SEED

# What torch's dynamo exporter emits natively for these models. Requesting an
# older opset makes it export at 18 and then down-convert, which fails for this
# graph and silently leaves the file at 18 anyway. Better to name the version
# actually produced and verify it than to request one and be quietly ignored.
DEFAULT_OPSET = 18

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

    Weights are written into the file rather than beside it. torch defaults to
    external data, which splits a model into a small graph plus a `.onnx.data`
    blob. That is necessary past protobuf's 2 GB ceiling and a liability below
    it: the size measured on disk becomes the graph alone, which would turn the
    quantization size comparison into nonsense. These models are tens of
    megabytes, so a single self-contained file is both simpler and honest.
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
