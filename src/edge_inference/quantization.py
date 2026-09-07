"""Post-training static INT8 quantization through ONNX Runtime.

Quantization stores weights and activations as 8-bit integers instead of 32-bit
floats. Three things follow, and only two of them are guaranteed:

* **The file shrinks about fourfold.** A property of the format, true anywhere.
* **Accuracy changes slightly**, because 256 integer levels cannot represent
  every float exactly. Also a property of the model, true anywhere.
* **Speed may or may not improve.** This depends entirely on whether the CPU
  has instructions for 8-bit dot products. A chip with AVX2 but no VNNI
  computes INT8 correctly and gains far less than published figures suggest.

*Post-training* means the model is quantized after it has finished training,
with no retraining. *Static* means the ranges are measured in advance from real
data, rather than computed per inference as dynamic quantization does. Static
is what a deployed vision model wants: the measurement cost is paid once,
offline, instead of on every image.

**Calibration** is that measurement step. A few hundred representative images
are pushed through the float model while the tooling records the range of
values at each layer. Those ranges decide how the float span maps onto 256
integer levels. Calibrate on unrepresentative data and the ranges are wrong,
which shows up as an accuracy drop that looks like a quantization problem but
is really a data problem.
"""

from pathlib import Path

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch.utils.data import DataLoader

from edge_inference.export import INPUT_NAME


class DataLoaderCalibrationReader(CalibrationDataReader):
    """Feeds batches from a dataloader to the calibrator until a limit is hit.

    ONNX Runtime asks for one batch at a time by calling `get_next`, and stops
    when it returns None. Wrapping a dataloader means calibration sees exactly
    the preprocessing inference will use, which is the whole point.
    """

    def __init__(self, loader: DataLoader, *, limit: int, input_name: str = INPUT_NAME) -> None:
        self.input_name = input_name
        self.limit = limit
        self.seen = 0
        self._iterator = iter(loader)

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.seen >= self.limit:
            return None
        try:
            images, _ = next(self._iterator)
        except StopIteration:
            return None
        self.seen += int(images.shape[0])
        return {self.input_name: images.numpy()}

    def rewind(self) -> None:
        raise NotImplementedError("this reader is single pass; build a new one instead")


def quantize_onnx(
    source: Path,
    destination: Path,
    calibration: CalibrationDataReader,
    *,
    per_channel: bool = True,
) -> Path:
    """Quantize an ONNX model to INT8 and return the written path.

    The model is preprocessed first. ONNX Runtime's own tooling requires shape
    inference and a few graph fixes before quantization, and skipping that step
    produces failures that point at the quantizer rather than at the missing
    preprocessing.

    `per_channel` gives each convolution filter its own scale rather than one
    scale for the whole layer. Filters in a layer often have very different
    weight ranges, and a single shared scale wastes integer levels on the
    narrow ones. It costs nothing at inference and usually recovers most of the
    accuracy that quantization would otherwise lose.

    Weights are signed and activations unsigned, which is what x86 kernels
    expect. Getting this backwards is a common way to end up on a slow fallback
    path without any error saying so.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared = destination.with_name(f"{destination.stem}.prepared.onnx")

    quant_pre_process(str(source), str(prepared), skip_symbolic_shape=False)

    quantize_static(
        str(prepared),
        str(destination),
        calibration,
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )

    prepared.unlink(missing_ok=True)
    return destination
