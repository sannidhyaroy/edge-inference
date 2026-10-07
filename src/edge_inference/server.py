"""Server-side time for every way a frame can arrive.

The device either sends the image itself (full offload) or the feature map at a
split point. Either way the server pays two costs, timed separately per image:

* **Receiving**: turning the bytes that arrived into the tensor the network
  takes. For a JPEG that is decoding and the same preprocessing the device
  would apply. For a float32 feature map it is reinterpreting the bytes, nearly
  free. For a compressed 8-bit one it is decompressing and mapping back to
  floats.
* **Computing**: the rest of the network from wherever the device stopped,
  including copying the input to the accelerator and the answer back. On a GPU
  the work is queued asynchronously, so timing waits for it to finish.

The payload is built on the server from the same image, outside the timer. Its
content barely affects either cost, and building it there means the server
needs only the checkpoint, not files produced on the device.

Two backends share one timing loop: PyTorch, for a GPU server, and ONNX Runtime
on CPU, for a CPU server such as the laptop at full thread count.
"""

import io
import random
import time
from collections.abc import Callable, Sequence
from compression import zstd
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torch.utils.data import Dataset

from edge_inference.bench import (
    cpu_model,
    cpu_policy,
    cpu_state,
    machine_model,
    power_source,
    synchronize,
)
from edge_inference.config import SEED
from edge_inference.data import build_transforms
from edge_inference.export import FEATURES_NAME, OUTPUT_NAME, run_stage
from edge_inference.models import EarlyExitResNet
from edge_inference.split import SPLIT_NAMES, dequantize_uint8, quantize_uint8


class Server(Protocol):
    """What the timing loop needs from a backend."""

    def features(self, x: np.ndarray) -> list[np.ndarray]:
        """Feature maps at every split point for one preprocessed image."""
        ...

    def run_from(self, stage: int, x: np.ndarray) -> int:
        """Run from `stage` (0 is the whole network) to the end; return the class."""
        ...


class TorchServer:
    """PyTorch on any device, final classifier only.

    The server only ever answers from the final exit: it took the frame because
    the device wanted the full network's answer, so the early heads are skipped.
    """

    def __init__(self, model: EarlyExitResNet, device: str) -> None:
        self.model = model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def features(self, x: np.ndarray) -> list[np.ndarray]:
        tensor = torch.from_numpy(x).to(self.device)
        maps = []
        for stage in self.model.stages[: len(SPLIT_NAMES)]:
            tensor = stage(tensor)
            maps.append(tensor.cpu().numpy())
        return maps

    @torch.inference_mode()
    def run_from(self, stage: int, x: np.ndarray) -> int:
        tensor = torch.from_numpy(x).to(self.device)
        for module in self.model.stages[stage:]:
            tensor = module(tensor)
        predicted = int(self.model.heads[-1](tensor).argmax(dim=1).item())
        synchronize(self.device)
        return predicted


class OnnxServer:
    """ONNX Runtime on CPU through the stage graphs.

    Stage graphs compute each stage's exit head too. Those heads are a few
    thousand weights against millions in the stages, so the time they add is
    within run-to-run noise.
    """

    def __init__(self, sessions: Sequence[ort.InferenceSession]) -> None:
        self.sessions = list(sessions)

    def features(self, x: np.ndarray) -> list[np.ndarray]:
        maps = []
        for session in self.sessions[: len(SPLIT_NAMES)]:
            x = run_stage(session, x)[FEATURES_NAME]
            maps.append(x)
        return maps

    def run_from(self, stage: int, x: np.ndarray) -> int:
        for session in self.sessions[stage:]:
            outputs = run_stage(session, x)
            x = outputs.get(FEATURES_NAME)
        return int(outputs[OUTPUT_NAME][0].argmax())


_TRANSFORM = build_transforms(train=False)


def _decode_jpeg(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return _TRANSFORM(image).unsqueeze(0).numpy()


def _receive_float32(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    # Into a mutable buffer, as a socket receive would be. A read-only view of
    # `bytes` makes PyTorch warn, and the copy is a few tens of microseconds.
    return np.frombuffer(bytearray(data), dtype=np.float32).reshape(shape)


def _receive_uint8_zstd(data: bytes, shape: tuple[int, ...], scale: float, zero: int) -> np.ndarray:
    levels = np.frombuffer(zstd.decompress(data), dtype=np.uint8).reshape(shape)
    return dequantize_uint8(levels, scale, zero)


@dataclass(frozen=True)
class Arrival:
    """One way a frame can reach the server, ready to be received and run."""

    mode: str
    split_name: str | None
    payload: str
    data: bytes
    # Stage the server starts from: 0 for a whole image, else the split point.
    stage: int
    receive: Callable[[], np.ndarray]


def arrivals(server: Server, image: torch.Tensor, data: bytes, *, level: int) -> list[Arrival]:
    """Every arrival for one image, built in full before anything is timed."""
    built = [Arrival("offload", None, "jpeg", data, 0, partial(_decode_jpeg, data))]
    for stage, (name, features) in enumerate(
        zip(SPLIT_NAMES, server.features(image.unsqueeze(0).numpy()), strict=True), start=1
    ):
        raw = features.tobytes()
        built.append(
            Arrival(
                "split", name, "float32", raw, stage, partial(_receive_float32, raw, features.shape)
            )
        )
        levels, scale, zero = quantize_uint8(features)
        packed = zstd.compress(levels.tobytes(), level)
        receive = partial(_receive_uint8_zstd, packed, features.shape, scale, zero)
        built.append(Arrival("split", name, "uint8_zstd", packed, stage, receive))
    return built


def _timed(function, *args):
    start = time.perf_counter()
    result = function(*args)
    return result, (time.perf_counter() - start) * 1000.0


def profile_server(
    server: Server,
    dataset: Dataset,
    files: Sequence[Path],
    *,
    warmup: int,
    labels: dict[str, object],
    limit: int | None = None,
    level: int = 3,
) -> list[dict[str, object]]:
    """Receive and compute time per image for every arrival mode.

    Returns five rows per image: the JPEG for full offload, then a float32 and
    a compressed 8-bit feature map at each of the two split points. `labels`
    are columns describing the backend, copied onto every row.
    """
    machine = machine_model()
    cpu = cpu_model()
    conditions = {"power_source": power_source(), **cpu_policy()}

    indices = list(range(len(dataset)))
    if limit is not None and limit < len(indices):
        indices = sorted(random.Random(SEED).sample(indices, limit))

    first, _ = dataset[indices[0]]
    warm = arrivals(server, first, files[indices[0]].read_bytes(), level=level)
    for _ in range(warmup):
        for arrival in warm:
            server.run_from(arrival.stage, arrival.receive())

    rows: list[dict[str, object]] = []
    for index in indices:
        image, label = dataset[index]
        first_row = len(rows)
        for arrival in arrivals(server, image, files[index].read_bytes(), level=level):
            x, receive_ms = _timed(arrival.receive)
            predicted, compute_ms = _timed(server.run_from, arrival.stage, x)
            rows.append(
                {
                    "image": index,
                    "label": int(label),
                    "mode": arrival.mode,
                    "split_name": arrival.split_name,
                    "payload": arrival.payload,
                    "payload_bytes": len(arrival.data),
                    "receive_ms": receive_ms,
                    "compute_ms": compute_ms,
                    "total_ms": receive_ms + compute_ms,
                    "predicted": predicted,
                    "correct": predicted == int(label),
                    **labels,
                    "machine": machine,
                    "cpu": cpu,
                    **conditions,
                }
            )

        # The CPU's clock and temperature, read after the image's timing. On a
        # GPU server the CPU still decodes JPEGs and copies data.
        state = cpu_state()
        for row in rows[first_row:]:
            row.update(state)
    return rows
