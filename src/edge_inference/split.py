"""What crosses the network when inference is split between device and server.

Split computing runs the first stages of the network on the device and sends
the feature map at the cut to a server, which runs the rest. Our exits sit at
the same boundaries, so every exit except the last is also a split point.

Whether splitting is worth it depends mostly on the size of that feature map
against the alternative, sending the image itself. A JPEG is already heavily
compressed; a feature map is not, so a naive split can send more than full
offload. Two cheap reductions are measured here, separately and together:

* **8-bit payload.** Each feature map is mapped onto 256 levels using its own
  minimum and maximum, a quarter of the float32 size. This is lossy, so the
  server's answer can change, and that is measured too.
* **Lossless compression** with zstd. Feature maps after a ReLU are full of
  zeros, which compress well. Compression costs device time, so its time is
  recorded alongside the bytes it saves.

Neither changes the network. A trained bottleneck layer, which literature on
split computing uses to shrink the feature map far further, would.
"""

import random
import time
from collections.abc import Sequence
from compression import zstd
from pathlib import Path

import numpy as np
import onnxruntime as ort
from torch.utils.data import Dataset

from edge_inference.bench import cpu_model, cpu_policy, cpu_state, machine_model, power_source
from edge_inference.config import SEED
from edge_inference.export import FEATURES_NAME, OUTPUT_NAME, run_stage
from edge_inference.models import EXIT_NAMES

# Every exit but the last is a place the network can be cut.
SPLIT_NAMES = EXIT_NAMES[:-1]

# Bytes sent with an 8-bit payload besides the values: a float32 scale and a
# one-byte zero point, so the server can map the levels back to floats.
UINT8_HEADER_BYTES = 5


def quantize_uint8(features: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Map a feature map onto 256 levels spanning its own value range.

    Returns the levels, the scale and the zero point. The range is per tensor,
    taken from this feature map alone, so no calibration is needed and nothing
    about other images leaks in.
    """
    low = min(float(features.min()), 0.0)
    high = max(float(features.max()), 0.0)
    scale = (high - low) / 255.0 or 1.0
    zero_point = round(-low / scale)
    levels = np.clip(np.round(features / scale) + zero_point, 0, 255).astype(np.uint8)
    return levels, scale, zero_point


def dequantize_uint8(levels: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Map 8-bit levels back to the floats the next stage expects."""
    return (levels.astype(np.float32) - zero_point) * scale


def _timed(function, *args):
    start = time.perf_counter()
    result = function(*args)
    return result, (time.perf_counter() - start) * 1000.0


def _finish(sessions: Sequence[ort.InferenceSession], x: np.ndarray) -> int:
    """Run the remaining stages, as the server would, and return its prediction."""
    for session in sessions:
        outputs = run_stage(session, x)
        x = outputs.get(FEATURES_NAME)
    return int(outputs[OUTPUT_NAME][0].argmax())


def profile_splits(
    sessions: Sequence[ort.InferenceSession],
    dataset: Dataset,
    *,
    threads: int,
    warmup: int,
    precision: str,
    files: Sequence[Path] | None = None,
    limit: int | None = None,
    level: int = 3,
) -> list[dict[str, object]]:
    """Payload size and preparation time at every split point, per image.

    Returns one row per image per split point. `sessions` are the stage graphs
    the device runs. The rest of the network is run on the same machine only
    to check whether an 8-bit payload changes the answer; how long the server
    takes is measured on the server.

    `files`, when given, are the image files in dataset order, whose sizes are
    the full-offload payload to compare against.
    """
    machine = machine_model()
    cpu = cpu_model()
    conditions = {"power_source": power_source(), **cpu_policy()}

    # Seeded sample for limited runs, for the same reason as the exit profiler:
    # image folders are sorted by class.
    indices = list(range(len(dataset)))
    if limit is not None and limit < len(indices):
        indices = sorted(random.Random(SEED).sample(indices, limit))

    first = dataset[0][0].unsqueeze(0).numpy()
    for _ in range(warmup):
        features = run_stage(sessions[0], first)[FEATURES_NAME]
        zstd.compress(quantize_uint8(features)[0].tobytes(), level)

    rows: list[dict[str, object]] = []
    for index in indices:
        image, label = dataset[index]
        x = image.unsqueeze(0).numpy()

        # Every split point's feature map comes from one pass, so each is the
        # exact tensor the device would hold at that point.
        maps = []
        for session in sessions[:-1]:
            x = run_stage(session, x)[FEATURES_NAME]
            maps.append(x)
        reference = _finish(sessions[len(maps) :], x)

        for split_index, (name, features) in enumerate(zip(SPLIT_NAMES, maps, strict=True)):
            raw = features.tobytes()
            raw_compressed, raw_ms = _timed(zstd.compress, raw, level)

            (levels, scale, zero_point), quantize_ms = _timed(quantize_uint8, features)
            packed = levels.tobytes()
            packed_compressed, packed_ms = _timed(zstd.compress, packed, level)

            received = dequantize_uint8(levels, scale, zero_point)
            predicted = _finish(sessions[split_index + 1 :], received)

            rows.append(
                {
                    "image": index,
                    "label": int(label),
                    "split": split_index + 1,
                    "split_name": name,
                    "elements": int(features.size),
                    "float32_bytes": len(raw),
                    "float32_zstd_bytes": len(raw_compressed),
                    "float32_zstd_ms": raw_ms,
                    "uint8_bytes": len(packed) + UINT8_HEADER_BYTES,
                    "uint8_quantize_ms": quantize_ms,
                    "uint8_zstd_bytes": len(packed_compressed) + UINT8_HEADER_BYTES,
                    "uint8_zstd_ms": packed_ms,
                    "zero_fraction": float((features == 0).mean()),
                    "predicted": reference,
                    "correct": reference == int(label),
                    "predicted_uint8": predicted,
                    "correct_uint8": predicted == int(label),
                    "jpeg_bytes": files[index].stat().st_size if files is not None else None,
                    "zstd_level": level,
                    "precision": precision,
                    "runtime": "onnxruntime",
                    "threads": threads,
                    "machine": machine,
                    "cpu": cpu,
                    **conditions,
                }
            )

        # Read after the image's timing, so the read itself is never timed.
        state = cpu_state()
        for row in rows[-len(maps) :]:
            row.update(state)

    return rows
