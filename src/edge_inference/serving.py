"""A server for the model menu: receive a photo, prepare it, run a model, answer.

It holds one or more models, each an ONNX graph with its input recipe, and a
request names the one it wants. It needs ONNX Runtime and Pillow, not torch.
Its reply says what it answered, how sure it was, and how long it took: the
time to decode and prepare the photo, and the time to run the model.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from edge_inference.ort_session import INPUT_NAME, OUTPUT_NAME
from edge_inference.preprocess import Preprocess, decode_photo


def answer(logits: np.ndarray) -> tuple[int, float]:
    """The class picked and the probability given to it."""
    logits = logits.astype(np.float64)
    exponentials = np.exp(logits - logits.max())
    probabilities = exponentials / exponentials.sum()
    predicted = int(probabilities.argmax())
    return predicted, float(probabilities[predicted])


@dataclass
class Served:
    """One model the server can run, on ONNX Runtime's CPU provider.

    `serving_torch` holds the same shape for PyTorch on a GPU: anything with
    a `recipe` and an `infer` from a prepared batch of one to its logits.
    """

    session: ort.InferenceSession
    recipe: Preprocess

    def infer(self, x: np.ndarray) -> np.ndarray:
        return self.session.run([OUTPUT_NAME], {INPUT_NAME: x})[0][0]


def open_model(graph: Path, recipe: Path, *, threads: int) -> Served:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
    return Served(session, Preprocess.load(recipe))


@dataclass
class MenuServer:
    """Answers photo requests with the model each request names, or the default."""

    models: dict[str, Served]
    default: str
    clock: object = field(default=time.perf_counter)

    def handle(self, header: dict, body: bytes) -> dict:
        if header.get("kind") != "photo":
            raise ValueError(f"cannot answer a {header.get('kind')!r} request")
        name = header.get("model") or self.default
        served = self.models[name]
        start = self.clock()
        x = served.recipe(decode_photo(body))
        prepared = self.clock()
        logits = served.infer(x)
        predicted, confidence = answer(logits)
        done = self.clock()
        return {
            "model": name,
            "predicted": predicted,
            "confidence": confidence,
            "receive_ms": (prepared - start) * 1000.0,
            "compute_ms": (done - prepared) * 1000.0,
            "server_ms": (done - start) * 1000.0,
        }
