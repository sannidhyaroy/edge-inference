"""Per-image latency, confidence and correctness at every exit.

The other result files summarise: one row per configuration, with every image
averaged away. This keeps every image, so one profile can answer questions that
a summary cannot:

* **Any confidence threshold can be applied afterwards.** Earlier work fixes a
  threshold before measuring; here the threshold is a filter over the table.
* **Distributions survive.** A decision policy under deadlines needs the tail
  of the latency distribution, not its mean.
* **The `a` vector falls out directly**, as accuracy per exit over all images.

**What varies between images, and what does not.** Every image does exactly
the same arithmetic in a CNN, so the time to reach an exit is a property of the
network and the machine, not the picture. Its spread across images is system
jitter. Confidence and correctness are what genuinely depend on the image, and
they are what decides where an early-exit network stops.
"""

import math
import random
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import Dataset

from edge_inference.bench import (
    cpu_model,
    cpu_policy,
    cpu_state,
    machine_model,
    power_source,
    set_thread_count,
)
from edge_inference.config import SEED
from edge_inference.export import FEATURES_NAME, OUTPUT_NAME, run_stage
from edge_inference.models import EXIT_NAMES, EarlyExitResNet

# One exit's work: take the previous stage's output, run the next stage and its
# head, and return (features for the next stage, probabilities, confidence,
# predicted class). Everything inside it is what gets timed.
ExitStep = Callable[[int, Any], tuple[Any, Any, float, int]]


def entropy(probabilities: torch.Tensor) -> float:
    """Shannon entropy in nats: zero when certain, ln(classes) when uniform.

    A second measure of confidence alongside the top probability. The top
    probability looks only at the winner; entropy looks at the whole spread, so
    it notices when the runner-up is close behind. BranchyNet uses entropy.
    """
    p = probabilities.clamp_min(1e-12)
    return float(-(p * p.log()).sum())


def profile_exits(
    model: EarlyExitResNet,
    dataset: Dataset,
    *,
    threads: int,
    warmup: int,
    macs_per_exit: list[float],
    limit: int | None = None,
    device: str = "cpu",
) -> list[dict[str, object]]:
    """Run each image through the network one stage at a time.

    Returns one row per image per exit. Every exit is always reached, because
    the point is to record what each exit *would* answer, so that any stopping
    rule can be evaluated afterwards from the same measurements.

    The timed region for each exit covers its stage, its head, and the softmax
    and argmax a deployed network would need to decide whether to stop. Entropy
    is computed outside the timer: it is for analysis, not part of the decision.
    """
    effective_threads = set_thread_count(threads)
    model = model.to(device).eval()

    def step(exit_index: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, int]:
        x = model.stages[exit_index](x)
        probabilities = model.heads[exit_index](x).softmax(dim=1)[0]
        confidence, predicted = probabilities.max(dim=0)
        return x, probabilities, float(confidence), int(predicted)

    with torch.inference_mode():
        return _profile(
            step,
            lambda image: image.unsqueeze(0).to(device),
            dataset,
            warmup=warmup,
            macs_per_exit=macs_per_exit,
            limit=limit,
            labels={
                "precision": "float32",
                "runtime": "pytorch",
                "device": device,
                "threads": effective_threads,
            },
        )


def profile_exit_sessions(
    sessions: Sequence[ort.InferenceSession],
    dataset: Dataset,
    *,
    threads: int,
    warmup: int,
    macs_per_exit: list[float],
    precision: str,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """The same profile as `profile_exits`, through chained ONNX Runtime stages.

    This is the deployed path: each stage is its own graph, and the exit
    decision happens in the application between them. `threads` must be what
    the sessions were built with; ONNX Runtime cannot report it back.
    """

    def step(exit_index: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
        outputs = run_stage(sessions[exit_index], x)
        logits = outputs[OUTPUT_NAME][0]
        exponentials = np.exp(logits - logits.max())
        probabilities = exponentials / exponentials.sum()
        predicted = int(probabilities.argmax())
        return outputs.get(FEATURES_NAME), probabilities, float(probabilities[predicted]), predicted

    return _profile(
        step,
        lambda image: image.unsqueeze(0).numpy(),
        dataset,
        warmup=warmup,
        macs_per_exit=macs_per_exit,
        limit=limit,
        labels={
            "precision": precision,
            "runtime": "onnxruntime",
            "device": "cpu",
            "threads": threads,
        },
    )


def _profile(
    step: ExitStep,
    prepare: Callable[[torch.Tensor], Any],
    dataset: Dataset,
    *,
    warmup: int,
    macs_per_exit: list[float],
    limit: int | None,
    labels: dict[str, object],
) -> list[dict[str, object]]:
    """Time every exit for every image, whichever runtime `step` uses."""
    machine = machine_model()
    cpu = cpu_model()
    # Read once at the start. Power and frequency policy decide latency by large
    # factors on a laptop, so every row carries the conditions it ran under.
    conditions = {"power_source": power_source(), **cpu_policy()}

    # A limited run takes a seeded random sample rather than the first N
    # images. Image folders are stored sorted by class, so the first N would
    # all be one class, and a single class says nothing about the dataset.
    indices = list(range(len(dataset)))
    if limit is not None and limit < len(indices):
        indices = sorted(random.Random(SEED).sample(indices, limit))

    # Warm up on the first image so allocator growth and kernel selection are
    # paid before anything is timed, as in the benchmark harness.
    first, _ = dataset[0]
    for _ in range(warmup):
        x = prepare(first)
        for exit_index in range(len(EXIT_NAMES)):
            x, *_ = step(exit_index, x)

    rows: list[dict[str, object]] = []
    for index in indices:
        image, label = dataset[index]
        x = prepare(image)
        cumulative_ms = 0.0

        for exit_index, exit_name in enumerate(EXIT_NAMES):
            start = time.perf_counter()
            x, probabilities, confidence, predicted = step(exit_index, x)
            step_ms = (time.perf_counter() - start) * 1000.0
            cumulative_ms += step_ms

            rows.append(
                {
                    "image": index,
                    "label": int(label),
                    "exit": exit_index + 1,
                    "exit_name": exit_name,
                    "step_ms": step_ms,
                    "cumulative_ms": cumulative_ms,
                    "mmacs": macs_per_exit[exit_index] / 1e6,
                    "confidence": confidence,
                    "entropy": entropy(torch.as_tensor(probabilities)),
                    "predicted": predicted,
                    "correct": predicted == int(label),
                    **labels,
                    "machine": machine,
                    "cpu": cpu,
                    **conditions,
                }
            )

        # The clock and temperature the image just ran at, read after its
        # timing so the read itself is never timed.
        state = cpu_state()
        for row in rows[-len(EXIT_NAMES) :]:
            row.update(state)

    return rows


def max_entropy(num_classes: int) -> float:
    """Entropy of a uniform guess, the ceiling for a given number of classes."""
    return math.log(num_classes)
