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

import torch
from torch.utils.data import Dataset

from edge_inference.bench import (
    cpu_model,
    cpu_policy,
    machine_model,
    power_source,
    set_thread_count,
)
from edge_inference.config import SEED
from edge_inference.models import EXIT_NAMES, EarlyExitResNet


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

    with torch.inference_mode():
        # Warm up on the first image so allocator growth and kernel selection
        # are paid before anything is timed, as in the benchmark harness.
        first, _ = dataset[0]
        for _ in range(warmup):
            model(first.unsqueeze(0).to(device))

        rows: list[dict[str, object]] = []
        for index in indices:
            image, label = dataset[index]
            x = image.unsqueeze(0).to(device)
            cumulative_ms = 0.0

            for exit_index, (stage, head) in enumerate(zip(model.stages, model.heads, strict=True)):
                start = time.perf_counter()
                x = stage(x)
                probabilities = head(x).softmax(dim=1)[0]
                confidence, predicted = probabilities.max(dim=0)
                step_ms = (time.perf_counter() - start) * 1000.0
                cumulative_ms += step_ms

                rows.append(
                    {
                        "image": index,
                        "label": int(label),
                        "exit": exit_index + 1,
                        "exit_name": EXIT_NAMES[exit_index],
                        "step_ms": step_ms,
                        "cumulative_ms": cumulative_ms,
                        "mmacs": macs_per_exit[exit_index] / 1e6,
                        "confidence": float(confidence),
                        "entropy": entropy(probabilities),
                        "predicted": int(predicted),
                        "correct": int(predicted) == int(label),
                        "precision": "float32",
                        "runtime": "pytorch",
                        "device": device,
                        "threads": effective_threads,
                        "machine": machine,
                        "cpu": cpu,
                        **conditions,
                    }
                )

    return rows


def max_entropy(num_classes: int) -> float:
    """Entropy of a uniform guess, the ceiling for a given number of classes."""
    return math.log(num_classes)
