"""Model cost accounting: parameters, operations, and size on disk.

These are the hardware-independent half of the picture. A latency measurement
describes a model on one machine; an operation count describes the model
itself, and transfers unchanged to any device. The Angelucci paper's `c` vector
is exactly this: operations required to reach each exit, which is why a
scheduler can treat a profiled network as a black box.

**A note on units, because the literature is inconsistent.** A multiply and an
accumulate are one MAC (multiply-accumulate) but two FLOPs (floating point
operations). Convolutions are almost entirely MACs, so a network's "FLOP count"
is roughly twice its MAC count, and papers disagree about which they quote.
Both are reported here, and results state which is used.
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from edge_inference.config import IMAGE_SIZE


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count weights, split by whether they are trained.

    Parameters determine file size and memory footprint, but not runtime cost.
    A large fully connected layer holds many weights and uses each one once,
    while a small convolution kernel is reused across every spatial position.
    That is why parameter count and operation count must both be reported.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "parameters_total": total,
        "parameters_trainable": trainable,
    }


def count_operations(
    model: nn.Module,
    *,
    image_size: int = IMAGE_SIZE,
    batch_size: int = 1,
    device: str = "cpu",
) -> dict[str, float]:
    """Count operations for one forward pass.

    Uses PyTorch's own `FlopCounterMode`, which intercepts the dispatcher and
    tallies real operations rather than pattern matching on module types. That
    matters for the early-exit work later: a counter that walks the module tree
    would miscount a network whose forward pass returns early, while this one
    counts what actually executed.
    """
    model = model.to(device).eval()
    inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)

    counter = FlopCounterMode(display=False)
    with counter, torch.inference_mode():
        model(inputs)

    flops = float(counter.get_total_flops())
    return {
        "flops": flops,
        "mflops": flops / 1e6,
        "macs": flops / 2,
        "mmacs": flops / 2 / 1e6,
        "image_size": image_size,
        "batch_size": batch_size,
    }


def operations_by_module(
    model: nn.Module,
    *,
    image_size: int = IMAGE_SIZE,
    batch_size: int = 1,
    device: str = "cpu",
    depth: int = 2,
) -> dict[str, float]:
    """Return FLOPs attributed to each submodule, as a flat mapping.

    This is the groundwork for the `c` vector. Once exit heads are attached
    partway through the backbone, the cost of reaching each exit is the running
    sum of the stages before it, which requires a per-stage breakdown rather
    than a single total.
    """
    model = model.to(device).eval()
    inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)

    counter = FlopCounterMode(display=False, depth=depth)
    with counter, torch.inference_mode():
        model(inputs)

    # get_flop_counts maps a module path to a mapping of operator to count, so
    # the per-module total is the sum over its operators.
    return {module: float(sum(ops.values())) for module, ops in counter.get_flop_counts().items()}


def checkpoint_size_bytes(path: Path) -> int:
    """Size of a saved checkpoint on disk.

    Reported as measured rather than derived from parameter count, because the
    two diverge: a serialised file carries metadata, and a quantized model
    stores eight bit integers where the original stored thirty-two bit floats.
    The size reduction from quantization is one of the headline numbers, so it
    is measured on the real artefact.
    """
    return path.stat().st_size


def profile_model(
    model: nn.Module,
    *,
    image_size: int = IMAGE_SIZE,
    batch_size: int = 1,
    device: str = "cpu",
    checkpoint: Path | None = None,
) -> dict[str, float]:
    """Collect every hardware-independent cost measure into one row."""
    row: dict[str, float] = {
        **count_parameters(model),
        **count_operations(
            model,
            image_size=image_size,
            batch_size=batch_size,
            device=device,
        ),
    }
    if checkpoint is not None and checkpoint.exists():
        size = checkpoint_size_bytes(checkpoint)
        row["checkpoint_bytes"] = size
        row["checkpoint_mib"] = size / (1024 * 1024)
    return row
