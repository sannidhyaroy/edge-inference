"""The menu server's models in PyTorch, for a server with a GPU.

A server with a GPU runs the same pretrained weights as the ONNX graphs, from
their makers, with the same input recipe: the photo is decoded and prepared
on the CPU exactly as `serving` does, then the batch goes to the GPU. The
time to run therefore includes copying the batch over and waiting for the
answer, which is what a request pays.

Only float32 is offered, so the answers are those already recorded on the
CPU for every photo; a timing run records its own answers to check that.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from edge_inference.preprocess import Preprocess
from edge_inference.zoo import build, recipe


@dataclass
class TorchServed:
    """One model on a torch device."""

    model: nn.Module
    recipe: Preprocess
    device: torch.device

    @torch.inference_mode()
    def infer(self, x: np.ndarray) -> np.ndarray:
        logits = self.model(torch.from_numpy(x).to(self.device))
        return logits[0].float().cpu().numpy()


def open_torch_model(key: str, device: str = "cuda") -> TorchServed:
    """`model@resolution/precision`, its weights downloaded if needed."""
    name, _, rest = key.partition("@")
    resolution, _, precision = rest.partition("/")
    if (precision or "float32") != "float32":
        raise ValueError(f"{key}: the PyTorch backend runs float32 only")
    target = torch.device(device)
    model = build(name).to(target)
    return TorchServed(model, recipe(name).at(int(resolution)), target)


def device_name(device: str = "cuda") -> str:
    target = torch.device(device)
    if target.type == "cuda":
        return torch.cuda.get_device_name(target)
    return "cpu"
