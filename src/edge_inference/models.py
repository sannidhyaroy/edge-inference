"""Backbone construction.

A backbone is the feature extracting body of a network, everything before the
final classification layer. Keeping construction behind one function means a
second architecture arrives as an argument rather than a rewrite, which matters
because the plan is to compare architectures under the same measurements.
"""

from torch import nn
from torchvision import models

SUPPORTED_BACKBONES = ("resnet18",)


def build_backbone(
    name: str = "resnet18",
    *,
    num_classes: int | None = None,
    pretrained: bool = False,
) -> nn.Module:
    """Build a backbone, optionally with ImageNet weights and a fresh head.

    `pretrained=True` downloads weights trained on the full ImageNet dataset.
    Starting from those instead of random values is transfer learning: the
    early layers have already learned generic edge and texture detectors, so
    only the later, more task specific layers need adjusting. This is what
    makes a few minutes of fine-tuning enough.

    Passing `num_classes` replaces the final fully connected layer, because
    ImageNet has a thousand classes and Imagenette has ten. The replacement is
    randomly initialised and therefore useless until trained, which is expected:
    fine-tuning is what teaches it.
    """
    if name not in SUPPORTED_BACKBONES:
        supported = ", ".join(SUPPORTED_BACKBONES)
        raise ValueError(f"unknown backbone {name!r}, expected one of: {supported}")

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if num_classes is not None:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
