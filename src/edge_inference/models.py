"""Backbone construction.

A backbone is the feature extracting body of a network, everything before the
final classification layer. Keeping construction behind one function means a
second architecture arrives as an argument rather than a rewrite, which matters
because the plan is to compare architectures under the same measurements.
"""

import torch
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


# Where each exit sits, named after the ResNet block it follows. Used to label
# results, so a row says which exit it describes rather than an index.
EXIT_NAMES = ("layer2", "layer3", "layer4")


def _out_channels(stage: nn.Module) -> int:
    """Channel count a stage produces, read from its last batch norm layer."""
    norms = [module for module in stage.modules() if isinstance(module, nn.BatchNorm2d)]
    return norms[-1].num_features


def _exit_head(in_channels: int, num_classes: int) -> nn.Sequential:
    """Pool the feature map to one value per channel, then classify.

    This is the same shape as ResNet's own final classifier, so all three exits
    are structurally identical and differ only in how deep the features they
    see are. The head is deliberately tiny: a few thousand weights against the
    millions in the stage before it, so reaching an exit costs almost nothing
    beyond the stage itself.
    """
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Linear(in_channels, num_classes),
    )


class EarlyExitResNet(nn.Module):
    """A ResNet with classifier heads after layer2 and layer3.

    The network is split into three stages, each followed by a head:

        stem, layer1, layer2   head after layer2
        layer3                 head after layer3
        layer4                 the original final classifier

    Exits sit after layer2 and layer3 because those are where ResNet halves the
    spatial resolution and doubles the channels, so each exit sees a distinctly
    more abstract representation than the one before. An exit after layer1
    would see features too low-level to classify with, and the stem is mostly
    edges.

    Keeping stages and heads as separate lists means the same model serves
    three purposes. Training runs everything and needs every exit's output.
    Latency profiling runs one stage at a time and times each. Split computing
    cuts the network between two stages and sends the intermediate tensor over
    the network.
    """

    def __init__(self, backbone: models.ResNet, num_classes: int) -> None:
        super().__init__()
        stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )
        self.stages = nn.ModuleList([stem, backbone.layer3, backbone.layer4])
        self.heads = nn.ModuleList(
            [
                _exit_head(_out_channels(stem), num_classes),
                _exit_head(_out_channels(backbone.layer3), num_classes),
                # The final exit reuses the backbone's own classifier, so its
                # weights, and its output, are exactly the plain ResNet's.
                nn.Sequential(backbone.avgpool, nn.Flatten(1), backbone.fc),
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run every stage and return the raw scores from every exit.

        All exits are always computed here, because training needs a loss at
        each and accuracy has to be measured at each over the whole test set.
        Actually stopping early is a separate concern, handled where inference
        is timed.
        """
        outputs = []
        for stage, head in zip(self.stages, self.heads, strict=True):
            x = stage(x)
            outputs.append(head(x))
        return outputs


def build_early_exit(
    name: str = "resnet18",
    *,
    num_classes: int,
    pretrained: bool = False,
) -> EarlyExitResNet:
    """Build a backbone and wrap it with early exits."""
    backbone = build_backbone(name, num_classes=num_classes, pretrained=pretrained)
    return EarlyExitResNet(backbone, num_classes)
