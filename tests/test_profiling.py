"""Sanity checks for operation counting."""

import pytest
import torch
from torchvision import models

from edge_inference.models import EarlyExitResNet
from edge_inference.profiling import count_operations, operations_per_exit

NUM_CLASSES = 10
IMAGE_SIZE = 64


def test_exits_cost_more_the_deeper_they_are():
    model = EarlyExitResNet(models.resnet18(num_classes=NUM_CLASSES), NUM_CLASSES)

    macs = operations_per_exit(model, image_size=IMAGE_SIZE)

    assert len(macs) == 3
    assert macs[0] < macs[1] < macs[2]


def test_final_exit_costs_what_the_plain_network_costs():
    """The extra heads are tiny, so the full path must match a plain ResNet."""
    torch.manual_seed(0)
    plain = models.resnet18(num_classes=NUM_CLASSES)
    expected = count_operations(plain, image_size=IMAGE_SIZE)["macs"]
    model = EarlyExitResNet(plain, NUM_CLASSES)

    macs = operations_per_exit(model, image_size=IMAGE_SIZE)

    assert macs[-1] == pytest.approx(expected, rel=1e-3)
    assert macs[-1] >= expected
