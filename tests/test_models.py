"""Sanity checks for the early-exit model."""

import torch
from torchvision import models

from edge_inference.models import EXIT_NAMES, EarlyExitResNet, build_early_exit

NUM_CLASSES = 10
IMAGE_SIZE = 64


def test_every_exit_produces_class_scores():
    model = build_early_exit(num_classes=NUM_CLASSES).eval()
    inputs = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)

    with torch.inference_mode():
        outputs = model(inputs)

    assert len(outputs) == len(EXIT_NAMES)
    for output in outputs:
        assert output.shape == (3, NUM_CLASSES)


def test_final_exit_is_the_plain_resnet():
    """The wrapper must not change what the full network computes.

    If it did, the final exit's accuracy and cost would not be comparable to
    the baseline model, and every early-exit result would be measured against
    a different network than the one quantized and benchmarked before.
    """
    torch.manual_seed(0)
    plain = models.resnet18(num_classes=NUM_CLASSES).eval()
    wrapped = EarlyExitResNet(plain, NUM_CLASSES).eval()
    inputs = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)

    with torch.inference_mode():
        expected = plain(inputs)
        actual = wrapped(inputs)[-1]

    torch.testing.assert_close(actual, expected)


def test_stages_can_run_one_at_a_time():
    """Per-exit timing and split computing both run the network stage by stage."""
    model = build_early_exit(num_classes=NUM_CLASSES).eval()
    inputs = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    with torch.inference_mode():
        together = model(inputs)
        x = inputs
        separate = []
        for stage, head in zip(model.stages, model.heads, strict=True):
            x = stage(x)
            separate.append(head(x))

    for joint, staged in zip(together, separate, strict=True):
        torch.testing.assert_close(joint, staged)
