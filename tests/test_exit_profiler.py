"""Sanity checks for the per-image exit profiler."""

import math

import torch
from torch.utils.data import TensorDataset

from edge_inference.exit_profiler import entropy, max_entropy, profile_exits
from edge_inference.models import EXIT_NAMES, build_early_exit

NUM_CLASSES = 4


def test_entropy_bounds():
    certain = torch.tensor([1.0, 0.0, 0.0, 0.0])
    uniform = torch.full((NUM_CLASSES,), 1 / NUM_CLASSES)

    assert math.isclose(entropy(certain), 0.0, abs_tol=1e-6)
    assert math.isclose(entropy(uniform), max_entropy(NUM_CLASSES), rel_tol=1e-6)


def test_one_row_per_image_per_exit():
    torch.manual_seed(0)
    model = build_early_exit(num_classes=NUM_CLASSES)
    images = torch.randn(3, 3, 32, 32)
    labels = torch.tensor([0, 1, 2])
    macs = [1e6, 2e6, 3e6]

    rows = profile_exits(
        model,
        TensorDataset(images, labels),
        threads=1,
        warmup=1,
        macs_per_exit=macs,
    )

    assert len(rows) == 3 * len(EXIT_NAMES)
    for image in range(3):
        mine = [row for row in rows if row["image"] == image]
        assert [row["exit_name"] for row in mine] == list(EXIT_NAMES)
        # Reaching a deeper exit can never take less time than a shallower one.
        times = [row["cumulative_ms"] for row in mine]
        assert times == sorted(times)
        for row in mine:
            assert 0.0 <= row["confidence"] <= 1.0
            assert 0.0 <= row["entropy"] <= max_entropy(NUM_CLASSES) + 1e-6
            assert row["correct"] == (row["predicted"] == row["label"])


def test_limit_samples_across_the_dataset():
    """A limited run must not simply take the first N images.

    Image folders are sorted by class, so the first N would all share a label
    and measure one class instead of the dataset.
    """
    model = build_early_exit(num_classes=NUM_CLASSES)
    labels = torch.arange(4).repeat_interleave(10)  # sorted by class, like a folder
    dataset = TensorDataset(torch.randn(40, 3, 32, 32), labels)

    rows = profile_exits(
        model, dataset, threads=1, warmup=0, macs_per_exit=[1.0, 2.0, 3.0], limit=8
    )

    sampled = {row["image"] for row in rows}
    assert len(sampled) == 8
    assert len({row["label"] for row in rows}) > 1
