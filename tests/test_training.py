"""Sanity checks for multi-exit training."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from edge_inference.models import EXIT_NAMES, build_early_exit
from edge_inference.training import combined_loss, run_epoch


def test_combined_loss_is_a_weighted_average():
    """Weights are relative: an exit weighted zero contributes nothing."""
    criterion = nn.CrossEntropyLoss()
    targets = torch.tensor([0, 1])
    confident = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
    wrong = torch.tensor([[-5.0, 5.0], [5.0, -5.0]])

    only_first = combined_loss([confident, wrong], targets, criterion, [1.0, 0.0])
    torch.testing.assert_close(only_first, criterion(confident, targets))

    even = combined_loss([confident, wrong], targets, criterion, [1.0, 1.0])
    expected = (criterion(confident, targets) + criterion(wrong, targets)) / 2
    torch.testing.assert_close(even, expected)


def test_run_epoch_reports_every_exit():
    torch.manual_seed(0)
    model = build_early_exit(num_classes=4)
    images = torch.randn(6, 3, 32, 32)
    targets = torch.randint(0, 4, (6,))
    loader = DataLoader(TensorDataset(images, targets), batch_size=3)

    metrics = run_epoch(model, loader, device="cpu", criterion=nn.CrossEntropyLoss())

    assert len(metrics["exit_accuracies"]) == len(EXIT_NAMES)
    assert metrics["accuracy"] == metrics["exit_accuracies"][-1]
    assert all(0.0 <= a <= 1.0 for a in metrics["exit_accuracies"])
