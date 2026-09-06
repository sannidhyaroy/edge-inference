"""Fine-tuning and evaluation.

Fine-tuning means continuing to train a network that already has useful weights
rather than starting from random ones. The backbone here was trained on the
full ImageNet dataset, so its early layers already detect edges, textures, and
shapes. Only the final classification layer is new, and the rest needs nudging
rather than teaching. That is why a few minutes of work reaches accuracy that
training from scratch would need days to match.

One function runs both training and evaluation, because the two differ in
exactly three ways and writing them separately invites the differences to drift
apart:

* training updates the weights, evaluation does not
* training puts the model in train mode, which makes dropout and batch
  normalisation behave differently
* evaluation runs without gradient tracking, which is faster and uses less
  memory
"""

from collections.abc import Iterable

import torch
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from torch import nn
from torch.utils.data import DataLoader

console = Console()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    progress: Progress | None = None,
    description: str = "",
) -> dict[str, float]:
    """Run one pass over `loader`, training if an optimizer is given.

    Returns the mean loss and the accuracy over the whole pass.

    Passing `optimizer=None` makes this an evaluation pass: no weights change
    and no gradients are tracked.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    seen = 0

    task = None
    if progress is not None:
        task = progress.add_task(description, total=len(loader))

    # inference_mode is the stricter, faster sibling of no_grad. It is only
    # valid when nothing needs gradients, so training uses a plain null context.
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            if training:
                # Gradients accumulate by default, so they must be cleared
                # before each step or every batch would be polluted by the last.
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            # argmax over the class dimension turns raw scores into a predicted
            # label. The scores need no softmax first: it is monotonic, so it
            # cannot change which class is largest.
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            seen += batch_size

            if progress is not None and task is not None:
                progress.advance(task)

    if progress is not None and task is not None:
        progress.remove_task(task)

    return {
        "loss": total_loss / seen,
        "accuracy": correct / seen,
    }


def fine_tune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: str,
    epochs: int,
    learning_rate: float,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
) -> list[dict[str, float]]:
    """Fine-tune `model` and return one metrics row per epoch.

    Stochastic gradient descent with momentum is used rather than a modern
    adaptive optimizer. It is the setting these backbones were originally
    trained under, it has one obvious knob, and for a short fine-tune it is
    entirely sufficient.

    The learning rate follows a cosine schedule: high at first so the new
    classifier head moves quickly, then decaying towards zero so the pretrained
    layers are not disturbed once the model is close to a good solution.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: list[dict[str, float]] = []

    columns: Iterable = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )

    with Progress(*columns, console=console) as progress:
        for epoch in range(1, epochs + 1):
            # Read the rate before stepping the scheduler, so the recorded
            # value is the one this epoch actually trained at rather than the
            # one the next epoch will use.
            epoch_learning_rate = optimizer.param_groups[0]["lr"]

            train_metrics = run_epoch(
                model,
                train_loader,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                progress=progress,
                description=f"epoch {epoch}/{epochs} train",
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                device=device,
                criterion=criterion,
                progress=progress,
                description=f"epoch {epoch}/{epochs} val",
            )

            # Stepped once per epoch, matching the T_max the scheduler was
            # built with.
            scheduler.step()

            row = {
                "epoch": epoch,
                "learning_rate": epoch_learning_rate,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            history.append(row)

            console.print(
                f"  epoch {epoch}/{epochs}: "
                f"train loss {row['train_loss']:.4f}, "
                f"val loss {row['val_loss']:.4f}, "
                f"val accuracy [bold]{row['val_accuracy'] * 100:.2f}%[/bold]"
            )

    return history
