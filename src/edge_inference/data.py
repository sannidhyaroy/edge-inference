"""Imagenette loading and preprocessing.

Imagenette is a ten-class subset of ImageNet published by fast.ai. The classes
were chosen to be visually distinct, so a network can reach useful accuracy in
minutes rather than the days full ImageNet demands.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette
from torchvision.transforms import v2

from edge_inference.config import (
    DATA_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGENETTE_SIZE,
    SEED,
)

# Ratio between the resize target and the final crop at evaluation time. The
# short side is resized slightly above the crop so the centre crop keeps the
# subject rather than trimming into it. 1.14 is the familiar 256/224 ratio.
EVAL_RESIZE_RATIO = 1.14


def build_transforms(*, train: bool) -> v2.Compose:
    """Build the preprocessing pipeline for one split.

    Training and evaluation deliberately differ. Training applies random crops
    and horizontal flips, which is data augmentation: showing the network a
    slightly different view of each image every epoch, so it learns the object
    rather than memorising exact pixels. Evaluation must be deterministic, so it
    takes a fixed centre crop and nothing else. Augmenting the test set would
    make results irreproducible and optimistic.
    """
    if train:
        stages: list[v2.Transform] = [
            v2.RandomResizedCrop(IMAGE_SIZE, antialias=True),
            v2.RandomHorizontalFlip(),
        ]
    else:
        stages = [
            v2.Resize(int(IMAGE_SIZE * EVAL_RESIZE_RATIO), antialias=True),
            v2.CenterCrop(IMAGE_SIZE),
        ]

    return v2.Compose(
        [
            *stages,
            # ToImage converts to a tensor, ToDtype scales the 0-255 integer
            # pixels into 0.0-1.0 floats, and Normalize then applies the
            # ImageNet channel statistics the pretrained weights expect.
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_split(
    split: str,
    *,
    root: Path = DATA_DIR,
    download: bool = False,
    augment: bool | None = None,
) -> Imagenette:
    """Load one Imagenette split, downloading it first if asked.

    `split` is "train" or "val". The validation split is used as the test set
    throughout, since Imagenette ships only these two.

    `augment` defaults to augmenting the training split and not the validation
    one, which is what training wants. It is overridable because quantization
    calibration needs training *images* with evaluation *preprocessing*:
    calibration measures the range of values flowing through the network, and
    those ranges have to match what inference will actually see. Random crops
    would measure a distribution that never occurs at inference time.
    """
    root.mkdir(parents=True, exist_ok=True)
    transform = build_transforms(train=split == "train" if augment is None else augment)

    try:
        return Imagenette(
            root=str(root),
            split=split,
            size=IMAGENETTE_SIZE,
            download=download,
            transform=transform,
        )
    except RuntimeError:
        # torchvision refuses to download over an archive it has already
        # extracted, and signals that with a RuntimeError. Treating an existing
        # copy as success is what makes `data prepare` safe to re-run.
        return Imagenette(
            root=str(root),
            split=split,
            size=IMAGENETTE_SIZE,
            download=False,
            transform=transform,
        )


def build_dataloader(
    dataset: Imagenette,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap a dataset in a loader with reproducible shuffling.

    `num_workers` above zero loads images in background processes, and how
    those processes start matters. Windows and macOS have always used `spawn`,
    which launches a fresh interpreter that re-imports the entry module. As of
    Python 3.14 Linux defaults to `forkserver` rather than `fork`, which
    re-imports as well, because forking a process that already has threads is
    unsafe.

    The practical consequence is that **every** platform now re-imports the
    entry module in its workers, so any caller must be guarded by
    `if __name__ == "__main__":` or the import recurses and the workers die
    with a BrokenPipeError. The CLI is written that way for exactly this
    reason.
    """
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
