"""Shared paths, dataset constants, and reproducibility helpers.

Everything here is deliberately boring and global. The point is that a number
produced today can be reproduced next week on another machine, which means the
seed, the image size, and the normalisation constants have to live in exactly
one place.
"""

import random
from pathlib import Path

import torch

# One fixed seed for the whole project. Any script that draws random numbers
# calls seed_everything() before doing so.
SEED = 42

# Paths resolve relative to the repository root rather than the current working
# directory, so a command behaves identically no matter where it is run from.
# This file lives at <root>/src/edge_inference/config.py, hence parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Imagenette is a ten-class subset of ImageNet. The classes were picked to be
# easy to tell apart, which makes it possible to reach useful accuracy in a few
# epochs instead of the days a full ImageNet run would take.
#
# It ships at several resolutions. The 160px variant has its shortest side
# resized to 160 pixels, small enough to fine-tune on a CPU in minutes.
IMAGENETTE_SIZE = "160px"
NUM_CLASSES = 10

# Images are cropped to a fixed square before entering the network. Every
# convolutional backbone here accepts any input size, but latency scales with
# it, so it is pinned to keep measurements comparable.
IMAGE_SIZE = 160

# Per-channel mean and standard deviation of the ImageNet training set, in RGB
# order. Normalisation rescales each colour channel to roughly zero mean and
# unit variance. The pretrained weights were trained on inputs prepared this
# way, so applying the same transform is what makes those weights meaningful
# here. Getting this wrong degrades accuracy silently rather than raising.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = SEED) -> None:
    """Seed every random number generator this project actually draws from.

    Python's `random` covers standard library shuffling, and `torch` covers
    weight initialisation, dropout, and dataloader shuffling.

    NumPy's global generator is deliberately not seeded. Nothing in this
    pipeline draws from it, and seeding a global generator is the legacy NumPy
    pattern anyway. Any future NumPy randomness should take an explicit
    `np.random.default_rng(SEED)` generator instead, which is both reproducible
    and local to the caller.

    Note that this does not enable deterministic algorithm selection via
    `torch.use_deterministic_algorithms`. That setting makes some operations
    considerably slower and makes others raise outright, which is a poor trade
    for a project whose headline numbers are latency measurements.
    """
    random.seed(seed)
    torch.manual_seed(seed)
