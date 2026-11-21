"""Which images a dataset holds, where each came from, and what that does to accuracy.

Imagenette is cut from ImageNet, and the backbone here starts from weights
trained on ImageNet. Most of Imagenette's validation images are ImageNet
*training* images, so the pretrained weights have seen them. If that inflated
accuracy, images ImageNet kept for validation would score lower than the rest.
This module labels every image with its origin, read from its file name, and
compares the two groups with a confidence interval wide enough to be honest
about how few validation-origin images there are.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

IMAGENET_VAL = "imagenet_val"
IMAGENET_TRAIN = "imagenet_train"


def origin_of(path: Path | str) -> str:
    """Where in ImageNet an image came from, from the file name ImageNet gave it.

    ImageNet names validation images `ILSVRC2012_val_<n>.JPEG` and training
    images `<wnid>_<n>.JPEG`. Imagenette keeps those names.
    """
    return IMAGENET_VAL if Path(path).name.startswith("ILSVRC2012_val_") else IMAGENET_TRAIN


def manifest(
    files: Sequence[Path], labels: Sequence[int], root: Path | None = None
) -> pd.DataFrame:
    """One row per image, in dataset index order: index, label, file and origin.

    Paths are written relative to `root` when it is given, so the manifest does
    not record where the dataset happens to live on this machine.
    """
    if len(files) != len(labels):
        raise ValueError(f"{len(files)} files but {len(labels)} labels")
    return pd.DataFrame(
        {
            "image": np.arange(len(files)),
            "label": np.asarray(labels, dtype=int),
            "file": [str(Path(f).relative_to(root)) if root else str(f) for f in files],
            "origin": [origin_of(f) for f in files],
        }
    )


def wilson(successes: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    """A 95% Wilson score interval for a proportion.

    Unlike the normal approximation it stays inside [0, 1] and behaves near
    100%, which is where a final exit's accuracy sits.
    """
    if trials == 0:
        return float("nan"), float("nan")
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return float(centre - half), float(centre + half)


def accuracy_by_origin(profile: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per exit for each origin, with a 95% interval.

    `profile` has one row per image and exit (`image`, `exit_name`, `correct`);
    `images` is a manifest. Every profiled image must be in the manifest.
    """
    origin = images.set_index("image")["origin"]
    missing = set(profile["image"]) - set(origin.index)
    if missing:
        raise ValueError(f"{len(missing)} profiled images are not in the manifest")
    frame = profile.assign(origin=profile["image"].map(origin))
    rows = []
    for (exit_name, where), group in frame.groupby(["exit_name", "origin"], sort=True):
        right, total = int(group["correct"].sum()), len(group)
        low, high = wilson(right, total)
        rows.append(
            {
                "exit_name": exit_name,
                "origin": where,
                "images": total,
                "correct": right,
                "accuracy": right / total,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)
