"""How sure a difference between two rules is, from the frames both played.

Two rules on a scoreboard play the same frames over the same recorded link,
so their difference is judged frame by frame: a paired comparison. Frames
next to each other meet nearly the same network, so they are not independent
draws, and resampling single frames would make an interval too narrow. The
resampling therefore draws whole runs of consecutive frames (a circular block
bootstrap): each replicate is as many blocks as fit in the run, each block
`block` frames long from a random start, wrapping past the end.
"""

import numpy as np

from edge_inference.config import SEED

BLOCK_FRAMES = 100
RESAMPLES = 2000


def paired_interval(
    difference: np.ndarray,
    *,
    block: int = BLOCK_FRAMES,
    resamples: int = RESAMPLES,
    level: float = 0.95,
    seed: int = SEED,
) -> tuple[float, float]:
    """A `level` interval for the mean of a per-frame `difference`, in frame order."""
    difference = np.asarray(difference, dtype=float)
    count = len(difference)
    block = max(1, min(block, count))
    blocks = int(np.ceil(count / block))
    # The sum of every block, one per start, wrapping round.
    wrapped = np.concatenate([difference, difference[: block - 1]])
    sums = np.convolve(wrapped, np.ones(block), mode="valid")[:count]
    starts = np.random.default_rng(seed).integers(0, count, size=(resamples, blocks))
    means = sums[starts].sum(axis=1) / (blocks * block)
    tail = (1 - level) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return float(low), float(high)
