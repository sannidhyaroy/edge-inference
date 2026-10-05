"""Turn a per-image exit profile into exit vectors and threshold trade-offs.

A profile records what every exit answers for every image. Nothing here runs a
model: a stopping rule is applied to the recorded answers after the fact, which
is what lets one profile stand in for any number of thresholds.

Two views come out of it, matching the two kinds of early exit:

* **Exit vectors**, `c` and `a`: cost and accuracy at each exit over all images.
  This is what a controller that picks the exit from system state consumes,
  as in Angelucci et al.
* **A threshold sweep**: stop at the first exit whose confidence clears a
  threshold, and report the resulting accuracy, latency and exit mix. This is
  confidence-based early exit, decided per image.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _per_image(frame: pd.DataFrame, column: str) -> np.ndarray:
    """One row per image, one column per exit, shallowest exit first."""
    return frame.pivot(index="image", columns="exit", values=column).sort_index(axis=1).to_numpy()


def exit_vectors(frame: pd.DataFrame) -> pd.DataFrame:
    """Cost and accuracy of each exit when every image is forced to stop there."""
    grouped = frame.groupby(["exit", "exit_name"], sort=True)
    return grouped.agg(
        mmacs=("mmacs", "first"),
        accuracy=("correct", "mean"),
        median_ms=("cumulative_ms", "median"),
        p95_ms=("cumulative_ms", lambda s: s.quantile(0.95)),
        mean_confidence=("confidence", "mean"),
    ).reset_index()


def choose_exits(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    """Index of the exit each image stops at under a confidence threshold.

    An image stops at the first exit whose top probability reaches the
    threshold. The final exit always answers, whatever its confidence, because
    a network cannot decline to produce an output.
    """
    confident = _per_image(frame, "confidence") >= threshold
    confident[:, -1] = True
    # argmax on booleans returns the first True along the row.
    return confident.argmax(axis=1)


def sweep_thresholds(frame: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    """Accuracy, latency, cost and exit mix at each confidence threshold.

    Latency for an image is the cumulative time to the exit it stopped at, as
    recorded in the profile. The mean shows the average saving; the p95 shows
    whether the tail, which a deadline cares about, moved with it.
    """
    correct = _per_image(frame, "correct").astype(bool)
    latency = _per_image(frame, "cumulative_ms")
    mmacs = _per_image(frame, "mmacs")
    exits = correct.shape[1]
    rows_index = np.arange(correct.shape[0])

    rows = []
    for threshold in thresholds:
        chosen = choose_exits(frame, threshold)
        taken_ms = latency[rows_index, chosen]
        row: dict[str, float] = {
            "threshold": float(threshold),
            "accuracy": float(correct[rows_index, chosen].mean()),
            "mean_ms": float(taken_ms.mean()),
            "median_ms": float(np.median(taken_ms)),
            "p95_ms": float(np.percentile(taken_ms, 95)),
            "mean_mmacs": float(mmacs[rows_index, chosen].mean()),
        }
        counts = np.bincount(chosen, minlength=exits)
        for index in range(exits):
            row[f"share_exit{index + 1}"] = float(counts[index] / len(chosen))
        rows.append(row)

    return pd.DataFrame(rows)


def default_thresholds() -> list[float]:
    """0.10 to 0.95 in steps of 0.05, plus 0.99.

    The range starts low on purpose. The early heads here are underconfident:
    the first exit is right about two times in three while its median top
    probability is near 0.26, so a sweep starting at 0.5 would barely ever let
    an image stop there.
    """
    return [round(t, 2) for t in np.arange(0.10, 0.951, 0.05)] + [0.99]
