"""How much a device can know, cheaply, about whether an image is hard.

A device that could tell a hard image in advance would send it straight to the
server. This measures the signals it actually has, from the per-image table:
the size of the photo, known before anything runs, and each exit's confidence,
known once the device has run that far.

Each signal is scored by its AUROC for predicting that a given exit answers
correctly: the chance that a randomly chosen image answered correctly has a
higher signal than a randomly chosen image answered wrongly. 0.5 is a coin
flip and 1.0 a perfect signal. A signal that runs the other way, lower for
correct answers, is reported with its sign flipped and says so.
"""

import numpy as np
import pandas as pd


def auroc(signal: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve, from ranks; tied signals share their rank."""
    signal = np.asarray(signal, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    hits, misses = int(positive.sum()), int((~positive).sum())
    if hits == 0 or misses == 0:
        return float("nan")
    ranks = pd.Series(signal).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - hits * (hits + 1) / 2) / (hits * misses))


def _local(components: pd.DataFrame, point: str, precision: str) -> pd.DataFrame:
    rows = components[
        (components["mode"] == "local")
        & (components["point"] == point)
        & (components["device_precision"] == precision)
    ]
    return rows.set_index("image").sort_index()


def difficulty_signals(components: pd.DataFrame, precision: str = "float32") -> pd.DataFrame:
    """AUROC of each cheap signal for each outcome a device would care about."""
    photo = components[components["mode"] == "offload"].drop_duplicates("image")
    photo_bytes = photo.set_index("image").sort_index()["payload_bytes"]
    exits = {
        point: _local(components, point, precision) for point in ("layer2", "layer3", "layer4")
    }

    cases = [
        ("photo size, before running", photo_bytes, "layer2"),
        ("photo size, before running", photo_bytes, "layer4"),
        ("exit 1 confidence", exits["layer2"]["confidence"], "layer2"),
        ("exit 1 confidence", exits["layer2"]["confidence"], "layer4"),
        ("exit 2 confidence", exits["layer3"]["confidence"], "layer3"),
        ("exit 2 confidence", exits["layer3"]["confidence"], "layer4"),
    ]
    names = {"layer2": "exit 1", "layer3": "exit 2", "layer4": "final exit"}
    rows = []
    for signal_name, signal, point in cases:
        correct = exits[point]["correct"].astype(bool)
        signal = signal.reindex(correct.index)
        score = auroc(signal.to_numpy(), correct.to_numpy())
        flipped = score < 0.5
        rows.append(
            {
                "signal": signal_name,
                "predicts": f"{names[point]} is right",
                "precision": precision,
                "auroc": 1.0 - score if flipped else score,
                "direction": "lower means right" if flipped else "higher means right",
                "right": int(correct.sum()),
                "images": len(correct),
            }
        )
    return pd.DataFrame(rows)
