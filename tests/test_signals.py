import numpy as np
import pandas as pd
import pytest

from edge_inference.signals import auroc, difficulty_signals


def test_auroc_perfect_chance_and_reversed():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.5


def test_auroc_counts_pairs():
    # Right: 0.2, 0.4. Wrong: 0.1, 0.3. Of the 2 x 2 pairs, 3 have the right one higher.
    assert auroc(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 1, 0, 1])) == pytest.approx(0.75)


def test_auroc_undefined_without_both_classes():
    assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([1, 1])))


def _table() -> pd.DataFrame:
    rows = []
    # Image 0 and 1 are easy (confident, right); 2 and 3 are hard.
    confidence = [0.9, 0.8, 0.3, 0.2]
    correct = [True, True, False, True]
    size = [8000, 9000, 6000, 7000]
    for image in range(4):
        for point in ("layer2", "layer3", "layer4"):
            rows.append(
                {
                    "image": image,
                    "mode": "local",
                    "point": point,
                    "device_precision": "float32",
                    "payload_bytes": 0,
                    "confidence": confidence[image],
                    "correct": correct[image],
                }
            )
        rows.append(
            {
                "image": image,
                "mode": "offload",
                "point": "input",
                "device_precision": "none",
                "payload_bytes": size[image],
                "confidence": confidence[image],
                "correct": correct[image],
            }
        )
    return pd.DataFrame(rows)


def test_difficulty_signals_reads_the_right_rows():
    table = difficulty_signals(_table())
    assert len(table) == 6
    exit1 = table[
        (table.signal == "exit 1 confidence") & (table.predicts == "exit 1 is right")
    ].iloc[0]
    # Right: 0.9, 0.8, 0.2. Wrong: 0.3. Two of the three pairs rank the right one higher.
    assert exit1.auroc == pytest.approx(2 / 3) and exit1.right == 3 and exit1.images == 4
    size = table[
        (table.signal == "photo size, before running") & (table.predicts == "exit 1 is right")
    ]
    assert size.iloc[0].auroc == 1.0 and size.iloc[0].direction == "higher means right"
