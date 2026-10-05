"""Sanity checks for applying stopping rules to an exit profile.

The profile is built by hand so every expected value can be worked out on
paper. Image 0 is easy: confident and right from the first exit. Image 1 is
hard: unsure and wrong early, right only at the end.
"""

import pandas as pd
import pytest

from edge_inference.exit_analysis import choose_exits, exit_vectors, sweep_thresholds

#          image exit  ms   mmacs confidence correct
ROWS = [
    (0, 1, 10.0, 500.0, 0.90, True),
    (0, 2, 15.0, 700.0, 0.95, True),
    (0, 3, 20.0, 900.0, 0.99, True),
    (1, 1, 10.0, 500.0, 0.30, False),
    (1, 2, 15.0, 700.0, 0.60, False),
    (1, 3, 20.0, 900.0, 0.80, True),
]


def profile() -> pd.DataFrame:
    frame = pd.DataFrame(
        ROWS, columns=["image", "exit", "cumulative_ms", "mmacs", "confidence", "correct"]
    )
    frame["exit_name"] = frame["exit"].map({1: "layer2", 2: "layer3", 3: "layer4"})
    return frame


def test_exit_vectors_force_every_image_to_each_exit():
    vectors = exit_vectors(profile())

    assert list(vectors["mmacs"]) == [500.0, 700.0, 900.0]
    assert list(vectors["accuracy"]) == [0.5, 0.5, 1.0]


def test_images_stop_at_the_first_confident_exit():
    assert list(choose_exits(profile(), 0.5)) == [0, 1]
    assert list(choose_exits(profile(), 0.85)) == [0, 2]


def test_final_exit_answers_even_when_unsure():
    """Nothing reaches 0.999, so everything must fall through to the last exit."""
    assert list(choose_exits(profile(), 0.999)) == [2, 2]


def test_sweep_reports_the_trade_off():
    sweep = sweep_thresholds(profile(), [0.5, 0.85]).set_index("threshold")

    # At 0.5, image 1 stops at exit 2 and is wrong there.
    assert sweep.loc[0.5, "accuracy"] == 0.5
    assert sweep.loc[0.5, "mean_ms"] == pytest.approx(12.5)
    assert sweep.loc[0.5, "share_exit1"] == 0.5
    # At 0.85, image 1 runs to the end and is right.
    assert sweep.loc[0.85, "accuracy"] == 1.0
    assert sweep.loc[0.85, "mean_ms"] == pytest.approx(15.0)
    assert sweep.loc[0.85, "share_exit3"] == 0.5
