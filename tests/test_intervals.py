import numpy as np
import pytest

from edge_inference.intervals import paired_interval


def test_a_constant_difference_has_no_width():
    assert paired_interval(np.full(500, 0.25)) == pytest.approx((0.25, 0.25))


def test_independent_frames_give_the_usual_width():
    rng = np.random.default_rng(1)
    difference = rng.choice([-1.0, 0.0, 1.0], size=20_000, p=[0.2, 0.5, 0.3])
    low, high = paired_interval(difference, block=1)
    assert low < difference.mean() < high
    expected = 2 * 1.96 * difference.std() / np.sqrt(len(difference))
    assert high - low == pytest.approx(expected, rel=0.15)


def test_runs_of_alike_frames_widen_the_interval():
    # The difference flips sign only every 100 frames, as when a link stays
    # good or bad for a while: far less evidence than 5000 independent frames.
    signs = np.random.default_rng(2).choice([-1.0, 1.0], size=50)
    difference = np.repeat(signs, 100)
    single = paired_interval(difference, block=1)
    blocks = paired_interval(difference, block=100)
    assert blocks[1] - blocks[0] > 5 * (single[1] - single[0])
