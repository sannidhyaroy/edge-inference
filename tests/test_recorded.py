import math

import numpy as np
import pytest

from builders import NETWORK, measurements
from edge_inference.recorded import FAIR_TARGET, RecordedSetup, run_recorded
from edge_inference.scoreboard import HEDGE, OURS, OURS_HEDGED
from edge_inference.traces import Trace

RUNGS = [
    "1. best fixed option",
    "2. best option per network",
    FAIR_TARGET,
    "3h. best plan per network, hedges included",
    "4. best possible per frame",
]


def varying(ticks=400, seed=0):
    """A link that is near half the time and far the other half."""
    rng = np.random.default_rng(seed)
    rtt = np.where(rng.random(ticks) < 0.5, 2.0, 40.0) + rng.random(ticks)
    return Trace("net", rtt, np.full(ticks, NETWORK.uplink_mbps), 100.0)


def board(trace_tune, trace_test, **setup):
    table = measurements()
    return run_recorded(table, table, trace_tune, trace_test,
                        RecordedSetup(deadline_ms=20, passes=50, **setup)).set_index("rule")  # fmt: skip


def test_every_rule_and_yardstick_is_scored_and_the_yardsticks_climb():
    scores = board(varying(seed=1), varying(seed=2))
    for name in (OURS, HEDGE, OURS_HEDGED, "always local float32", "always offload via photo"):
        assert name in scores.index
    useful = scores.loc[RUNGS, "useful"].to_numpy()
    assert (np.diff(useful) >= -1e-12).all()
    assert scores.loc[FAIR_TARGET, "behind_fair"] == pytest.approx(0)
    assert scores.loc[OURS, "useful"] <= scores.loc["4. best possible per frame", "useful"]


def test_on_a_dead_link_sending_scores_nothing_and_the_lookup_stays_local():
    dead = Trace("net", np.full(40, math.inf), np.full(40, 8.0), 100.0)
    # States are fitted on a live tuning recording; every test frame is down.
    scores = board(varying(seed=1), dead)
    assert scores.loc["always offload via photo", "useful"] == 0
    best_local = scores.loc[["always local float32", "always local int8"], "useful"].max()
    assert scores.loc[OURS, "useful"] >= best_local - 1e-12
    assert scores.loc[OURS, "network_share_down"] == 1.0
