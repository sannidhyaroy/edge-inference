"""Hedged plans, the record score, and the yardsticks that bound them.

A hedge sends the photo as the frame arrives and runs a local plan at the same
time. By the deadline the frame has the server's answer if it arrived, and the
device's own answer otherwise.
"""

import math

import numpy as np
import pytest

from builders import NETWORK, measurements
from edge_inference.offload import Network
from edge_inference.policies import plan_family
from edge_inference.scoreboard import HEDGE, OURS_HEDGED, Setup, run_setup
from edge_inference.simulate import (
    Measurements,
    Plan,
    Scenario,
    markov_states,
    resolve,
    score,
)
from edge_inference.yardsticks import FIELDS, ladder, ladder_runs
from test_offload import components
from test_yardsticks import random_table

ROWS = np.arange(4)
INSTANT = Network(rtt_ms=0, uplink_mbps=1e9)
NEVER = Network(rtt_ms=1e6, uplink_mbps=1e-6)


def run(plan, network=NETWORK, *, deadline_ms=20.0, slowdown=1.0, cancel=False):
    return resolve(
        measurements(),
        plan,
        ROWS,
        network.rtt_ms,
        network.uplink_mbps,
        slowdown,
        deadline_ms=deadline_ms,
        cancel=cancel,
    )


def useful(outcome, deadline_ms=20.0):
    return outcome.correct & (outcome.total_ms <= deadline_ms)


def test_a_hedge_over_a_link_that_never_delivers_answers_locally():
    local = run(Plan("int8"), NEVER)
    hedged = run(Plan("int8", hedge=True), NEVER)

    assert useful(hedged).tolist() == useful(local).tolist()
    # The local answer is ready at 3 ms but given at the deadline.
    assert hedged.total_ms.tolist() == [20.0] * 4
    assert (hedged.answered_at == 2).all()


def test_a_hedge_over_an_instant_link_answers_like_sending():
    sent = run(Plan(), INSTANT)
    hedged = run(Plan("int8", hedge=True), INSTANT)

    assert useful(hedged).tolist() == useful(sent).tolist()
    assert hedged.total_ms.tolist() == sent.total_ms.tolist()
    assert (hedged.answered_at == -1).all()


def test_a_hedge_sends_every_frame_and_keeps_both_answers():
    hedged = run(Plan("float32", hedge=True))

    assert hedged.sent_bytes.tolist() == [1000.0] * 4
    # The photo takes 15 ms; float32 to the end takes 6 ms.
    assert hedged.server_at_ms.tolist() == [15.0] * 4
    assert hedged.local_ms.tolist() == [6.0] * 4
    assert hedged.server_correct.all()
    assert hedged.local_correct.tolist() == [True, True, True, False]


def test_when_both_are_late_the_first_answer_is_given():
    # At 4x float32 takes 24 ms and the photo 15 ms; both miss 10 ms.
    hedged = run(Plan("float32", hedge=True), deadline_ms=10.0, slowdown=4.0)

    assert hedged.total_ms.tolist() == [15.0] * 4
    assert (hedged.answered_at == -1).all()


def test_cancelling_stops_the_local_run_at_the_next_exit_and_never_costs_more():
    # At 4x the exits come at 8, 16 and 24 ms; the answer is in at 15 ms.
    whole = run(Plan("float32", hedge=True), slowdown=4.0)
    cut = run(Plan("float32", hedge=True), slowdown=4.0, cancel=True)

    assert whole.device_busy_ms.tolist() == [24.0] * 4
    assert cut.device_busy_ms.tolist() == [16.0] * 4
    assert cut.correct.tolist() == whole.correct.tolist()


def test_a_hedge_needs_a_deadline():
    with pytest.raises(ValueError):
        resolve(measurements(), Plan("int8", hedge=True), ROWS, 10, 8, 1.0)


def test_the_record_keeps_a_late_answer_until_the_horizon():
    # Image 3 is wrong locally; the server is right on every image.
    hedged = run(Plan("int8", hedge=True), deadline_ms=10.0)

    def scores(horizon_ms):
        return score(
            hedged.total_ms,
            hedged.correct,
            hedged.device_busy_ms,
            hedged.sent_bytes,
            10.0,
            local_correct=hedged.local_correct,
            server_at_ms=hedged.server_at_ms,
            server_correct=hedged.server_correct,
            horizon_ms=horizon_ms,
        )

    at_deadline, open_ended = scores(10.0), scores(math.inf)
    # By 10 ms only the local answers are in: three right.
    assert at_deadline["useful"] == at_deadline["record"] == 0.75
    assert at_deadline["corrected"] == 0.0
    # The server's answer lands at 15 ms and puts image 3 right.
    assert open_ended["record"] == 1.0
    assert open_ended["corrected"] == 0.25


def test_hedged_plans_join_the_family_only_when_asked():
    table = measurements()
    plain = plan_family(table, (0.5,))
    both = plan_family(table, (0.5,), hedge=True)

    assert not any(plan.hedge for plan in plain)
    assert both[: len(plain)] == plain
    hedged = [plan for plan in both if plan.hedge]
    assert hedged and all(plan.otherwise == "answer" for plan in hedged)


def test_no_hedged_plan_beats_the_best_possible_on_any_frame():
    rng = np.random.default_rng(7)
    for _ in range(3):
        table = random_table(rng)
        scenario = Scenario(
            table,
            rng.permutation(len(table)),
            markov_states(len(table), (0.8, 0.8), 3),
            [NETWORK, Network(rtt_ms=60, uplink_mbps=2)],
            ["good", "bad"],
            slowdown=2.0,
            deadline_ms=25.0,
        )
        runs = {name: run for name, _, run in ladder_runs(scenario, (0.4,), hedge=True)}
        best = runs["4. best possible per frame"]
        best_hit = best["correct"].astype(bool) & (best["total_ms"] <= 25.0)
        for plan in plan_family(table, (0.4,), hedge=True):
            outcome = scenario.resolve(plan)
            hit = outcome.correct & (outcome.total_ms <= 25.0)
            assert not (hit & ~best_hit).any(), plan.describe()


def test_the_rungs_never_cross_with_hedges_and_records():
    rng = np.random.default_rng(11)
    for seed in range(3):
        table = random_table(rng)
        scenario = Scenario(
            table,
            rng.permutation(len(table)),
            markov_states(len(table), (0.8, 0.8), seed),
            [NETWORK, Network(rtt_ms=60, uplink_mbps=2)],
            ["good", "bad"],
            slowdown=2.0,
            deadline_ms=25.0,
            horizon_ms=40.0,
        )
        board = ladder(scenario, (0.4,), hedge=True).set_index("rule")["useful"]
        order = [
            "1. best fixed option",
            "2. best option per network",
            "3. best confidence rule per network",
            "3h. best plan per network, hedges included",
            "4. best possible per frame",
        ]
        values = board[order].to_numpy()
        assert (np.diff(values) >= -1e-12).all()
        records = ladder(scenario, (0.4,), hedge=True).set_index("rule")["record"]
        assert (records <= records["4r. best possible record per frame"] + 1e-12).all()


def test_every_run_carries_the_events():
    scenario = Scenario(measurements(), ROWS, np.zeros(4, dtype=int), [NETWORK], ["net"], 1.0, 20.0)
    for _, _, run in ladder_runs(scenario, (0.5,), hedge=True):
        assert set(FIELDS) <= set(run)


def test_the_table_refuses_blocks_that_disagree_on_images():
    rows = components()
    photo = rows[(rows["mode"] == "offload") & (rows["server"] == "t4")]
    with pytest.raises(ValueError):
        Measurements.from_components(rows.drop(photo.index[:1]), "t4")


def test_the_scoreboard_adds_hedging_rules_and_rungs_only_when_asked():
    table = Measurements.from_arrays(**_bigger())
    plain = run_setup(table, Setup(passes=1), every_variant=False).board
    hedged = run_setup(table, Setup(passes=1, hedge=True), every_variant=False).board

    new = {HEDGE, OURS_HEDGED, "3h. best plan per network, hedges included"}
    assert not new & set(plain["rule"])
    assert new <= set(hedged["rule"])
    assert {"record", "corrected"} <= set(hedged.columns)
    # Adding hedges changes nothing the plain board already had.
    shared = hedged.set_index("rule").loc[plain["rule"], "useful"].to_numpy()
    np.testing.assert_allclose(shared, plain["useful"].to_numpy())


def _bigger() -> dict:
    """The four-image table repeated ten times, so each class can be split in half."""
    from builders import arrays

    base = arrays()

    def tile(value):
        if isinstance(value, dict):
            return {key: tile(array) for key, array in value.items()}
        return np.concatenate([value] * 10)

    tiled = {name: tile(value) for name, value in base.items()}
    tiled["images"] = np.arange(40)
    return tiled
