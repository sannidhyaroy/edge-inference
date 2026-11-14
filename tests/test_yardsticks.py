"""Checks that the four yardsticks are what they claim, and never cross."""

import dataclasses

import numpy as np

from builders import CONFIDENCE, NETWORK, measurements
from edge_inference.offload import Network
from edge_inference.policies import Always, plan_family
from edge_inference.simulate import Measurements, Scenario, markov_states, score_trace, simulate
from edge_inference.yardsticks import ladder

BAD = Network(rtt_ms=100, uplink_mbps=8)


def scenario(table, states, *, slowdown, deadline_ms):
    return Scenario(
        table,
        frames=np.arange(len(states)) % len(table),
        states=np.array(states),
        networks=[NETWORK, BAD],
        state_names=["good", "bad"],
        slowdown=slowdown,
        deadline_ms=deadline_ms,
    )


def useful(table):
    return table.set_index("rule")["useful"]


def test_watching_the_network_beats_any_fixed_option():
    # At 3x with 16 ms: the int8 final exit (9 ms) is right on images 0 to 2,
    # and offloading (15 ms) is right on all four but only on the good network.
    # Images 0 and 1 meet the bad network, images 2 and 3 the good one.
    rungs = ladder(scenario(measurements(), [1, 1, 0, 0], slowdown=3.0, deadline_ms=16))
    scores = useful(rungs)

    assert scores["1. best fixed option"] == 0.75
    assert scores["2. best option per network"] == 1.0
    assert rungs.set_index("rule").loc["2. best option per network", "plans"] == (
        "good: offload; bad: int8 to exit 2"
    )


def test_confidence_helps_when_a_deeper_exit_is_wrong():
    # Image 0 is right and sure at the first exit and wrong afterwards; image 1
    # is right only at the final exit. No single exit gets both.
    table = measurements().subset(np.array([0, 1]))
    correct = np.array([[True, False, False], [False, False, True]])
    table = dataclasses.replace(
        table,
        correct={"float32": correct, "int8": correct},
        confidence={"float32": CONFIDENCE[:2], "int8": CONFIDENCE[:2]},
        photo_correct=np.array([False, False]),
        features_correct={
            "float32": np.zeros((2, 2), dtype=bool),
            "uint8_zstd": np.zeros((2, 2), dtype=bool),
        },
    )
    scores = useful(ladder(scenario(table, [1, 1], slowdown=1.0, deadline_ms=20)))

    assert scores["2. best option per network"] == 0.5
    assert scores["3. best confidence rule per network"] == 1.0


def test_the_best_possible_needs_the_answer_key():
    # At 3x with 7 ms on the bad network only int8 exits 1 and 2 fit. Exit 2
    # is right on images 0 and 1, and nothing in time is right on 2 and 3.
    scores = useful(ladder(scenario(measurements(), [1, 1, 1, 1], slowdown=3.0, deadline_ms=7)))

    assert scores["4. best possible per frame"] == 0.5


def random_table(rng, images=40):
    def column(low, high, shape):
        return rng.uniform(low, high, shape)

    def exits():
        return np.cumsum(column(1, 5, (images, 3)), axis=1)

    def flags(shape):
        return rng.random(shape) < 0.6

    return Measurements(
        images=np.arange(images),
        labels=np.zeros(images, dtype=int),
        device_ms={"float32": exits(), "int8": exits()},
        confidence={"float32": rng.random((images, 3)), "int8": rng.random((images, 3))},
        correct={"float32": flags((images, 3)), "int8": flags((images, 3))},
        photo_bytes=column(500, 5000, images),
        photo_server_ms=column(1, 6, images),
        photo_correct=flags(images),
        features_bytes={p: column(500, 9000, (images, 2)) for p in ("float32", "uint8_zstd")},
        features_prepare_ms={p: column(0, 1, (images, 2)) for p in ("float32", "uint8_zstd")},
        features_server_ms={p: column(1, 4, (images, 2)) for p in ("float32", "uint8_zstd")},
        features_correct={p: flags((images, 2)) for p in ("float32", "uint8_zstd")},
    )


def test_the_rungs_never_cross_and_no_rule_beats_the_top():
    rng = np.random.default_rng(0)
    for trial in range(8):
        table = random_table(rng)
        states = markov_states(len(table), (0.7, 0.7), seed=trial)
        run = scenario(
            table,
            states,
            slowdown=float(rng.choice([1, 2, 4])),
            deadline_ms=float(rng.choice([8, 15, 30, 120])),
        )
        scores = ladder(run, thresholds=(0.3, 0.6))["useful"].to_numpy()
        assert (np.diff(scores) >= 0).all(), scores

        plans = plan_family((0.3, 0.6))
        for plan in [plans[i] for i in rng.choice(len(plans), 12, replace=False)]:
            trace = simulate(Always("one plan", plan), run)
            assert score_trace(trace, run.deadline_ms)["useful"] <= scores[-1]
