"""Checks for carrying out a plan, the frame stream, the network and what a rule sees.

Expected values come from the table in `builders.py`.
"""

import dataclasses
import math

import numpy as np
import pytest

from builders import NETWORK, measurements
from edge_inference.offload import Network
from edge_inference.simulate import (
    FEATURES,
    FINAL,
    PHOTO,
    Feedback,
    Measurements,
    Plan,
    Scenario,
    View,
    frame_stream,
    markov_states,
    resolve,
    score_trace,
    simulate,
    split_images,
)
from test_offload import components

ROWS = np.arange(4)
OTHER = Network(rtt_ms=100, uplink_mbps=8)


def run(plan, slowdown=1.0):
    return resolve(measurements(), plan, ROWS, NETWORK.rtt_ms, NETWORK.uplink_mbps, slowdown)


def test_the_table_is_reshaped_per_image():
    table = Measurements.from_components(components(), "t4")

    assert list(table.images) == [0, 1]
    assert table.device_ms["float32"].tolist() == [[2.0, 4.0, 6.0]] * 2
    assert table.device_ms["int8"].tolist() == [[1.0, 2.0, 3.0]] * 2
    assert table.photo_bytes.tolist() == [1000, 1000]
    assert table.photo_server_ms.tolist() == [4.0, 4.0]
    assert table.features_bytes["uint8_zstd"].tolist() == [[500, 500]] * 2
    assert table.features_server_ms["float32"].tolist() == [[1.5, 1.5]] * 2
    # The server gets image 1 wrong after an 8-bit payload.
    assert table.features_correct["uint8_zstd"].tolist() == [[True, True], [False, False]]


def test_a_frame_stops_at_the_first_confident_exit():
    outcome = run(Plan("float32", FINAL, (0.5, 0.5)))

    assert outcome.answered_at.tolist() == [0, 1, 2, 2]
    assert outcome.total_ms.tolist() == [2.0, 4.0, 6.0, 6.0]
    # Image 3 is never confident, and the final exit answers it anyway.
    assert outcome.correct.tolist() == [True, True, True, False]
    assert outcome.sent_bytes.tolist() == [0, 0, 0, 0]


def test_a_missing_threshold_means_never_stop_there():
    outcome = run(Plan("int8", 1))

    assert outcome.answered_at.tolist() == [1, 1, 1, 1]
    assert outcome.total_ms.tolist() == [2.0] * 4


def test_sending_the_photo_after_a_failed_check():
    outcome = run(Plan("float32", 0, (0.5,), PHOTO), slowdown=3.0)

    # Image 0 answers at the first exit: 2 ms, three times slower.
    # The rest add the photo's 15 ms to the 6 ms already spent.
    assert outcome.total_ms.tolist() == [6.0, 21.0, 21.0, 21.0]
    assert outcome.device_busy_ms.tolist() == [6.0] * 4
    assert outcome.answered_at.tolist() == [0, -1, -1, -1]
    assert outcome.sent_bytes.tolist() == [0, 1000, 1000, 1000]
    assert outcome.correct.tolist() == [True] * 4


def test_sending_features_after_a_failed_check():
    outcome = run(Plan("float32", 1, (math.inf, 0.5), FEATURES))

    # Images 0 and 1 are confident at the second exit, 4 ms in. The others
    # pack for 0.5 ms, send 500 bytes (10 + 0.5 ms) and wait 1.5 ms.
    assert outcome.answered_at.tolist() == [1, 1, -1, -1]
    assert outcome.total_ms.tolist() == [4.0, 4.0, 16.5, 16.5]
    assert outcome.device_busy_ms.tolist() == [4.0, 4.0, 4.5, 4.5]


def test_offloading_ignores_the_device():
    outcome = run(Plan(), slowdown=5.0)

    assert outcome.total_ms.tolist() == [15.0] * 4
    assert outcome.device_busy_ms.tolist() == [0.0] * 4
    assert outcome.answered_at.tolist() == [-1] * 4


def test_only_reached_exits_report_a_confidence():
    seen = run(Plan("float32", FINAL, (0.5, 0.5))).confidences

    assert seen[0, 0] == 0.9 and np.isnan(seen[0, 1:]).all()
    assert seen[1, :2].tolist() == [0.2, 0.8] and np.isnan(seen[1, 2])
    assert not np.isnan(seen[3]).any()
    assert np.isnan(run(Plan()).confidences).all()


def test_plans_the_profiles_cannot_answer_are_refused():
    with pytest.raises(ValueError):
        Plan("int8", 0, (0.5,), FEATURES)
    with pytest.raises(ValueError):
        Plan("float32", FINAL + 1)


def test_plans_describe_themselves():
    assert Plan().describe() == "offload"
    assert Plan("int8").describe() == "int8 to final exit"
    assert Plan("int8", 0).describe() == "int8 to exit 1"
    assert Plan("float32", FINAL, (0.5, 0.6)).describe() == "float32 to final exit, stop at 0.5/0.6"
    assert Plan("float32", 0, (0.5,), PHOTO).describe() == (
        "float32 to exit 1, stop at 0.5, else send photo"
    )


def test_the_network_chain_keeps_or_switches_state():
    assert markov_states(6, (1.0, 1.0), seed=0).tolist() == [0] * 6
    assert markov_states(6, (0.0, 0.0), seed=0).tolist() == [0, 1, 0, 1, 0, 1]
    assert (markov_states(500, (0.9, 0.9), 3) == markov_states(500, (0.9, 0.9), 3)).all()
    # Equal chances of staying give each state about half the frames.
    assert 0.4 < markov_states(20000, (0.9, 0.9), 0).mean() < 0.6


def test_the_stream_shows_every_image_before_repeating():
    stream = frame_stream(images=5, frames=12, seed=0)

    assert len(stream) == 12
    assert sorted(stream[:5]) == sorted(stream[5:10]) == [0, 1, 2, 3, 4]


def test_tuning_images_are_a_share_of_each_class():
    labels = np.repeat([0, 1, 2], 10)
    tuning = split_images(labels, tune_share=0.3, seed=0)

    assert [int(tuning[labels == label].sum()) for label in range(3)] == [3, 3, 3]
    assert (tuning == split_images(labels, tune_share=0.3, seed=0)).all()


class Recorder:
    """Offloads on the frames listed, runs locally otherwise, and keeps what it saw."""

    name = "recorder"

    def __init__(self, send_on=()):
        self.send_on = set(send_on)
        self.views, self.feedback = [], []

    def choose(self, view):
        self.views.append(view)
        return Plan() if view.frame in self.send_on else Plan("int8")

    def observe(self, feedback):
        self.feedback.append(feedback)


def scenario(states=(0, 0, 1, 1, 1, 0)):
    return Scenario(
        measurements(),
        frames=np.arange(len(states)) % 4,
        states=np.array(states),
        networks=[NETWORK, OTHER],
        state_names=["good", "bad"],
        slowdown=1.0,
        deadline_ms=20,
    )


def test_exact_knowledge_shows_the_current_network():
    rule = Recorder()
    simulate(rule, scenario(), "exact")

    assert [view.network for view in rule.views] == [NETWORK, NETWORK, OTHER, OTHER, OTHER, NETWORK]
    assert [view.network_age for view in rule.views] == [0] * 6


def test_a_probe_goes_stale_between_probes():
    rule = Recorder()
    simulate(rule, scenario(), "probe:3")

    # Probed at frames 0 and 3, where the network was good and then bad.
    assert [view.network for view in rule.views] == [NETWORK] * 3 + [OTHER] * 3
    assert [view.network_age for view in rule.views] == [0, 1, 2, 0, 1, 2]


def test_only_a_send_reveals_the_network():
    rule = Recorder(send_on={2})
    simulate(rule, scenario(), "sends")

    # Nothing is known until the send on frame 2, which met the bad network.
    assert [view.network for view in rule.views] == [None] * 3 + [OTHER] * 3
    assert [view.network_age for view in rule.views[3:]] == [1, 2, 3]


def test_unknown_knowledge_is_refused():
    with pytest.raises(ValueError):
        simulate(Recorder(), scenario(), "guess")


def test_a_rule_is_never_told_whether_it_was_right():
    for shown in (View, Feedback):
        assert not any("correct" in f.name for f in dataclasses.fields(shown))

    rule = Recorder(send_on={2})
    simulate(rule, scenario(), "exact")
    sent = rule.feedback[2]
    assert sent.sent and sent.answered_at == -1 and sent.total_ms == 105.0


def test_the_trace_scores_correct_and_in_time():
    # Frames show images 0, 1, 2, 3, 0, 1. Offloading takes 15 ms on the good
    # network and 105 ms on the bad one, against a 20 ms deadline.
    trace = simulate(Recorder(send_on=range(6)), scenario(), "exact")
    scores = score_trace(trace, deadline_ms=20)

    assert trace["image"].tolist() == [100, 101, 102, 103, 100, 101]
    assert trace["network"].tolist() == ["good", "good", "bad", "bad", "bad", "good"]
    assert trace["useful"].tolist() == [True, True, False, False, False, True]
    assert scores["useful"] == 0.5
    assert scores["accuracy"] == 1.0
    assert scores["sent_share"] == 1.0
