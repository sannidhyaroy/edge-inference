"""Checks for the live runtime, with stand-ins for the stage graphs.

A stand-in stage adds one to its input and reports logits that are set per
test, so every expected value can be read off the test itself. The server runs
in a thread of the test process on a port the system picks.
"""

import socket
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from builders import measurements
from edge_inference.export import FEATURES_NAME, INPUT_NAME, OUTPUT_NAME
from edge_inference.offload import Network
from edge_inference.policies import Always
from edge_inference.runtime import (
    Device,
    LiveRun,
    Responder,
    answer,
    compare,
    connect,
    nearest_network,
    receive_message,
    send_message,
    serve,
)
from edge_inference.simulate import FEATURES, FINAL, PHOTO, Plan, Scenario, simulate

GOOD = Network(rtt_ms=10, uplink_mbps=8)
BAD = Network(rtt_ms=100, uplink_mbps=8)
SURE, UNSURE = [9.0, 0.0], [0.1, 0.0]


class Stage:
    """Adds one to its input; its head reports the logits it was given."""

    def __init__(self, logits, features=True):
        self.logits = np.array([logits], dtype=np.float32)
        self.names = ([FEATURES_NAME] if features else []) + [OUTPUT_NAME]

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in self.names]

    def run(self, names, feeds):
        outputs = {FEATURES_NAME: feeds[INPUT_NAME] + 1.0, OUTPUT_NAME: self.logits}
        return [outputs[name] for name in names]


def stages(first=UNSURE, second=UNSURE, final=SURE):
    return [Stage(first), Stage(second), Stage(final, features=False)]


@pytest.fixture
def server():
    """A server on a free port whose every stage is sure of class 0."""
    ready, stop, port = threading.Event(), threading.Event(), []
    responder = Responder(
        stages(SURE, SURE, SURE), decode=lambda body: np.zeros((1, 4), np.float32)
    )

    def started(bound):
        port.append(bound)
        ready.set()

    thread = threading.Thread(
        target=serve, args=(responder, "127.0.0.1", 0), kwargs={"on_ready": started, "stop": stop}
    )
    thread.start()
    ready.wait(5)
    yield port[0]
    stop.set()
    thread.join(5)


def device(port, sessions=None, **settings):
    waits = []
    made = Device(
        {"float32": sessions or stages(), "int8": sessions or stages()},
        connect("127.0.0.1", port),
        sleep=waits.append,
        **settings,
    )
    return made, waits


X = np.zeros((1, 4), dtype=np.float32)


def test_a_message_survives_the_wire():
    near, far = socket.socketpair()
    with near, far:
        send_message(near, {"kind": "photo", "n": 3}, b"\x00\x01\x02" * 1000)
        header, body = receive_message(far)

    assert header == {"kind": "photo", "n": 3}
    assert body == b"\x00\x01\x02" * 1000


def test_a_closed_connection_is_reported():
    near, far = socket.socketpair()
    near.close()
    with far, pytest.raises(ConnectionError):
        receive_message(far)


def test_a_head_answers_with_its_most_likely_class():
    predicted, confidence = answer(np.array([0.0, np.log(3.0)]))

    assert predicted == 1
    assert confidence == pytest.approx(0.75)


def test_the_server_finishes_from_where_the_device_stopped():
    seen = []

    class Recording(Stage):
        def run(self, names, feeds):
            seen.append(float(feeds[INPUT_NAME].flat[0]))
            return super().run(names, feeds)

    responder = Responder([Recording(SURE), Recording(SURE), Recording(SURE, features=False)])
    features = np.full((1, 4), 5.0, dtype=np.float32)
    reply = responder.reply(
        {"kind": "features", "point": 0, "payload": "float32", "shape": [1, 4]}, features.tobytes()
    )

    # Sent after the first exit, so the server runs stages two and three only.
    assert seen == [5.0, 6.0]
    assert reply["predicted"] == 0 and reply["confidence"] > 0.99
    assert responder.reply({"kind": "ping"}, b"") == {"server_ms": 0.0}


def test_a_confident_exit_answers_on_the_device(server):
    made, _ = device(server, stages(first=SURE))
    measured = made.run(Plan("int8", FINAL, (0.5, 0.5)), X, b"photo")

    assert measured.answered_at == 0 and measured.sent_bytes == 0
    assert measured.confidences[0] > 0.99 and np.isnan(measured.confidences[1])
    assert np.isnan(measured.server_confidence)


def test_the_final_exit_answers_even_when_unsure(server):
    made, _ = device(server, stages(final=UNSURE))
    measured = made.run(Plan("float32", FINAL, (0.9, 0.9)), X, b"photo")

    assert measured.answered_at == FINAL
    assert not np.isnan(measured.confidences).any()


def test_an_unsure_exit_sends_the_photo(server):
    made, _ = device(server)
    measured = made.run(Plan("int8", 0, (0.9,), PHOTO), X, b"p" * 1234)

    assert measured.answered_at == -1 and measured.sent_bytes == 1234
    assert measured.predicted == 0 and measured.server_confidence > 0.99
    assert measured.exchange_ms >= measured.server_ms > 0


def test_an_unsure_exit_sends_its_features(server):
    made, _ = device(server)
    raw = made.run(Plan("float32", 1, (9.0, 0.9), FEATURES, "float32"), X, b"photo")
    packed = made.run(Plan("float32", 1, (9.0, 0.9), FEATURES, "uint8_zstd"), X, b"photo")

    # Four float32 values raw; the same map packed is far smaller than that.
    assert raw.sent_bytes == 16
    assert raw.answered_at == packed.answered_at == -1
    assert raw.predicted == packed.predicted == 0


def test_offloading_runs_nothing_on_the_device(server):
    made, _ = device(server)
    measured = made.run(Plan(), X, b"photo")

    assert measured.device_busy_ms == 0.0 and measured.answered_at == -1
    assert np.isnan(measured.confidences).all()


def test_a_slower_device_waits_out_the_difference(server):
    made, waits = device(server, stages(first=SURE), slowdown=4.0)
    measured = made.run(Plan("int8", 0), X, b"photo")

    # One stage ran, so one wait, three times as long as the stage took.
    assert len(waits) == 1
    assert measured.device_busy_ms == pytest.approx(waits[0] * 1000 * 4 / 3)


def test_a_shaped_link_waits_before_each_send(server):
    made, waits = device(server, link=lambda size: float(GOOD.transfer_ms(size)))
    made.run(Plan(), X, b"p" * 1000)

    # 10 ms round trip plus 1000 bytes at 8 Mbps, in seconds.
    assert waits == [pytest.approx(0.011)]


def test_a_transfer_is_matched_to_the_nearest_network():
    networks = [GOOD, BAD]

    # 1000 bytes should take 11 ms on the good network and 101 ms on the bad one.
    assert nearest_network(networks, 1000, 14.0) == 0
    assert nearest_network(networks, 1000, 90.0) == 1


def test_a_live_run_reuses_the_simulators_loop(server):
    table = measurements()
    scenario = Scenario(
        table,
        frames=np.array([0, 1, 2, 3]),
        states=np.array([0, 0, 1, 1]),
        networks=[GOOD, BAD],
        state_names=["good", "bad"],
        slowdown=1.0,
        deadline_ms=1000.0,
    )
    made, waits = device(server)
    shown = []
    live = LiveRun(
        made,
        scenario,
        images=lambda image: (X, 0 if image in (100, 101) else 1),
        photos=lambda image: b"p" * 1000,
        on_frame=shown.append,
    )
    rule = Always("always offload", Plan())
    carried_out = simulate(rule, scenario, execute=live)
    looked_up = simulate(rule, scenario)

    # The server always says class 0, which is right for the first two images.
    assert carried_out["correct"].tolist() == [True, True, False, False]
    assert carried_out["image"].tolist() == [100, 101, 102, 103]
    # Each send waited for its frame's network: 11 ms twice, then 101 ms twice.
    assert waits == [pytest.approx(v) for v in (0.011, 0.011, 0.101, 0.101)]
    assert [row["network_guessed"] for row in live.log] == ["good"] * 4
    # Every frame is also reported as it happens, for watching a run.
    assert [(row["network"], row["correct"]) for row in shown] == [
        ("good", True),
        ("good", True),
        ("bad", False),
        ("bad", False),
    ]

    table = compare(looked_up, carried_out, scenario.deadline_ms).set_index("run")
    assert table.loc["live", "same_plan"] == 1.0
    assert table.loc["simulated", "accuracy"] == 1.0 and table.loc["live", "accuracy"] == 0.5
