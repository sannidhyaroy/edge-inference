"""The device runs a plan for real: locally, by sending, or both at once."""

import threading
import time

import numpy as np
import pytest
from PIL import Image

from edge_inference.device import Device, PhotoRoute, Variant
from edge_inference.preprocess import Preprocess
from edge_inference.simulate import AT_ONCE, NEVER, Plan
from edge_inference.wire import Connection, serve

FRAME = Image.fromarray(np.full((24, 32, 3), 120, dtype=np.uint8))
RECIPE = Preprocess(16, 18, "bilinear", (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
RIGHT = 3


def slow_infer(seconds):
    """A local run that takes `seconds`, gives up if told to stop, and answers RIGHT."""

    def infer(variant, x, options):
        until = time.perf_counter() + seconds
        while time.perf_counter() < until:
            if options.terminate:
                raise RuntimeError("terminated")
            time.sleep(0.001)
        logits = np.zeros(5)
        logits[RIGHT] = 5.0
        return logits

    return infer


@pytest.fixture
def server():
    """Answers every photo with class RIGHT after 2 ms."""
    ready, stop, port = threading.Event(), threading.Event(), []

    def handle(header, body):
        time.sleep(0.002)
        return {"predicted": RIGHT, "confidence": 0.9, "server_ms": 2.0}

    thread = threading.Thread(
        target=serve,
        args=(handle, "127.0.0.1", 0),
        kwargs={"on_ready": lambda p: (port.append(p), ready.set()), "stop": stop},
    )
    thread.start()
    ready.wait(5)
    yield port[0]
    stop.set()
    thread.join()


def device(port, local_seconds, link_ms):
    connection = Connection.open("127.0.0.1", port, link=lambda size: link_ms)
    return Device(
        variants={"v": Variant(session=None, recipe=RECIPE)},
        routes={"photo": PhotoRoute(shorter=20, quality=90, model="big")},
        connection=connection,
        infer=slow_infer(local_seconds),
    )


def test_a_local_plan_answers_from_the_device(server):
    made = device(server, 0.01, 1000)
    frame = made.run(Plan("v"), FRAME, RIGHT, deadline_ms=100, horizon_ms=200)
    made.connection.close()
    assert frame.answered_by == "local" and frame.correct
    assert frame.sent_bytes == 0 and np.isinf(frame.server_at_ms)
    assert frame.total_ms == frame.local_ms >= 10


def test_a_hedge_takes_the_server_when_it_arrives_in_time(server):
    made = device(server, 0.08, 10)
    frame = made.run(
        Plan("v", route="photo", hedge=True), FRAME, RIGHT, deadline_ms=60, horizon_ms=200
    )
    made.connection.close()
    assert frame.answered_by == "server" and frame.total_ms == frame.server_at_ms < 60
    assert frame.sent_bytes > 0 and frame.local_ms >= 80


def test_a_hedge_answers_locally_at_the_deadline_when_the_server_is_late(server):
    made = device(server, 0.01, 120)
    frame = made.run(
        Plan("v", route="photo", hedge=True), FRAME, RIGHT, deadline_ms=50, horizon_ms=400
    )
    made.connection.close()
    assert frame.answered_by == "local" and frame.total_ms == 50
    # The late reply is still caught before the horizon, for the record.
    assert 120 <= frame.server_at_ms < 400 and frame.server_correct


def test_cancelling_now_stops_the_local_run_when_the_answer_lands(server):
    made = device(server, 0.3, 20)
    plan = Plan("v", route="photo", hedge=True)
    frame = made.run(plan, FRAME, RIGHT, deadline_ms=400, horizon_ms=500, cancel=AT_ONCE)
    made.connection.close()
    assert frame.cancelled and frame.answered_by == "server"
    assert frame.device_busy_ms < 150 and np.isinf(frame.local_ms)


def test_without_cancelling_the_local_run_finishes(server):
    made = device(server, 0.1, 20)
    plan = Plan("v", route="photo", hedge=True)
    frame = made.run(plan, FRAME, RIGHT, deadline_ms=400, horizon_ms=500, cancel=NEVER)
    made.connection.close()
    assert not frame.cancelled and frame.device_busy_ms >= 100


def test_offload_waits_for_the_server(server):
    made = device(server, 0.01, 30)
    frame = made.run(Plan(route="photo"), FRAME, RIGHT, deadline_ms=10, horizon_ms=300)
    made.connection.close()
    assert frame.answered_by == "server" and frame.total_ms >= 30
    assert frame.device_busy_ms == frame.prepare_ms
