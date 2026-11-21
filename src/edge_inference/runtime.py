"""Carry a decision out for real: a device process, a server process, real sockets.

The simulator looks every outcome up. Here a rule's plan is executed. The
device runs the stage graphs it was told to run, checks each exit's confidence,
and when the plan says so really sends the photo or the feature map over a TCP
connection to a server that really finishes the network and answers.

Two things are emulated, and labelled as such wherever they are reported:

* **A slower device.** After each stage the device waits for `slowdown - 1`
  times as long as the stage took. The work is real and the pace is stretched.
* **The network, when both ends share a machine.** Loopback delivers in a
  fraction of a millisecond, so before each send the device waits for the round
  trip and transfer time of the network it is meant to be on. Against a server
  on another machine the link is real and nothing is added.

What is not counted, as in the profiles: turning the stored image into the
tensor the device's network takes. The server's decoding of a photo it
receives is counted, as it is in the server profile.

Messages are a small header and a body, each prefixed with its length. The
device and the server run at different thread counts, and a process has one
shared thread pool, so they are always two processes.
"""

import json
import socket
import struct
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from compression import zstd
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort
import pandas as pd

from edge_inference.export import FEATURES_NAME, OUTPUT_NAME, run_stage
from edge_inference.models import EXIT_NAMES
from edge_inference.offload import Network
from edge_inference.server import _decode_jpeg, _receive_float32, _receive_uint8_zstd
from edge_inference.simulate import (
    ANSWER,
    FEATURES,
    PHOTO,
    Outcome,
    Plan,
    Scenario,
    score_trace,
)
from edge_inference.split import quantize_uint8

_LENGTHS = struct.Struct("!II")
ZSTD_LEVEL = 3


def send_message(connection: socket.socket, header: dict, body: bytes = b"") -> None:
    """Write one message: two lengths, a JSON header, then the body."""
    head = json.dumps(header).encode()
    connection.sendall(_LENGTHS.pack(len(head), len(body)) + head + body)


def _exactly(connection: socket.socket, count: int) -> bytes:
    chunks = []
    while count:
        chunk = connection.recv(count)
        if not chunk:
            raise ConnectionError("the other end closed the connection")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


def receive_message(connection: socket.socket) -> tuple[dict, bytes]:
    """Read one message written by `send_message`."""
    head_length, body_length = _LENGTHS.unpack(_exactly(connection, _LENGTHS.size))
    header = json.loads(_exactly(connection, head_length))
    return header, _exactly(connection, body_length)


def answer(logits: np.ndarray) -> tuple[int, float]:
    """The class a head picks and the probability it gives that class."""
    exponentials = np.exp(logits - logits.max())
    probabilities = exponentials / exponentials.sum()
    predicted = int(probabilities.argmax())
    return predicted, float(probabilities[predicted])


@dataclass
class Responder:
    """The server's side of a request: receive, finish the network, answer.

    `sessions` are the float32 stage graphs. A photo starts at the first; a
    feature map starts at the stage after the one that produced it.
    """

    sessions: Sequence[ort.InferenceSession]
    decode: Callable[[bytes], np.ndarray] = _decode_jpeg

    def reply(self, header: dict, body: bytes) -> dict:
        if header["kind"] == "ping":
            return {"server_ms": 0.0}
        start = time.perf_counter()
        if header["kind"] == "photo":
            x, stage = self.decode(body), 0
        else:
            shape, stage = tuple(header["shape"]), header["point"] + 1
            if header["payload"] == "float32":
                x = _receive_float32(body, shape)
            else:
                x = _receive_uint8_zstd(body, shape, header["scale"], header["zero"])
        for session in self.sessions[stage:]:
            outputs = run_stage(session, x)
            x = outputs.get(FEATURES_NAME)
        predicted, confidence = answer(outputs[OUTPUT_NAME][0])
        return {
            "predicted": predicted,
            "confidence": confidence,
            "server_ms": (time.perf_counter() - start) * 1000.0,
        }


def serve(
    responder: Responder,
    host: str,
    port: int,
    *,
    on_ready: Callable[[int], None] | None = None,
    stop: threading.Event | None = None,
) -> None:
    """Answer requests until `stop` is set, one connection at a time.

    `on_ready` is told the port actually bound, which matters when `port` is 0
    and the system picks one.
    """
    stop = stop or threading.Event()
    with socket.create_server((host, port)) as listener:
        listener.settimeout(0.2)
        if on_ready is not None:
            on_ready(listener.getsockname()[1])
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            with connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while not stop.is_set():
                    try:
                        header, body = receive_message(connection)
                    except ConnectionError:
                        break
                    send_message(connection, responder.reply(header, body))


def connect(host: str, port: int) -> socket.socket:
    """Open the device's connection to a server, with small sends not held back."""
    connection = socket.create_connection((host, port))
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return connection


@dataclass(frozen=True)
class Measured:
    """What one frame did, as measured."""

    total_ms: float
    predicted: int
    device_busy_ms: float
    sent_bytes: int
    answered_at: int
    confidences: tuple[float, ...]
    server_confidence: float
    exchange_ms: float
    server_ms: float


@dataclass
class Device:
    """The device's side: run stages, check confidence, send when the plan says so.

    `sessions` maps a precision to its stage graphs. `link`, when given, turns
    the bytes about to be sent into the milliseconds to wait first; leave it
    out when the link to the server is real.
    """

    sessions: Mapping[str, Sequence[ort.InferenceSession]]
    connection: socket.socket
    slowdown: float = 1.0
    link: Callable[[int], float] | None = None
    sleep: Callable[[float], None] = time.sleep

    def exchange(self, header: dict, body: bytes = b"") -> tuple[dict, float]:
        """Send a request and wait for its reply; return the reply and the milliseconds taken."""
        start = time.perf_counter()
        if self.link is not None:
            self.sleep(self.link(len(body)) / 1000.0)
        send_message(self.connection, header, body)
        reply, _ = receive_message(self.connection)
        return reply, (time.perf_counter() - start) * 1000.0

    def _paced(self, work: Callable[[], object]) -> tuple[object, float]:
        """Do `work`, then wait so that it took `slowdown` times as long. Returns its time."""
        start = time.perf_counter()
        result = work()
        took = time.perf_counter() - start
        if self.slowdown > 1.0:
            self.sleep((self.slowdown - 1.0) * took)
        return result, self.slowdown * took * 1000.0

    def run(self, plan: Plan, x: np.ndarray, photo: bytes) -> Measured:
        """Carry out `plan` on one frame: `x` for the device's network, `photo` to send."""
        start = time.perf_counter()
        confidences = [float("nan")] * len(EXIT_NAMES)
        busy = 0.0

        def finish(predicted: int, answered_at: int, reply: dict | None, sent: int, took: float):
            return Measured(
                total_ms=(time.perf_counter() - start) * 1000.0,
                predicted=predicted,
                device_busy_ms=busy,
                sent_bytes=sent,
                answered_at=answered_at,
                confidences=tuple(confidences),
                server_confidence=float("nan") if reply is None else reply["confidence"],
                exchange_ms=took,
                server_ms=0.0 if reply is None else reply["server_ms"],
            )

        def send_photo():
            reply, took = self.exchange({"kind": "photo"}, photo)
            return finish(reply["predicted"], -1, reply, len(photo), took)

        if plan.precision is None:
            return send_photo()

        final = len(self.sessions[plan.precision]) - 1
        target = final if plan.depth is None else plan.depth
        thresholds = (*plan.thresholds, *[float("inf")] * (target + 1))
        for depth in range(target + 1):
            outputs, took = self._paced(
                lambda d=depth, x=x: run_stage(self.sessions[plan.precision][d], x)
            )
            busy += took
            x = outputs.get(FEATURES_NAME)
            predicted, confidences[depth] = answer(outputs[OUTPUT_NAME][0])
            last = depth == target
            if confidences[depth] >= thresholds[depth] or (
                last and (depth == final or plan.otherwise == ANSWER)
            ):
                return finish(predicted, depth, None, 0, 0.0)

        if plan.otherwise == PHOTO:
            return send_photo()
        assert plan.otherwise == FEATURES
        header = {"kind": "features", "point": plan.depth, "payload": plan.payload}
        header["shape"] = list(x.shape)
        if plan.payload == "float32":
            body = x.tobytes()
        else:

            def pack() -> tuple[bytes, float, int]:
                levels, scale, zero = quantize_uint8(x)
                return zstd.compress(levels.tobytes(), ZSTD_LEVEL), scale, zero

            (body, header["scale"], header["zero"]), took = self._paced(pack)
            busy += took
        reply, took = self.exchange(header, body)
        return finish(reply["predicted"], -1, reply, len(body), took)


def nearest_network(networks: Sequence[Network], sent_bytes: int, transfer_ms: float) -> int:
    """Which known network a measured transfer most resembles.

    A device never reads "Wi-Fi" or "4G" off the air. It sees how long a send
    of a known size took, and the nearest known network is the one whose
    arithmetic comes closest to that.
    """
    expected = [float(network.transfer_ms(sent_bytes)) for network in networks]
    return int(np.argmin([abs(transfer_ms - value) for value in expected]))


@dataclass
class LiveRun:
    """Executes plans on a scenario's frames for real, in place of the lookup.

    Pass it to `simulate` as `execute`. `images` gives the device's tensor and
    the label for an image id, and `photos` the bytes of its stored file.
    With `shaped`, each send waits for the network the scenario puts that frame
    on; without, the link to the server is taken as it is.
    """

    device: Device
    scenario: Scenario
    images: Callable[[int], tuple[np.ndarray, int]]
    photos: Callable[[int], bytes]
    shaped: bool = True
    on_frame: Callable[[dict[str, object]], None] | None = None
    log: list[dict[str, object]] = field(default_factory=list)

    def __call__(self, plan: Plan, frame: int) -> Outcome:
        scenario = self.scenario
        image = int(scenario.measurements.images[scenario.frames[frame]])
        network = scenario.networks[scenario.states[frame]]
        self.device.link = (lambda size: float(network.transfer_ms(size))) if self.shaped else None
        x, label = self.images(image)
        measured = self.device.run(plan, x, self.photos(image))

        guessed = None
        if measured.sent_bytes:
            transfer = measured.exchange_ms - measured.server_ms
            guessed = nearest_network(scenario.networks, measured.sent_bytes, transfer)
        self.log.append(
            {
                "frame": frame,
                "exchange_ms": measured.exchange_ms,
                "server_ms": measured.server_ms,
                "network_guessed": None if guessed is None else scenario.state_names[guessed],
            }
        )
        if self.on_frame is not None:
            self.on_frame(
                {
                    "frame": frame,
                    "network": scenario.state_names[scenario.states[frame]],
                    "plan": plan.describe(),
                    "total_ms": measured.total_ms,
                    "correct": measured.predicted == label,
                    "in_time": measured.total_ms <= scenario.deadline_ms,
                }
            )
        sent = measured.sent_bytes > 0
        right = bool(measured.predicted == label)
        return Outcome(
            total_ms=np.array([measured.total_ms]),
            correct=np.array([right]),
            device_busy_ms=np.array([measured.device_busy_ms]),
            sent_bytes=np.array([float(measured.sent_bytes)]),
            answered_at=np.array([measured.answered_at]),
            confidences=np.array([measured.confidences]),
            server_confidence=np.array([measured.server_confidence]),
            # One answer per frame here: the device's, or the server's once it
            # arrived. A hedged run that keeps both is the protocol's next step.
            local_ms=np.array([np.inf if sent else measured.total_ms]),
            local_correct=np.array([not sent and right]),
            server_at_ms=np.array([measured.total_ms if sent else np.inf]),
            server_correct=np.array([sent and right]),
        )


def compare(simulated: pd.DataFrame, live: pd.DataFrame, deadline_ms: float) -> pd.DataFrame:
    """The same frames looked up and carried out, side by side.

    Besides both scoreboards' columns, it reports how often the two runs
    played the same plan and gave the same answer, and how far apart their
    times were frame by frame.
    """
    rows = pd.DataFrame(
        {"simulated": score_trace(simulated, deadline_ms), "live": score_trace(live, deadline_ms)}
    ).T
    gap = live["total_ms"].to_numpy() - simulated["total_ms"].to_numpy()
    rows["same_plan"] = float((simulated["plan"] == live["plan"]).mean())
    rows["same_answer"] = float((simulated["correct"] == live["correct"]).mean())
    rows["median_gap_ms"] = float(np.median(gap))
    rows["p95_abs_gap_ms"] = float(np.percentile(np.abs(gap), 95))
    return rows.rename_axis("run").reset_index()
