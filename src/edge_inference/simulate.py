"""Replay the measurements as a stream of frames under a changing network.

Nothing here runs a model. Every frame is one of the profiled images, and what
happens to it is looked up: how long the device takes to reach each exit, how
confident each exit is, how many bytes sending takes, and how long the server
needs. The network is the only thing not looked up. It is a sequence of states,
one per frame, and its cost is arithmetic on the bytes sent.

A rule decides what to do with each frame by returning a `Plan`. It sees a
`View` before deciding and a `Feedback` afterwards, and neither says whether
the answer was right: a device in the field has no labels. Only the scoring
here, and the yardsticks, read correctness.

**What a rule knows about the network is a setting**, because a device does
not learn the network for free. Sending a frame reveals it; running locally
reveals nothing.

* ``exact``: the network the frame is about to meet. An upper bound.
* ``probe:N``: the network as it was at the last probe, one every N frames.
* ``sends``: the network as it was the last time this rule sent anything, or
  nothing at all before its first send.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from edge_inference.models import EXIT_NAMES
from edge_inference.offload import Network
from edge_inference.split import SPLIT_NAMES

PRECISIONS = ("float32", "int8")
PAYLOADS = ("float32", "uint8_zstd")
FINAL = len(EXIT_NAMES) - 1

# What a device does at the deepest exit it runs to when that exit is not
# confident enough: answer anyway, send the feature map, or send the photo.
ANSWER, FEATURES, PHOTO = "answer", "features", "photo"


@dataclass(frozen=True)
class Measurements:
    """The per-image table for one server, as arrays with one row per image.

    Local arrays have a column per exit, shallowest first. Feature arrays have
    a column per split point. Feature maps come from the float32 device only,
    which is all the profiles measured.
    """

    images: np.ndarray
    labels: np.ndarray
    device_ms: dict[str, np.ndarray]
    confidence: dict[str, np.ndarray]
    correct: dict[str, np.ndarray]
    photo_bytes: np.ndarray
    photo_server_ms: np.ndarray
    photo_correct: np.ndarray
    features_bytes: dict[str, np.ndarray]
    features_prepare_ms: dict[str, np.ndarray]
    features_server_ms: dict[str, np.ndarray]
    features_correct: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.images)

    @classmethod
    def from_components(cls, components: pd.DataFrame, server: str) -> Measurements:
        """Reshape the rows for `server`, and the local rows, into arrays."""

        def wide(frame: pd.DataFrame, value: str, columns: Sequence[str]) -> np.ndarray:
            return frame.pivot(index="image", columns="point", values=value)[
                list(columns)
            ].to_numpy()

        components = components.assign(
            server_ms=components["server_receive_ms"] + components["server_compute_ms"]
        )
        local = components[components["mode"] == "local"]
        by_precision = {p: local[local["device_precision"] == p] for p in PRECISIONS}
        sent = components[components["server"] == server]
        photo = sent[sent["mode"] == "offload"].set_index("image").sort_index()
        by_payload = {p: sent[(sent["mode"] == "split") & (sent["payload"] == p)] for p in PAYLOADS}

        return cls(
            images=photo.index.to_numpy(),
            labels=photo["label"].to_numpy(),
            device_ms={p: wide(f, "device_ms", EXIT_NAMES) for p, f in by_precision.items()},
            confidence={p: wide(f, "confidence", EXIT_NAMES) for p, f in by_precision.items()},
            correct={p: wide(f, "correct", EXIT_NAMES) for p, f in by_precision.items()},
            photo_bytes=photo["payload_bytes"].to_numpy(),
            photo_server_ms=photo["server_ms"].to_numpy(),
            photo_correct=photo["correct"].to_numpy(),
            features_bytes={
                p: wide(f, "payload_bytes", SPLIT_NAMES) for p, f in by_payload.items()
            },
            features_prepare_ms={
                p: wide(f, "prepare_ms", SPLIT_NAMES) for p, f in by_payload.items()
            },
            features_server_ms={
                p: wide(f, "server_ms", SPLIT_NAMES) for p, f in by_payload.items()
            },
            features_correct={p: wide(f, "correct", SPLIT_NAMES) for p, f in by_payload.items()},
        )

    def subset(self, rows: np.ndarray) -> Measurements:
        """The same table restricted to `rows`, a boolean mask or an index array."""

        def take(value):
            if isinstance(value, dict):
                return {key: array[rows] for key, array in value.items()}
            return value[rows]

        return Measurements(
            **{name: take(getattr(self, name)) for name in self.__dataclass_fields__}
        )


@dataclass(frozen=True)
class Plan:
    """What to do with one frame.

    With no `precision` the photo is sent at once. Otherwise the device runs
    the network in that precision up to exit `depth`, counted from zero. On the
    way it stops and answers at the first exit whose confidence reaches that
    exit's entry in `thresholds`; a missing entry means never stop there. At
    `depth` itself, without enough confidence, it does `otherwise`. The final
    exit always answers: a network cannot decline to produce an output.
    """

    precision: str | None = None
    depth: int = FINAL
    thresholds: tuple[float, ...] = ()
    otherwise: str = ANSWER
    payload: str = "uint8_zstd"

    def __post_init__(self) -> None:
        if self.precision is None:
            return
        if not 0 <= self.depth <= FINAL:
            raise ValueError(f"depth must be between 0 and {FINAL}")
        if self.otherwise == FEATURES and (
            self.precision != "float32" or self.depth >= len(SPLIT_NAMES)
        ):
            raise ValueError("feature maps were profiled for float32 at the split points only")

    def describe(self) -> str:
        if self.precision is None:
            return "offload"
        padded = (*self.thresholds, *[math.inf] * (self.depth + 1))[: self.depth + 1]
        checks = self.depth + (self.otherwise != ANSWER and self.depth != FINAL)
        stops = [f"{t:g}" for t in padded[:checks] if t != math.inf]
        where = "final exit" if self.depth == FINAL else f"exit {self.depth + 1}"
        text = f"{self.precision} to {where}"
        if stops:
            text += f", stop at {'/'.join(stops)}"
        if self.depth != FINAL and self.otherwise == PHOTO:
            text += ", else send photo"
        if self.depth != FINAL and self.otherwise == FEATURES:
            text += f", else send {self.payload} features"
        return text


@dataclass(frozen=True)
class Outcome:
    """What happened to each frame; every field is an array with one entry per frame."""

    total_ms: np.ndarray
    correct: np.ndarray
    device_busy_ms: np.ndarray
    sent_bytes: np.ndarray
    # The exit that answered, counted from zero, or -1 when the server answered.
    answered_at: np.ndarray
    # Confidence at each exit the device reached; NaN for exits it did not.
    confidences: np.ndarray


def resolve(
    measurements: Measurements,
    plan: Plan,
    rows: np.ndarray,
    rtt_ms: np.ndarray | float,
    uplink_mbps: np.ndarray | float,
    slowdown: float = 1.0,
) -> Outcome:
    """Carry out `plan` on the images at `rows`, each over its own network."""
    rows = np.asarray(rows)
    count = len(rows)
    # `Network` only does arithmetic, so it takes one value per frame as well.
    network = Network(rtt_ms=rtt_ms, uplink_mbps=uplink_mbps)
    no_exit = np.full((count, len(EXIT_NAMES)), np.nan)

    if plan.precision is None:
        size = measurements.photo_bytes[rows]
        return Outcome(
            total_ms=network.transfer_ms(size) + measurements.photo_server_ms[rows],
            correct=measurements.photo_correct[rows],
            device_busy_ms=np.zeros(count),
            sent_bytes=size,
            answered_at=np.full(count, -1),
            confidences=no_exit,
        )

    depth = plan.depth
    reached = measurements.confidence[plan.precision][rows, : depth + 1]
    thresholds = np.array((*plan.thresholds, *[math.inf] * (depth + 1))[: depth + 1])
    confident = reached >= thresholds
    if depth == FINAL or plan.otherwise == ANSWER:
        confident[:, depth] = True
    answers = confident.any(axis=1)
    # argmax on booleans gives the first True; a row with none stops at `depth`.
    stopped = np.where(answers, confident.argmax(axis=1), depth)

    busy = slowdown * measurements.device_ms[plan.precision][rows, stopped]
    total = busy.copy()
    correct = measurements.correct[plan.precision][rows, stopped].copy()
    sent_bytes = np.zeros(count)
    answered_at = stopped.copy()

    sends = ~answers
    if sends.any():
        if plan.otherwise == PHOTO:
            size = measurements.photo_bytes[rows]
            prepare = np.zeros(count)
            server = measurements.photo_server_ms[rows]
            right = measurements.photo_correct[rows]
        else:
            size = measurements.features_bytes[plan.payload][rows, depth]
            prepare = slowdown * measurements.features_prepare_ms[plan.payload][rows, depth]
            server = measurements.features_server_ms[plan.payload][rows, depth]
            right = measurements.features_correct[plan.payload][rows, depth]
        busy = busy + np.where(sends, prepare, 0.0)
        total = busy + np.where(sends, network.transfer_ms(size) + server, 0.0)
        correct = np.where(sends, right, correct)
        sent_bytes = np.where(sends, size, 0.0)
        answered_at = np.where(sends, -1, stopped)

    confidences = no_exit
    confidences[:, : depth + 1] = reached
    confidences[np.arange(len(EXIT_NAMES)) > stopped[:, None]] = np.nan
    return Outcome(total, correct.astype(bool), busy, sent_bytes, answered_at, confidences)


def split_images(labels: np.ndarray, tune_share: float, seed: int) -> np.ndarray:
    """Mark a share of each class for tuning; the rest is for testing.

    Settings chosen on the images they are then scored on look better than
    they are, so the two sets never overlap. Returns a boolean mask.
    """
    rng = np.random.default_rng(seed)
    tuning = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == label))
        tuning[members[: round(tune_share * len(members))]] = True
    return tuning


def frame_stream(images: int, frames: int, seed: int) -> np.ndarray:
    """`frames` rows to show in order: every image once, shuffled, then again."""
    rng = np.random.default_rng(seed)
    passes = [rng.permutation(images) for _ in range(math.ceil(frames / images))]
    return np.concatenate(passes)[:frames]


def markov_states(frames: int, stay: tuple[float, float], seed: int) -> np.ndarray:
    """A sequence of 0s and 1s that keeps its state with probability `stay[state]`.

    State 0 is the good network and 1 the bad one. With `stay` at 0.95 a state
    lasts 20 frames on average. This is the two-state Markov chain networking
    calls the Gilbert-Elliott model.
    """
    rng = np.random.default_rng(seed)
    draws = rng.random(frames)
    states = np.zeros(frames, dtype=int)
    for frame in range(1, frames):
        previous = states[frame - 1]
        states[frame] = previous if draws[frame] < stay[previous] else 1 - previous
    return states


@dataclass(frozen=True)
class View:
    """What a rule knows before it decides on a frame."""

    frame: int
    deadline_ms: float
    # The network as last observed, or None if it has never been observed.
    network: Network | None
    # How many frames ago that observation was made.
    network_age: int


@dataclass(frozen=True)
class Feedback:
    """What a rule learns after a frame. Never whether the answer was right."""

    frame: int
    plan: Plan
    total_ms: float
    sent: bool
    answered_at: int
    confidences: tuple[float, ...]


class Rule(Protocol):
    """A decision rule: one plan per frame, and optionally learning from feedback."""

    name: str

    def choose(self, view: View) -> Plan: ...

    def observe(self, feedback: Feedback) -> None: ...


@dataclass
class Scenario:
    """Everything fixed for one run: the frames, their networks, the device and the deadline."""

    measurements: Measurements
    frames: np.ndarray
    states: np.ndarray
    networks: Sequence[Network]
    state_names: Sequence[str]
    slowdown: float
    deadline_ms: float
    rtt_ms: np.ndarray = field(init=False)
    uplink_mbps: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.rtt_ms = np.array([n.rtt_ms for n in self.networks], dtype=float)[self.states]
        self.uplink_mbps = np.array([n.uplink_mbps for n in self.networks], dtype=float)[
            self.states
        ]

    def resolve(self, plan: Plan, at: np.ndarray | None = None) -> Outcome:
        """Carry out one plan on every frame, or on the frames at positions `at`."""
        at = np.arange(len(self.frames)) if at is None else at
        return resolve(
            self.measurements,
            plan,
            self.frames[at],
            self.rtt_ms[at],
            self.uplink_mbps[at],
            self.slowdown,
        )


def _probe_interval(knowledge: str) -> int | None:
    if knowledge in ("exact", "sends"):
        return None
    kind, _, interval = knowledge.partition(":")
    if kind != "probe" or not interval.isdigit() or int(interval) < 1:
        raise ValueError(f"knowledge must be exact, sends or probe:N, got {knowledge!r}")
    return int(interval)


def simulate(rule: Rule, scenario: Scenario, knowledge: str = "exact") -> pd.DataFrame:
    """Run one rule over the scenario's frames; return one row per frame."""
    interval = _probe_interval(knowledge)
    last_send: int | None = None
    rows = []
    for frame in range(len(scenario.frames)):
        if knowledge == "exact":
            seen: int | None = frame
        elif knowledge == "sends":
            seen = last_send
        else:
            seen = frame - frame % interval
        view = View(
            frame=frame,
            deadline_ms=scenario.deadline_ms,
            network=None if seen is None else scenario.networks[scenario.states[seen]],
            network_age=0 if seen is None else frame - seen,
        )
        plan = rule.choose(view)
        outcome = scenario.resolve(plan, np.array([frame]))
        sent = bool(outcome.sent_bytes[0] > 0)
        if sent:
            last_send = frame
        rule.observe(
            Feedback(
                frame=frame,
                plan=plan,
                total_ms=float(outcome.total_ms[0]),
                sent=sent,
                answered_at=int(outcome.answered_at[0]),
                confidences=tuple(outcome.confidences[0]),
            )
        )
        rows.append(
            {
                "frame": frame,
                "image": int(scenario.measurements.images[scenario.frames[frame]]),
                "network": scenario.state_names[scenario.states[frame]],
                "plan": plan.describe(),
                "answered_at": int(outcome.answered_at[0]),
                "sent_bytes": float(outcome.sent_bytes[0]),
                "device_busy_ms": float(outcome.device_busy_ms[0]),
                "total_ms": float(outcome.total_ms[0]),
                "correct": bool(outcome.correct[0]),
            }
        )
    trace = pd.DataFrame(rows)
    trace["useful"] = trace["correct"] & (trace["total_ms"] <= scenario.deadline_ms)
    return trace


def score(
    total_ms: np.ndarray,
    correct: np.ndarray,
    device_busy_ms: np.ndarray,
    sent_bytes: np.ndarray,
    deadline_ms: float,
) -> dict[str, float]:
    """The scoreboard's columns for one run."""
    on_time = total_ms <= deadline_ms
    return {
        "useful": float((correct & on_time).mean()),
        "accuracy": float(correct.mean()),
        "on_time": float(on_time.mean()),
        "mean_ms": float(total_ms.mean()),
        "p95_ms": float(np.percentile(total_ms, 95)),
        "device_busy_ms": float(device_busy_ms.mean()),
        "sent_bytes": float(sent_bytes.mean()),
        "sent_share": float((sent_bytes > 0).mean()),
    }


def score_trace(trace: pd.DataFrame, deadline_ms: float) -> dict[str, float]:
    """`score` for a trace returned by `simulate`."""
    return score(
        trace["total_ms"].to_numpy(),
        trace["correct"].to_numpy(),
        trace["device_busy_ms"].to_numpy(),
        trace["sent_bytes"].to_numpy(),
        deadline_ms,
    )
