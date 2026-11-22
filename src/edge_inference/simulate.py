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

**A frame is a timeline, not one answer.** `resolve` records when the device's
own answer was ready and whether it was right, and when the server's arrived
and whether it was right. What the frame is worth depends on the application:
an answer by the deadline for something that acts once, or the best answer by
a later horizon for something that keeps a record. `score` reads both.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from edge_inference.constants import EXIT_NAMES, SPLIT_NAMES
from edge_inference.offload import Network

# The early-exit table: two precisions of one network with three exits, and
# feature maps at the first two. Other tables bring their own variants and routes.
PRECISIONS = ("float32", "int8")
PAYLOADS = ("float32", "uint8_zstd")
FINAL = len(EXIT_NAMES) - 1

# What a device does at the deepest exit it runs to when that exit is not
# confident enough: answer anyway, send the feature map, or send the photo.
ANSWER, FEATURES, PHOTO = "answer", "features", "photo"

# The horizon a record is kept open for: a late answer that arrives by then
# still replaces the one shown.
HORIZON_MS = 1000.0

# When a hedged local run stops once the server has answered: not at all, at
# the next exit it reaches, or at once. A runtime can abort a graph while it
# runs, so stopping at once needs no exit to stop at.
NEVER, AT_EXIT, AT_ONCE = "never", "exit", "now"


@dataclass(frozen=True)
class Local:
    """One way of running the network on the device: arrays of images by exits.

    `ms` is cumulative device time to each exit, shallowest first.
    """

    ms: np.ndarray
    confidence: np.ndarray
    correct: np.ndarray
    exit_names: tuple[str, ...]

    @property
    def final(self) -> int:
        return len(self.exit_names) - 1


@dataclass(frozen=True)
class Route:
    """One way of sending a frame to a server: arrays with one entry per image.

    A photo route sends the photo; a feature route sends the feature map the
    device computed up to `after`, a (variant, exit) pair. `correct` and
    `confidence` are the server's own.
    """

    bytes: np.ndarray
    prepare_ms: np.ndarray
    server_ms: np.ndarray
    correct: np.ndarray
    confidence: np.ndarray
    payload: str = PHOTO
    server: str = "server"
    link: str = "net"
    after: tuple[str, int] | None = None


def feature_route(variant: str, exit: int, payload: str) -> str:
    """The key of the route that sends `variant`'s feature map after `exit`."""
    return f"features:{payload}@{variant}/{exit}"


@dataclass(frozen=True)
class Measurements:
    """The per-image table, as arrays with one row per image.

    `local` has a block per variant (a precision, or a model at a resolution
    and precision), `routes` a block per way of sending.
    """

    images: np.ndarray
    labels: np.ndarray
    local: Mapping[str, Local]
    routes: Mapping[str, Route]

    def __len__(self) -> int:
        return len(self.images)

    def final(self, variant: str) -> int:
        return self.local[variant].final

    @property
    def photo_routes(self) -> list[str]:
        return [key for key, route in self.routes.items() if route.after is None]

    def photo_route(self, key: str | None) -> Route:
        """The route a plan sends its photo by: named, or the only one there is."""
        if key is not None:
            return self.routes[key]
        keys = self.photo_routes
        if len(keys) != 1:
            raise ValueError(f"name the photo route: there are {len(keys)}")
        return self.routes[keys[0]]

    @classmethod
    def from_arrays(
        cls,
        *,
        images: np.ndarray,
        labels: np.ndarray,
        device_ms: Mapping[str, np.ndarray],
        confidence: Mapping[str, np.ndarray],
        correct: Mapping[str, np.ndarray],
        photo_bytes: np.ndarray,
        photo_server_ms: np.ndarray,
        photo_correct: np.ndarray,
        features_bytes: Mapping[str, np.ndarray],
        features_prepare_ms: Mapping[str, np.ndarray],
        features_server_ms: Mapping[str, np.ndarray],
        features_correct: Mapping[str, np.ndarray],
        server: str = "server",
        link: str = "net",
    ) -> Measurements:
        """The early-exit table's shape: precisions of one network, one photo, feature maps.

        The profiles did not record the server's confidence. The server runs
        the float32 network to its final exit, so the device's own confidence
        there stands in for it, on every route.
        """
        exits = tuple(EXIT_NAMES[: next(iter(device_ms.values())).shape[1]])
        local = {
            precision: Local(device_ms[precision], confidence[precision], correct[precision], exits)
            for precision in device_ms
        }
        from_server = confidence["float32"][:, -1]
        routes = {
            PHOTO: Route(
                photo_bytes,
                np.zeros(len(images)),
                photo_server_ms,
                photo_correct,
                from_server,
                server=server,
                link=link,
            )
        }
        splits = next(iter(features_bytes.values())).shape[1] if features_bytes else 0
        for split in range(splits):
            for payload in features_bytes:
                routes[feature_route("float32", split, payload)] = Route(
                    features_bytes[payload][:, split],
                    features_prepare_ms[payload][:, split],
                    features_server_ms[payload][:, split],
                    features_correct[payload][:, split],
                    from_server,
                    payload=payload,
                    server=server,
                    link=link,
                    after=("float32", split),
                )
        return cls(np.asarray(images), np.asarray(labels), local, routes)

    @classmethod
    def from_components(cls, components: pd.DataFrame, server: str) -> Measurements:
        """The early-exit long table: the local rows and the rows for `server`, as arrays.

        Every block is pivoted on the image, and every block must hold the same
        images, so nothing is aligned by position alone.
        """

        def wide(frame: pd.DataFrame, value: str, columns: Sequence[str]) -> pd.DataFrame:
            return frame.pivot(index="image", columns="point", values=value)[list(columns)]

        components = components.assign(
            server_ms=components["server_receive_ms"] + components["server_compute_ms"]
        )
        local = components[components["mode"] == "local"]
        by_precision = {p: local[local["device_precision"] == p] for p in PRECISIONS}
        sent = components[components["server"] == server]
        photo = sent[sent["mode"] == "offload"].set_index("image").sort_index()
        by_payload = {p: sent[(sent["mode"] == "split") & (sent["payload"] == p)] for p in PAYLOADS}

        index = photo.index
        if not index.is_unique:
            raise ValueError("an image appears twice among the photo rows")

        def aligned(block: pd.DataFrame) -> np.ndarray:
            if not block.index.equals(index):
                raise ValueError("the blocks of the table do not hold the same images")
            return block.to_numpy()

        labels = local.drop_duplicates("image").set_index("image")["label"].sort_index()
        if not labels.index.equals(index) or not (labels.to_numpy() == photo["label"]).all():
            raise ValueError("local and sent rows disagree on the images or their labels")

        return cls.from_arrays(
            images=index.to_numpy(),
            labels=photo["label"].to_numpy(),
            device_ms={
                p: aligned(wide(f, "device_ms", EXIT_NAMES)) for p, f in by_precision.items()
            },
            confidence={
                p: aligned(wide(f, "confidence", EXIT_NAMES)) for p, f in by_precision.items()
            },
            correct={p: aligned(wide(f, "correct", EXIT_NAMES)) for p, f in by_precision.items()},
            photo_bytes=photo["payload_bytes"].to_numpy(),
            photo_server_ms=photo["server_ms"].to_numpy(),
            photo_correct=photo["correct"].to_numpy(),
            features_bytes={
                p: aligned(wide(f, "payload_bytes", SPLIT_NAMES)) for p, f in by_payload.items()
            },
            features_prepare_ms={
                p: aligned(wide(f, "prepare_ms", SPLIT_NAMES)) for p, f in by_payload.items()
            },
            features_server_ms={
                p: aligned(wide(f, "server_ms", SPLIT_NAMES)) for p, f in by_payload.items()
            },
            features_correct={
                p: aligned(wide(f, "correct", SPLIT_NAMES)) for p, f in by_payload.items()
            },
            server=server,
        )

    def subset(self, rows: np.ndarray) -> Measurements:
        """The same table restricted to `rows`, a boolean mask or an index array."""

        def take(block):
            return type(block)(
                **{
                    name: value[rows] if isinstance(value, np.ndarray) else value
                    for name, value in vars(block).items()
                }
            )

        return Measurements(
            self.images[rows],
            self.labels[rows],
            {key: take(block) for key, block in self.local.items()},
            {key: take(route) for key, route in self.routes.items()},
        )


@dataclass(frozen=True)
class Plan:
    """What to do with one frame.

    With no `precision` the photo is sent at once. Otherwise the device runs
    that variant of the network up to exit `depth`, counted from zero; `None`
    means the final exit. On the way it stops and answers at the first exit
    whose confidence reaches that exit's entry in `thresholds`; a missing entry
    means never stop there. At `depth` itself, without enough confidence, it
    does `otherwise`. At the final exit that can only be answering anyway or
    sending the photo, a cascade: there is no feature map after the last exit.

    `route` names the photo route when the table has more than one. A plan
    that `hedge`s sends the photo the moment the frame arrives and runs its
    local part at the same time; the frame gets whichever answer serves it.
    """

    precision: str | None = None
    depth: int | None = None
    thresholds: tuple[float, ...] = ()
    otherwise: str = ANSWER
    payload: str = "uint8_zstd"
    route: str | None = None
    hedge: bool = False

    def __post_init__(self) -> None:
        if self.depth is not None and self.depth < 0:
            raise ValueError("depth counts exits from zero")
        if self.otherwise == FEATURES and self.depth is None:
            raise ValueError("there is no feature map after the final exit to send")
        if self.hedge and (self.precision is None or self.otherwise != ANSWER):
            raise ValueError("a hedge pairs a sent photo with a local answer")

    def final_at(self, final: int) -> bool:
        """Whether this plan runs to the final exit of a variant whose last exit is `final`."""
        return self.depth is None or self.depth == final

    def describe(self) -> str:
        via = f" via {self.route}" if self.route is not None else ""
        if self.precision is None:
            return f"offload{via}"
        final = self.depth is None or self.depth == FINAL
        if final:
            stops = [f"{t:g}" for t in self.thresholds if t != math.inf]
        else:
            padded = (*self.thresholds, *[math.inf] * (self.depth + 1))[: self.depth + 1]
            checks = self.depth + (self.otherwise != ANSWER)
            stops = [f"{t:g}" for t in padded[:checks] if t != math.inf]
        where = "final exit" if final else f"exit {self.depth + 1}"
        text = f"{self.precision} to {where}"
        if stops:
            text += f", stop at {'/'.join(stops)}"
        if self.otherwise == PHOTO:
            text += f", else send photo{via}"
        if not final and self.otherwise == FEATURES:
            text += f", else send {self.payload} features"
        if self.hedge:
            text += f", photo sent alongside{via}"
        return text


@dataclass(frozen=True)
class Outcome:
    """What happened to each frame; every field is an array with one entry per frame.

    `total_ms` and `correct` are the answer the frame was given at the time it
    was given, which is what something that acts once sees. The events behind
    them are kept too: when the device's own answer was ready and whether it
    was right, and when the server's answer arrived and whether it was right.
    An event that never happened has time `inf` and is not right.
    """

    total_ms: np.ndarray
    correct: np.ndarray
    device_busy_ms: np.ndarray
    sent_bytes: np.ndarray
    # The exit that answered, counted from zero, or -1 when the server answered.
    answered_at: np.ndarray
    # Confidence at each exit the device reached; NaN for exits it did not.
    confidences: np.ndarray
    # Confidence the server reports with its answer; NaN when nothing was sent.
    server_confidence: np.ndarray
    local_ms: np.ndarray
    local_correct: np.ndarray
    server_at_ms: np.ndarray
    server_correct: np.ndarray


def resolve(
    measurements: Measurements,
    plan: Plan,
    rows: np.ndarray,
    rtt_ms: np.ndarray | float,
    uplink_mbps: np.ndarray | float,
    slowdown: float = 1.0,
    *,
    deadline_ms: float | None = None,
    cancel: str = NEVER,
) -> Outcome:
    """Carry out `plan` on the images at `rows`, each over its own network.

    A hedged plan needs `deadline_ms`: until then the device waits for the
    server, and only then gives its own answer. `cancel` says when a hedged
    local run stops once the server's answer is in: `NEVER`, `AT_EXIT` (the
    next exit it reaches) or `AT_ONCE`.
    """
    rows = np.asarray(rows)
    count = len(rows)
    # `Network` only does arithmetic, so it takes one value per frame as well.
    network = Network(rtt_ms=rtt_ms, uplink_mbps=uplink_mbps)
    never = np.full(count, np.inf)
    nothing = np.zeros(count, dtype=bool)

    if plan.precision is None:
        route = measurements.photo_route(plan.route)
        size = route.bytes[rows]
        prepare = slowdown * route.prepare_ms[rows]
        arrives = prepare + network.transfer_ms(size) + route.server_ms[rows]
        right = route.correct[rows].astype(bool)
        width = max(block.final for block in measurements.local.values()) + 1
        return Outcome(
            total_ms=arrives,
            correct=right,
            device_busy_ms=prepare,
            sent_bytes=size,
            answered_at=np.full(count, -1),
            confidences=np.full((count, width), np.nan),
            server_confidence=route.confidence[rows],
            local_ms=never,
            local_correct=nothing,
            server_at_ms=arrives,
            server_correct=right,
        )

    block = measurements.local[plan.precision]
    final = block.final
    depth = final if plan.depth is None else plan.depth
    if depth > final:
        raise ValueError(f"{plan.precision} has exits 0 to {final}, not {depth}")
    reached = block.confidence[rows, : depth + 1]
    thresholds = np.array((*plan.thresholds, *[math.inf] * (depth + 1))[: depth + 1])
    confident = reached >= thresholds
    if plan.otherwise == ANSWER:
        confident[:, depth] = True
    answers = confident.any(axis=1)
    # argmax on booleans gives the first True; a row with none stops at `depth`.
    stopped = np.where(answers, confident.argmax(axis=1), depth)

    local_busy = slowdown * block.ms[rows, stopped]
    local_right = block.correct[rows, stopped].astype(bool)
    confidences = np.full((count, final + 1), np.nan)
    confidences[:, : depth + 1] = reached
    confidences[np.arange(final + 1) > stopped[:, None]] = np.nan

    if plan.hedge:
        return _hedge(
            measurements,
            plan,
            rows,
            network,
            slowdown,
            deadline_ms,
            cancel,
            block,
            stopped,
            local_busy,
            local_right,
            confidences,
        )

    busy = local_busy
    total = busy.copy()
    correct = local_right.copy()
    sent_bytes = np.zeros(count)
    answered_at = stopped.copy()
    server_at = never
    server_right = nothing
    server_confidence = np.full(count, np.nan)

    sends = ~answers
    if sends.any():
        if plan.otherwise == PHOTO:
            route = measurements.photo_route(plan.route)
        else:
            key = feature_route(plan.precision, depth, plan.payload)
            if key not in measurements.routes:
                raise ValueError(f"no feature map was profiled for {plan.describe()}")
            route = measurements.routes[key]
        size = route.bytes[rows]
        prepare = slowdown * route.prepare_ms[rows]
        busy = busy + np.where(sends, prepare, 0.0)
        arrives = busy + network.transfer_ms(size) + route.server_ms[rows]
        total = np.where(sends, arrives, busy)
        correct = np.where(sends, route.correct[rows], correct).astype(bool)
        sent_bytes = np.where(sends, size, 0.0)
        answered_at = np.where(sends, -1, stopped)
        server_at = np.where(sends, arrives, np.inf)
        server_right = sends & route.correct[rows].astype(bool)
        server_confidence = np.where(sends, route.confidence[rows], np.nan)

    return Outcome(
        total_ms=total,
        correct=correct,
        device_busy_ms=busy,
        sent_bytes=sent_bytes,
        answered_at=answered_at,
        confidences=confidences,
        server_confidence=server_confidence,
        local_ms=local_busy,
        local_correct=local_right,
        server_at_ms=server_at,
        server_correct=server_right,
    )


def _hedge(
    measurements: Measurements,
    plan: Plan,
    rows: np.ndarray,
    network: Network,
    slowdown: float,
    deadline_ms: float | None,
    cancel: str,
    block: Local,
    stopped: np.ndarray,
    local_busy: np.ndarray,
    local_right: np.ndarray,
    confidences: np.ndarray,
) -> Outcome:
    """Send the photo and run the local part at once; answer with whichever serves.

    The device encodes the photo first, then computes while the photo travels.
    By the deadline, the server's answer is given if it has arrived; otherwise
    the device's own answer is given at the deadline, since until then the
    device cannot know the server will be late. If both miss the deadline, the
    first to arrive is given.
    """
    if deadline_ms is None:
        raise ValueError("a hedged plan is resolved against a deadline")
    route = measurements.photo_route(plan.route)
    size = route.bytes[rows]
    prepare = slowdown * route.prepare_ms[rows]
    server_at = prepare + network.transfer_ms(size) + route.server_ms[rows]
    local_at = prepare + local_busy
    server_right = route.correct[rows].astype(bool)

    server_first = server_at <= deadline_ms
    local_in_time = ~server_first & (local_at <= deadline_ms)
    late = ~server_first & ~local_in_time
    late_server = late & (server_at <= local_at)
    by_server = server_first | late_server
    total = np.where(
        server_first,
        server_at,
        np.where(local_in_time, deadline_ms, np.minimum(server_at, local_at)),
    )
    correct = np.where(by_server, server_right, local_right)

    if cancel not in (NEVER, AT_EXIT, AT_ONCE):
        raise ValueError(f"cancel must be {NEVER}, {AT_EXIT} or {AT_ONCE}, got {cancel!r}")
    busy = local_at
    if cancel == AT_EXIT:
        # The device stops at the first exit it reaches after the answer is in.
        boundaries = prepare[:, None] + slowdown * block.ms[rows]
        reached = boundaries >= server_at[:, None]
        first = np.where(reached.any(axis=1), reached.argmax(axis=1), block.final)
        first = np.minimum(first, stopped)
        cut = boundaries[np.arange(len(rows)), first]
        busy = np.where(server_at < local_at, cut, local_at)
    elif cancel == AT_ONCE:
        # The device stops the moment the answer is in.
        busy = np.minimum(server_at, local_at)

    return Outcome(
        total_ms=total,
        correct=correct,
        device_busy_ms=busy,
        sent_bytes=size,
        answered_at=np.where(by_server, -1, stopped),
        confidences=confidences,
        server_confidence=route.confidence[rows],
        local_ms=local_at,
        local_correct=local_right,
        server_at_ms=server_at,
        server_correct=server_right,
    )


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
    # What the server said its confidence was; NaN when nothing was sent.
    server_confidence: float


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
    horizon_ms: float = HORIZON_MS
    # When a hedged local run stops once the server's answer is in.
    cancel: str = NEVER
    # A recorded link gives every frame its own values; `states` then only say
    # which bin each frame fell in, and `networks` stand for the bins. Without
    # them, every frame in a state meets that state's network exactly.
    rtt_ms: np.ndarray | None = None
    uplink_mbps: np.ndarray | None = None
    recorded: bool = field(init=False)

    def __post_init__(self) -> None:
        self.recorded = self.rtt_ms is not None
        if self.recorded:
            self.rtt_ms = np.asarray(self.rtt_ms, dtype=float)
            self.uplink_mbps = np.asarray(self.uplink_mbps, dtype=float)
            return
        self.rtt_ms = np.array([n.rtt_ms for n in self.networks], dtype=float)[self.states]
        self.uplink_mbps = np.array([n.uplink_mbps for n in self.networks], dtype=float)[
            self.states
        ]

    def observed(self, frame: int) -> Network:
        """The network a rule sees when it looks at `frame`: what was measured there."""
        if self.recorded:
            return Network(float(self.rtt_ms[frame]), float(self.uplink_mbps[frame]))
        return self.networks[self.states[frame]]

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
            deadline_ms=self.deadline_ms,
            cancel=self.cancel,
        )


def _probe_interval(knowledge: str) -> int | None:
    if knowledge in ("exact", "sends"):
        return None
    kind, _, interval = knowledge.partition(":")
    if kind != "probe" or not interval.isdigit() or int(interval) < 1:
        raise ValueError(f"knowledge must be exact, sends or probe:N, got {knowledge!r}")
    return int(interval)


def simulate(
    rule: Rule,
    scenario: Scenario,
    knowledge: str = "exact",
    execute: Callable[[Plan, int], Outcome] | None = None,
) -> pd.DataFrame:
    """Run one rule over the scenario's frames; return one row per frame.

    By default a plan's outcome is looked up in the measurements. `execute`
    replaces the lookup: given a plan and a frame's position it returns that
    one frame's outcome, which is how a live run reuses this loop.
    """
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
            network=None if seen is None else scenario.observed(seen),
            network_age=0 if seen is None else frame - seen,
        )
        plan = rule.choose(view)
        if execute is None:
            outcome = scenario.resolve(plan, np.array([frame]))
        else:
            outcome = execute(plan, frame)
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
                server_confidence=float(outcome.server_confidence[0]),
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
                "local_correct": bool(outcome.local_correct[0]),
                "server_at_ms": float(outcome.server_at_ms[0]),
                "server_correct": bool(outcome.server_correct[0]),
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
    *,
    local_correct: np.ndarray | None = None,
    server_at_ms: np.ndarray | None = None,
    server_correct: np.ndarray | None = None,
    horizon_ms: float = HORIZON_MS,
) -> dict[str, float]:
    """The scoreboard's columns for one run.

    `useful` is what something that acts once gets: a right answer by the
    deadline. Given the events behind each frame, two more columns say what
    something that keeps a record gets. `record` is the share whose final
    answer is right, where the server's answer replaces the device's if it
    arrives by `horizon_ms`. `corrected` is the share that were not useful at
    the deadline but were put right before the horizon.
    """
    on_time = total_ms <= deadline_ms
    useful = correct & on_time
    extra = {}
    if server_at_ms is not None:
        arrived = server_at_ms <= horizon_ms
        record = np.where(arrived, server_correct, local_correct)
        late = (server_at_ms > deadline_ms) & arrived
        extra = {
            "record": float(record.mean()),
            "corrected": float((~useful & late & server_correct).mean()),
        }
    return {
        "useful": float(useful.mean()),
        "accuracy": float(correct.mean()),
        "on_time": float(on_time.mean()),
        "mean_ms": float(total_ms.mean()),
        # Frames that never got an answer take forever, and interpolating
        # between two of them is undefined: then the value at or above is taken.
        "p95_ms": float(
            np.percentile(total_ms, 95)
            if np.isfinite(total_ms).all()
            else np.percentile(total_ms, 95, method="higher")
        ),
        "device_busy_ms": float(device_busy_ms.mean()),
        "sent_bytes": float(sent_bytes.mean()),
        "sent_share": float((sent_bytes > 0).mean()),
        **extra,
    }


def score_trace(
    trace: pd.DataFrame, deadline_ms: float, horizon_ms: float = HORIZON_MS
) -> dict[str, float]:
    """`score` for a trace returned by `simulate`."""
    events = {}
    if "server_at_ms" in trace:
        events = {
            name: trace[name].to_numpy()
            for name in ("local_correct", "server_at_ms", "server_correct")
        }
    return score(
        trace["total_ms"].to_numpy(),
        trace["correct"].to_numpy(),
        trace["device_busy_ms"].to_numpy(),
        trace["sent_bytes"].to_numpy(),
        deadline_ms,
        horizon_ms=horizon_ms,
        **events,
    )
