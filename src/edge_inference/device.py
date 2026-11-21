"""The device's side of the menu, run for real: answer locally, send, or both at once.

For each frame the device follows a plan. Locally it prepares the frame and
runs a graph. To send, it encodes the frame as a photo and asks a server. A
hedged plan does both: it encodes and sends first, then computes while the
photo travels. By the deadline it gives the server's answer if it has arrived,
and its own otherwise. It then waits, up to the horizon, for a reply still on
its way, since a record can still use it.

With `cancel` set to "now", the local run is aborted the moment the server's
answer lands: ONNX Runtime stops a graph mid-run when its run options are told
to. The time the device spent, and whether stopping saved any, is measured,
not assumed.

Every time is taken from the frame's arrival. Decoding the stored image is the
camera's work and is not counted, as in the profiles.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort
from PIL import Image

from edge_inference.ort_session import INPUT_NAME, OUTPUT_NAME
from edge_inference.preprocess import Preprocess, encode_photo
from edge_inference.serving import answer
from edge_inference.simulate import AT_ONCE, NEVER, Plan
from edge_inference.wire import Connection, Pending


@dataclass(frozen=True)
class PhotoRoute:
    """How a photo is sent: its shorter side, its JPEG quality, and the model asked for."""

    shorter: int
    quality: int
    model: str


@dataclass
class Variant:
    """A graph on the device, and the recipe that prepares its input."""

    session: ort.InferenceSession
    recipe: Preprocess


@dataclass(frozen=True)
class Frame:
    """What happened to one frame, measured. Times are ms from the frame's arrival."""

    total_ms: float
    correct: bool
    answered_by: str
    local_ms: float
    local_correct: bool
    server_at_ms: float
    server_correct: bool
    device_busy_ms: float
    prepare_ms: float
    preprocess_ms: float
    sent_bytes: int
    server_ms: float
    cancelled: bool


@dataclass
class Device:
    variants: dict[str, Variant]
    routes: dict[str, PhotoRoute]
    connection: Connection | None = None
    clock: Callable[[], float] = time.perf_counter
    # How a graph is run; replaceable so a test can make the local run slow.
    infer: Callable[[Variant, np.ndarray, ort.RunOptions], np.ndarray] = field(default=None)

    def __post_init__(self) -> None:
        if self.infer is None:
            self.infer = lambda variant, x, options: variant.session.run(
                [OUTPUT_NAME], {INPUT_NAME: x}, options
            )[0][0]

    def run(
        self,
        plan: Plan,
        frame: Image.Image,
        label: int,
        *,
        deadline_ms: float,
        horizon_ms: float,
        cancel: str = NEVER,
    ) -> Frame:
        start = self.clock()

        def since() -> float:
            return (self.clock() - start) * 1000.0

        sends = plan.precision is None or plan.hedge
        pending: Pending | None = None
        prepare_ms = 0.0
        sent_bytes = 0
        options = ort.RunOptions()
        if sends:
            route = self.routes[plan.route]
            photo = encode_photo(frame, route.shorter, route.quality)
            prepare_ms = since()
            sent_bytes = len(photo)

            def stop_local(_: Pending) -> None:
                if cancel == AT_ONCE:
                    options.terminate = True

            pending = self.connection.request(
                {"kind": "photo", "model": route.model}, photo, on_arrival=stop_local
            )

        local_ms, local_correct, preprocess_ms, cancelled = np.inf, False, 0.0, False
        busy_ms = prepare_ms
        if plan.precision is not None:
            variant = self.variants[plan.precision]
            before = since()
            x = variant.recipe(frame)
            preprocess_ms = since() - before
            try:
                logits = self.infer(variant, x, options)
            except Exception:
                if not options.terminate:
                    raise
                cancelled = True
            else:
                local_ms = since()
                local_correct = answer(logits)[0] == label
            busy_ms = since() if cancelled else local_ms

        server_at_ms, server_correct, server_ms = np.inf, False, np.nan
        if pending is not None:
            # By the deadline: is the server's answer in?
            pending.wait(max(0.0, deadline_ms - since()) / 1000.0)
            if not pending.arrived.is_set() and np.isinf(local_ms):
                # Nothing local to give: wait for the server, up to the horizon.
                pending.wait(max(0.0, horizon_ms - since()) / 1000.0)

        def server_state() -> tuple[float, bool, float]:
            if pending is None or not pending.arrived.is_set():
                return np.inf, False, np.nan
            arrived = (pending.arrived_at - start) * 1000.0
            return arrived, pending.reply["predicted"] == label, pending.reply["server_ms"]

        server_at_ms, server_correct, server_ms = server_state()
        if server_at_ms <= deadline_ms:
            total, correct, by = server_at_ms, server_correct, "server"
        elif local_ms <= deadline_ms:
            total, correct, by = (deadline_ms if sends else local_ms), local_correct, "local"
        elif server_at_ms <= local_ms:
            total, correct, by = server_at_ms, server_correct, "server"
        else:
            total, correct, by = local_ms, local_correct, "local"

        if pending is not None and not pending.arrived.is_set():
            # Keep the frame open until the horizon, for a reply a record can use.
            pending.wait(max(0.0, horizon_ms - since()) / 1000.0)
            server_at_ms, server_correct, server_ms = server_state()
            if by == "local" and np.isinf(local_ms) and server_at_ms < np.inf:
                total, correct, by = server_at_ms, server_correct, "server"

        return Frame(
            total_ms=float(total),
            correct=bool(correct),
            answered_by=by,
            local_ms=float(local_ms),
            local_correct=bool(local_correct),
            server_at_ms=float(server_at_ms),
            server_correct=bool(server_correct),
            device_busy_ms=float(busy_ms),
            prepare_ms=float(prepare_ms),
            preprocess_ms=float(preprocess_ms),
            sent_bytes=int(sent_bytes),
            server_ms=float(server_ms),
            cancelled=cancelled,
        )
