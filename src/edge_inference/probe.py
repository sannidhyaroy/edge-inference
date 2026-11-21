"""Record a link the way frames will use it: over our own connection, at the frame rate.

Every tick sends two echoes to the server, which answers at once without
computing anything. A tiny one gives the round trip; a larger one, timed after
it, gives the uplink rate from the difference. Every few ticks a photo-sized
echo checks that the two numbers predict what sending a photo really takes.

Probing at the frame period matters: a connection left idle slows its next
large send down again, and frames see the same.
"""

import os
import time
from collections.abc import Callable

import pandas as pd

from edge_inference.wire import Connection


def probe(
    connection: Connection,
    *,
    link: str,
    seconds: float,
    tick_ms: float = 100.0,
    small: int = 64,
    large: int = 32_768,
    photo: int = 8_000,
    photo_every: int = 10,
    timeout_s: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> pd.DataFrame:
    """Probe for `seconds`; one row per tick."""
    rows = []
    start = clock()
    tick = 0
    payloads = {size: os.urandom(size) for size in (small, large, photo)}
    while clock() - start < seconds:
        due = start + tick * tick_ms / 1000.0
        if clock() < due:
            sleep(due - clock())
        row = {"tick": tick, "t_s": clock() - start, "link": link}
        try:
            short = connection.request({"kind": "echo"}, payloads[small])
            if short.wait(timeout_s) is None:
                raise TimeoutError("no echo within the timeout")
            long = connection.request({"kind": "echo"}, payloads[large])
            if long.wait(timeout_s) is None:
                raise TimeoutError("no echo within the timeout")
            row["rtt_ms"] = short.round_trip_ms
            row["large_ms"] = long.round_trip_ms
            extra = row["large_ms"] - row["rtt_ms"]
            row["uplink_mbps"] = (large - small) * 8 / extra / 1000.0 if extra > 0 else float("nan")
            if tick % photo_every == 0:
                sent = connection.request({"kind": "echo"}, payloads[photo])
                if sent.wait(timeout_s) is None:
                    raise TimeoutError("no echo within the timeout")
                row["photo_ms"] = sent.round_trip_ms
                if extra > 0:
                    row["predicted_photo_ms"] = row["rtt_ms"] + (photo - small) * 8 / (
                        row["uplink_mbps"] * 1000.0
                    )
        except (TimeoutError, OSError) as error:
            row["error"] = type(error).__name__
        rows.append(row)
        tick += 1
    columns = [
        "tick", "t_s", "link", "rtt_ms", "large_ms", "uplink_mbps",
        "photo_ms", "predicted_photo_ms", "error",
    ]  # fmt: skip
    return pd.DataFrame(rows).reindex(columns=columns)
