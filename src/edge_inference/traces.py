"""Recorded links, replayed as the network that frames meet.

`edge probe` records a link at the frame period: each tick, the round trip and
the uplink rate. Here a recording becomes a `Trace`, and frame i meets the tick
the recording was at when the frame arrived. A tick whose probe failed is a
frame the server never answers. A tick whose large echo came back no slower
than the small one gives no rate, and keeps the last rate measured.

Rules that look a plan up per network need a few network states, not a
continuum, so `Bins` cuts a tuning recording into states by what sending the
median photo would cost there, plus a state for a link that is down. The bins
are fitted on the tuning recording only, and every frame still meets its own
recorded values: the bins decide what a rule plays, never what a frame costs.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from edge_inference.offload import Network
from edge_inference.simulate import HORIZON_MS, NEVER, Measurements, Scenario, frame_stream

DOWN = "down"
NAMES = {1: ("steady",), 2: ("fast", "slow"), 3: ("fast", "middle", "slow")}


def transfer_ms(rtt_ms, uplink_mbps, payload_bytes):
    """A round trip plus the time the payload takes to go up; `inf` on a dead link."""
    return np.asarray(rtt_ms) + np.asarray(payload_bytes) * 8 / (np.asarray(uplink_mbps) * 1000)


@dataclass(frozen=True)
class Trace:
    """One recorded link: a round trip and an uplink rate for every tick."""

    link: str
    rtt_ms: np.ndarray
    uplink_mbps: np.ndarray
    tick_ms: float

    def __len__(self) -> int:
        return len(self.rtt_ms)

    @classmethod
    def from_probe(cls, rows: pd.DataFrame, tick_ms: float = 100.0) -> Trace:
        """A recording written by `edge probe`, one row per tick."""
        rows = rows.sort_values("tick")
        links = rows["link"].unique()
        if len(links) != 1:
            raise ValueError(f"a trace is one link, got {list(links)}")
        failed = (rows["error"].notna() | rows["rtt_ms"].isna()).to_numpy()
        rtt = rows["rtt_ms"].to_numpy(dtype=float).copy()
        rtt[failed] = math.inf
        rate = rows["uplink_mbps"].where(rows["uplink_mbps"] > 0).ffill().bfill()
        if rate.isna().all():
            raise ValueError("the recording never measured an uplink rate")
        return cls(str(links[0]), rtt, rate.to_numpy(dtype=float), tick_ms)

    @classmethod
    def from_seconds(cls, rows: pd.DataFrame, link: str) -> Trace:
        """A recording written by `scripts/record_network.py`: one row a second."""
        rows = rows.rename(columns={"second": "tick"}).assign(link=link)
        return cls.from_probe(rows, tick_ms=1000.0)

    def halves(self) -> tuple[Trace, Trace]:
        """The first half and the second, for a link recorded only once."""
        middle = len(self) // 2
        return tuple(
            Trace(self.link, self.rtt_ms[part], self.uplink_mbps[part], self.tick_ms)
            for part in (slice(None, middle), slice(middle, None))
        )

    def frames(self, count: int, period_ms: float, start: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """The round trip and rate each of `count` frames meets, one frame every `period_ms`.

        A run longer than the recording starts it again from the beginning.
        """
        ticks = start + np.floor(np.arange(count) * period_ms / self.tick_ms).astype(int)
        ticks %= len(self)
        return self.rtt_ms[ticks], self.uplink_mbps[ticks]


@dataclass(frozen=True)
class Bins:
    """Network states cut from a recording by what sending a reference payload costs.

    `edges` separate the live states, cheapest first; the last state is a
    link that is down. `networks` stand for the states when a plan is chosen:
    the median round trip and rate of the tuning ticks in each.
    """

    reference_bytes: float
    edges: tuple[float, ...]
    networks: tuple[Network, ...]
    names: tuple[str, ...]

    @classmethod
    def fit(
        cls, trace: Trace, reference_bytes: float, *, count: int = 3, min_ticks: int = 200
    ) -> Bins:
        """Cut at quantiles, with fewer states when a state would hold under `min_ticks`.

        Many ticks can cost exactly the same, which makes quantiles coincide; an
        edge that would leave a state with no ticks is dropped.
        """
        cost = transfer_ms(trace.rtt_ms, trace.uplink_mbps, reference_bytes)
        live = np.isfinite(cost)
        values = cost[live]
        if not len(values):
            raise ValueError("the link was down for the whole recording")
        count = max(1, min(count, len(values) // min_ticks))
        edges = sorted({float(q) for q in np.quantile(values, np.arange(1, count) / count)})
        while edges:
            sizes = np.bincount(
                np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1
            )
            empty = np.flatnonzero(sizes == 0)
            if not len(empty):
                break
            del edges[min(int(empty[0]), len(edges) - 1)]
        edges = tuple(edges)
        count = len(edges) + 1
        states = np.searchsorted(edges, values, side="right")
        rtt, rate = trace.rtt_ms[live], trace.uplink_mbps[live]
        networks = tuple(
            Network(float(np.median(rtt[states == s])), float(np.median(rate[states == s])))
            for s in range(count)
        )
        names = NAMES.get(count, tuple(f"bin{s}" for s in range(count)))
        down = Network(math.inf, float(np.median(rate)))
        return cls(float(reference_bytes), edges, (*networks, down), (*names, DOWN))

    def assign(self, rtt_ms: np.ndarray, uplink_mbps: np.ndarray) -> np.ndarray:
        """The state each (round trip, rate) pair falls in."""
        cost = transfer_ms(rtt_ms, uplink_mbps, self.reference_bytes)
        states = np.searchsorted(self.edges, cost, side="right")
        return np.where(np.isfinite(cost), states, len(self.names) - 1)

    def state_of(self, network: Network) -> int:
        return int(self.assign(np.array([network.rtt_ms]), np.array([network.uplink_mbps]))[0])


def recorded_scenario(
    measurements: Measurements,
    trace: Trace,
    bins: Bins,
    *,
    period_ms: float,
    deadline_ms: float,
    seed: int,
    passes: int = 1,
    horizon_ms: float = HORIZON_MS,
    cancel: str = NEVER,
    start: int = 0,
) -> Scenario:
    """Every image `passes` times, shuffled by `seed`, over the recorded link.

    The device's times are measured on the device itself, so nothing is scaled.
    """
    links = {route.link for route in measurements.routes.values()}
    if links - {trace.link}:
        raise ValueError(f"routes use {sorted(links)}, the recording is {trace.link}")
    frames = frame_stream(len(measurements), passes * len(measurements), seed)
    rtt, rate = trace.frames(len(frames), period_ms, start)
    return Scenario(
        measurements,
        frames,
        bins.assign(rtt, rate),
        list(bins.networks),
        list(bins.names),
        slowdown=1.0,
        deadline_ms=deadline_ms,
        horizon_ms=horizon_ms,
        cancel=cancel,
        rtt_ms=rtt,
        uplink_mbps=rate,
    )
