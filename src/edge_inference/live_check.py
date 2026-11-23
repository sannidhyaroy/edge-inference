"""Live runs against the simulator: the same images, plans and link, played both ways.

A live run carries a few plans out frame by frame against a real server. The
simulator plays the same plans on the same images from the measured table,
over the link's test recording. The recording was made at another time than
the live run, so the simulator plays the images over many stretches of it
and reports what it expects on average; the live run is one draw.

Two things are compared. Answers: every answer given live must be the one
the table holds for that image, on the device and on the server, or the
table does not describe what ran. Times and scores: how close the simulated
useful and record shares, and the times answers arrived, come to live.
"""

import numpy as np
import pandas as pd

from edge_inference.config import SEED
from edge_inference.menu_table import route_key
from edge_inference.simulate import Measurements, Plan, resolve
from edge_inference.traces import Trace

PLANS = ("local", "offload", "hedge")


def live_plan(name: str, variant: str, route: str) -> Plan:
    """The simulator's plan for a live run's plan name."""
    if name == "local":
        return Plan(variant)
    if name == "offload":
        return Plan(route=route)
    if name == "hedge":
        return Plan(variant, route=route, hedge=True)
    raise ValueError(f"no simulated plan for {name!r}")


def answer_agreement(table: Measurements, live: pd.DataFrame, variant: str, route: str) -> dict:
    """How many live answers, on the device and from the server, match the table's."""
    position = pd.Series(np.arange(len(table.images)), index=table.images)
    local = live[np.isfinite(live["local_ms"])]
    served = live[np.isfinite(live["server_at_ms"])]
    local_table = table.local[variant].correct[position[local["image"]].to_numpy(), -1]
    served_table = table.photo_route(route).correct[position[served["image"]].to_numpy()]
    return {
        "local_answers": len(local),
        "local_agree": int((local["local_correct"].to_numpy() == local_table).sum()),
        "server_answers": len(served),
        "server_agree": int((served["server_correct"].to_numpy() == served_table).sum()),
    }


def simulate_live(
    table: Measurements,
    live: pd.DataFrame,
    trace: Trace,
    server: str,
    *,
    draws: int = 50,
    period_ms: float = 100.0,
    seed: int = SEED,
) -> pd.DataFrame:
    """Per plan of a live run: its scores live, and what the simulator expects.

    `server` is the name the table gives the server the live run reached.
    """
    first = live.iloc[0]
    variant, deadline, horizon = first["variant"], first["deadline_ms"], first["horizon_ms"]
    route = route_key(first["photo"], server, first["server_model"])
    position = pd.Series(np.arange(len(table.images)), index=table.images)
    rng = np.random.default_rng(seed)
    rows = []
    for name, frames in live.groupby("plan", sort=False):
        images = position[frames["image"]].to_numpy()
        # Each draw plays the images over a stretch of the recording from a random start.
        starts = rng.integers(0, len(trace), size=draws)
        rtt, rate = zip(
            *(trace.frames(len(images), period_ms, int(s)) for s in starts), strict=True
        )
        outcome = resolve(
            table, live_plan(name, variant, route), np.tile(images, draws),
            np.concatenate(rtt), np.concatenate(rate),
            deadline_ms=deadline, cancel=first["cancel"],
        )  # fmt: skip
        useful = outcome.correct & (outcome.total_ms <= deadline)
        record = np.where(
            outcome.server_at_ms <= horizon, outcome.server_correct, outcome.local_correct
        )
        sent = np.isfinite(frames["server_at_ms"])
        rows.append(
            {
                "plan": name,
                "frames": len(frames),
                "live_useful": frames["useful"].mean(),
                "sim_useful": useful.mean(),
                "live_record": frames["record"].mean(),
                "sim_record": record.mean(),
                "live_median_ms": frames["total_ms"].median(),
                "sim_median_ms": float(np.median(outcome.total_ms)),
                "live_p95_ms": frames["total_ms"].quantile(0.95),
                "sim_p95_ms": float(np.quantile(outcome.total_ms, 0.95)),
                "live_server_median_ms": frames.loc[sent, "server_at_ms"].median(),
                "sim_server_median_ms": float(np.median(outcome.server_at_ms))
                if np.isfinite(outcome.server_at_ms).any()
                else np.nan,
                **answer_agreement(table, frames, variant, route),
            }
        )
    return pd.DataFrame(rows).assign(
        link=trace.link, variant=variant, route=route, deadline_ms=deadline,
        horizon_ms=horizon, cancel=first["cancel"], draws=draws,
    )  # fmt: skip
