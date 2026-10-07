"""Per-image cost of every way to process a frame, and what a network adds.

The draft's five inference modes reduce to three ways of moving work, which
combine with exits and precisions:

* **Local**: the device runs to an exit and answers itself. Stopping at the
  final exit is full local inference; stopping earlier is early exit.
* **Split**: the device runs to a split point, sends the feature map, and a
  server finishes the network.
* **Offload**: the device sends the image and the server does everything.

Everything here is assembled from measurements already taken, joined per image:
device time from the exit profiles, payload size and preparation time from the
split profiles, and receive and compute time from the server profiles.

**The network is left out of the stored table on purpose.** Its cost is
arithmetic, a round trip plus the payload's size over the uplink bandwidth, so
storing per-image components lets any channel model be applied exactly
afterwards, including one that draws a different channel per frame. The named
networks below are illustrative operating points for summaries, not
measurements.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from edge_inference.models import EXIT_NAMES

# Component columns, in the order a frame pays them.
COMPONENTS = ("device_ms", "prepare_ms", "server_receive_ms", "server_compute_ms")


@dataclass(frozen=True)
class Network:
    """An uplink: the fixed delay of a round trip, and the rate data flows at."""

    rtt_ms: float
    uplink_mbps: float

    def transfer_ms(self, payload_bytes: pd.Series | np.ndarray | float) -> pd.Series:
        """Time to send a payload and get a small answer back.

        The answer is a class index, a few bytes, so the return trip is covered
        by the round trip itself. Megabits are decimal, as bandwidth is quoted.
        """
        return self.rtt_ms + np.asarray(payload_bytes) * 8 / (self.uplink_mbps * 1000)


# Illustrative operating points, not measurements. They set the round trip and
# the uplink, which is what a device sending frames is limited by. Replace them
# with a measured or modelled channel for anything beyond a first look.
NETWORKS: dict[str, Network] = {
    "wifi": Network(rtt_ms=5, uplink_mbps=100),
    "5g": Network(rtt_ms=20, uplink_mbps=50),
    "4g": Network(rtt_ms=50, uplink_mbps=10),
    "4g_poor": Network(rtt_ms=100, uplink_mbps=2),
}

DEADLINES_MS = (20, 50, 100)


def build_components(
    exit_profiles: Mapping[str, pd.DataFrame],
    split_profile: pd.DataFrame,
    server_profiles: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """One row per image per way of processing it, with every non-network cost.

    `exit_profiles` maps a device precision to its exit profile, and
    `server_profiles` maps a server name to its server profile, all for the
    same checkpoint and the same images.

    Split rows use the float32 device's timings and payloads only. A feature
    map from INT8 stages differs from the float32 one, and no server profile
    has measured how a float32 server answers it.
    """
    rows = []
    for precision, profile in exit_profiles.items():
        local = profile[["image", "label", "exit_name", "cumulative_ms", "confidence", "correct"]]
        rows.append(
            local.rename(columns={"exit_name": "point", "cumulative_ms": "device_ms"}).assign(
                mode="local",
                device_precision=precision,
                payload="none",
                server="none",
                payload_bytes=0,
                prepare_ms=0.0,
                server_receive_ms=0.0,
                server_compute_ms=0.0,
            )
        )

    device = exit_profiles["float32"].set_index(["image", "exit_name"])["cumulative_ms"]
    split = split_profile.set_index(["image", "split_name"])
    payloads = {
        # Raw float32 bytes go out as they are, so preparing them costs nothing.
        "float32": (split["float32_bytes"], pd.Series(0.0, index=split.index)),
        "uint8_zstd": (
            split["uint8_zstd_bytes"],
            split["uint8_quantize_ms"] + split["uint8_zstd_ms"],
        ),
    }

    for server, profile in server_profiles.items():
        received = profile.set_index(["image", "split_name", "payload"])
        for payload, (size, prepare) in payloads.items():
            arrived = received.xs(payload, level="payload")
            frame = pd.DataFrame(
                {
                    "device_ms": device.loc[size.index].to_numpy(),
                    "payload_bytes": size,
                    "prepare_ms": prepare,
                    "server_receive_ms": arrived["receive_ms"].loc[size.index].to_numpy(),
                    "server_compute_ms": arrived["compute_ms"].loc[size.index].to_numpy(),
                    "correct": arrived["correct"].loc[size.index].to_numpy(),
                    "label": split["label"],
                }
            ).reset_index()
            rows.append(
                frame.rename(columns={"split_name": "point"}).assign(
                    mode="split",
                    device_precision="float32",
                    payload=payload,
                    server=server,
                    confidence=np.nan,
                )
            )

        jpeg = profile[profile["payload"] == "jpeg"]
        rows.append(
            pd.DataFrame(
                {
                    "image": jpeg["image"],
                    "label": jpeg["label"],
                    "point": "input",
                    "device_ms": 0.0,
                    "payload_bytes": jpeg["payload_bytes"],
                    "prepare_ms": 0.0,
                    "server_receive_ms": jpeg["receive_ms"],
                    "server_compute_ms": jpeg["compute_ms"],
                    "correct": jpeg["correct"],
                    "mode": "offload",
                    "device_precision": "none",
                    "payload": "jpeg",
                    "server": server,
                    "confidence": np.nan,
                }
            )
        )

    columns = [
        "image",
        "label",
        "mode",
        "point",
        "device_precision",
        "payload",
        "server",
        *COMPONENTS,
        "payload_bytes",
        "confidence",
        "correct",
    ]
    order = {name: index for index, name in enumerate(("input", *EXIT_NAMES))}
    frame = pd.concat(rows, ignore_index=True)[columns]
    frame["correct"] = frame["correct"].astype(bool)
    return frame.sort_values(
        ["image", "mode", "point", "device_precision", "payload", "server"],
        key=lambda column: column.map(order) if column.name == "point" else column,
        ignore_index=True,
    )


def total_ms(components: pd.DataFrame, network: Network) -> pd.Series:
    """End-to-end time per row: local rows pay nothing for the network."""
    compute = components[list(COMPONENTS)].sum(axis=1)
    sends = components["mode"] != "local"
    transfer = np.where(sends, network.transfer_ms(components["payload_bytes"]), 0.0)
    return compute + transfer


def summarise(
    components: pd.DataFrame,
    networks: Mapping[str, Network] = NETWORKS,
    deadlines: Iterable[float] = DEADLINES_MS,
) -> pd.DataFrame:
    """Accuracy, latency and deadline hit rate per configuration per network.

    A frame meets a deadline when it is answered in time, right or wrong;
    whether the answer was right is the accuracy column. Combining them, as a
    task-completion ratio that also demands a correct answer, is a choice for
    whoever sets the objective.
    """
    keys = ["mode", "point", "device_precision", "payload", "server"]
    deadlines = tuple(deadlines)
    summaries = []
    for name, network in networks.items():
        frame = components.assign(total_ms=total_ms(components, network))
        grouped = frame.groupby(keys, sort=False)
        summary = grouped.agg(
            accuracy=("correct", "mean"),
            mean_ms=("total_ms", "mean"),
            median_ms=("total_ms", "median"),
            p95_ms=("total_ms", lambda s: s.quantile(0.95)),
            median_bytes=("payload_bytes", "median"),
        )
        for deadline in deadlines:
            summary[f"within_{deadline:g}ms"] = grouped["total_ms"].apply(
                lambda s, limit=deadline: (s <= limit).mean()
            )
        summaries.append(summary.reset_index().assign(network=name, **_describe(network)))
    return pd.concat(summaries, ignore_index=True)


def breakeven_slowdown(summary: pd.DataFrame) -> pd.DataFrame:
    """How much slower than the measured device a device must be to gain from sending.

    For each network, takes the fastest configuration that sends anything, by
    median, and divides its median by the median of running the whole network
    locally in each precision. A device that many times slower than the one
    profiled would see the two take equally long, and offloading pays beyond
    it. This turns one laptop's measurements into a statement about devices
    in general, assuming a slower device is slower by a constant factor.

    Medians only: a deadline-driven policy should also compare the tails.
    """
    rows = []
    for network, frame in summary.groupby("network", sort=False):
        sent = frame[frame["mode"] != "local"]
        best = sent.loc[sent["median_ms"].idxmin()]
        local = frame[(frame["mode"] == "local") & (frame["point"] == EXIT_NAMES[-1])]
        for _, whole in local.iterrows():
            rows.append(
                {
                    "network": network,
                    "against": f"local {whole['device_precision']}",
                    "local_median_ms": whole["median_ms"],
                    "best_mode": best["mode"],
                    "best_point": best["point"],
                    "best_payload": best["payload"],
                    "best_server": best["server"],
                    "best_median_ms": best["median_ms"],
                    "slowdown": best["median_ms"] / whole["median_ms"],
                }
            )
    return pd.DataFrame(rows)


def _describe(network: Network) -> dict[str, float]:
    return {"rtt_ms": network.rtt_ms, "uplink_mbps": network.uplink_mbps}


def breakeven_mbps(components: pd.DataFrame, *, rtt_ms: float, local_ms: float) -> pd.DataFrame:
    """Uplink each sending configuration needs to match a local median time.

    Solves median(compute) + rtt + bytes * 8 / bandwidth = local_ms for the
    bandwidth, using median compute and median payload. Infinite where the
    round trip and computation alone already exceed the local time, so no
    bandwidth can win.
    """
    keys = ["mode", "point", "payload", "server"]
    sent = components[components["mode"] != "local"]
    medians = (
        sent.assign(compute_ms=sent[list(COMPONENTS)].sum(axis=1))
        .groupby(keys, sort=False)[["compute_ms", "payload_bytes"]]
        .median()
    )
    budget = local_ms - rtt_ms - medians["compute_ms"]
    needed = np.where(
        budget > 0, medians["payload_bytes"] * 8 / (budget.clip(lower=1e-9) * 1000), np.inf
    )
    return medians.assign(budget_ms=budget, breakeven_mbps=needed).reset_index()


def load_components(
    results: Path,
    checkpoint: str,
    *,
    threads: int,
    servers: Mapping[str, str],
) -> pd.DataFrame:
    """Read the profiles for one checkpoint from `results` and join them.

    `servers` maps the name a server gets in the table to the suffix of its
    server profile, such as {"t4": "pytorch_float32_cuda"}.
    """
    exits = {
        precision: pd.read_csv(
            results / f"exit_profile_{checkpoint}_onnxruntime_{precision}_t{threads}.csv.gz"
        )
        for precision in ("float32", "int8")
    }
    split = pd.read_csv(results / f"split_profile_{checkpoint}_float32_t{threads}.csv.gz")
    server_profiles = {
        name: pd.read_csv(results / f"server_profile_{checkpoint}_{suffix}.csv.gz")
        for name, suffix in servers.items()
    }
    return build_components(exits, split, server_profiles)
