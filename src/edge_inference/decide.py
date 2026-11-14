"""Which way of processing a frame is best, for a device, a network and a deadline.

The per-image table from `offload` records what every option costs on the
machine that was profiled. Two things turn it into a decision:

* **A device speed.** A slower device is modelled as the profiled one slowed by
  a constant factor, applied to everything the device computes. Server and
  network terms are untouched. This is an assumption, stated as one: a real
  slow device has its own table and replaces the factor.
* **A deadline.** A frame is useful when it is answered correctly and in time.
  Accuracy and time are reported next to that score, so a reader who rejects
  the deadline can still compare the options.

A scenario is a server, a device speed, a network and a deadline. For each
scenario this module scores every option and names the best single one. It also
reports the best possible: the share of frames for which *some* option would
have been useful, chosen per frame with the answer key. No rule can reach it;
it is the ceiling the others are measured against.

**Near-ties go to the faster option.** Two options often differ by a frame or
two out of thousands, which is noise, while one takes several times longer.
Options within `TIE_MARGIN` of the top score count as tied, and the one with
the lowest mean time wins.
"""

from collections.abc import Iterable, Mapping

import pandas as pd

from edge_inference.offload import NETWORKS, Network, total_ms

# What identifies an option once the server is fixed by the scenario.
OPTION_KEYS = ["mode", "point", "device_precision", "payload"]

SCENARIO_KEYS = ["server", "slowdown", "network", "rtt_ms", "uplink_mbps", "deadline_ms"]

SLOWDOWNS = (1.0, 2.0, 4.0, 8.0, 16.0)
DEADLINES_MS = (20, 30, 50, 100)

# A tenth of a percentage point: about four frames in the 3925 profiled, well
# inside the sampling error of an accuracy near 95% on that many images (0.35
# points).
TIE_MARGIN = 0.001


def describe_option(mode: str, point: str, device_precision: str, payload: str) -> str:
    """A short readable name for an option, as the tables print it."""
    if mode == "local":
        return f"local {device_precision} to {point}"
    if mode == "split":
        return f"split after {point}, {payload}"
    return "offload"


def device_busy_ms(components: pd.DataFrame, slowdown: float = 1.0) -> pd.Series:
    """Time the device itself spends computing, on a device `slowdown` times slower."""
    return slowdown * (components["device_ms"] + components["prepare_ms"])


def frame_ms(components: pd.DataFrame, network: Network, slowdown: float = 1.0) -> pd.Series:
    """End-to-end time per row on a device `slowdown` times slower than profiled."""
    return total_ms(components, network) + (slowdown - 1.0) * device_busy_ms(components)


def for_server(components: pd.DataFrame, server: str) -> pd.DataFrame:
    """The options a device has when `server` is the one it can reach."""
    return components[components["server"].isin(("none", server))]


def score_options(
    components: pd.DataFrame,
    network: Network,
    *,
    deadline_ms: float,
    slowdown: float = 1.0,
) -> pd.DataFrame:
    """Score every option in `components` for one device, network and deadline.

    `components` must hold one server's options, as `for_server` returns. Rows
    come back sorted by useful share, then by mean time.
    """
    busy = device_busy_ms(components, slowdown)
    total = frame_ms(components, network, slowdown)
    on_time = total <= deadline_ms
    frame = components.assign(
        total_ms=total,
        device_busy_ms=busy,
        on_time=on_time,
        useful=components["correct"] & on_time,
    )
    grouped = frame.groupby(OPTION_KEYS, sort=False)
    scores = grouped.agg(
        useful=("useful", "mean"),
        accuracy=("correct", "mean"),
        on_time=("on_time", "mean"),
        mean_ms=("total_ms", "mean"),
        device_busy_ms=("device_busy_ms", "mean"),
        payload_bytes=("payload_bytes", "mean"),
    )
    scores["p95_ms"] = grouped["total_ms"].quantile(0.95)
    scores = scores.reset_index()
    scores.insert(0, "option", [describe_option(*key) for key in scores[OPTION_KEYS].to_numpy()])
    return scores.sort_values(
        ["useful", "mean_ms"], ascending=[False, True], kind="stable", ignore_index=True
    )


def pick_best(scores: pd.DataFrame, tie_margin: float = TIE_MARGIN) -> pd.Series:
    """The fastest option among those within `tie_margin` of the top useful share."""
    tied = scores[scores["useful"] >= scores["useful"].max() - tie_margin]
    return tied.loc[tied["mean_ms"].idxmin()]


def best_possible(
    components: pd.DataFrame,
    network: Network,
    *,
    deadline_ms: float,
    slowdown: float = 1.0,
) -> float:
    """Share of frames for which at least one option is correct and in time.

    This picks per frame knowing which options answer correctly, which no
    device can know. It is a ceiling, not a rule.
    """
    on_time = frame_ms(components, network, slowdown) <= deadline_ms
    useful = components["correct"] & on_time
    return float(useful.groupby(components["image"]).max().mean())


def decision_map(
    components: pd.DataFrame,
    *,
    slowdowns: Iterable[float] = SLOWDOWNS,
    networks: Mapping[str, Network] = NETWORKS,
    deadlines: Iterable[float] = DEADLINES_MS,
    tie_margin: float = TIE_MARGIN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every option in every scenario, and name the best one in each.

    Returns two tables. The first has one row per scenario per option, with
    the chosen one flagged. The second has one row per scenario: the best
    single option, its scores, and the best possible per frame.
    """
    slowdowns, deadlines = tuple(slowdowns), tuple(deadlines)
    servers = [name for name in components["server"].unique() if name != "none"]

    options, best = [], []
    for server in servers:
        reachable = for_server(components, server)
        for slowdown in slowdowns:
            for name, network in networks.items():
                for deadline in deadlines:
                    scenario = {
                        "server": server,
                        "slowdown": slowdown,
                        "network": name,
                        "rtt_ms": network.rtt_ms,
                        "uplink_mbps": network.uplink_mbps,
                        "deadline_ms": deadline,
                    }
                    scores = score_options(
                        reachable, network, deadline_ms=deadline, slowdown=slowdown
                    )
                    chosen = pick_best(scores, tie_margin)
                    options.append(scores.assign(**scenario, best=scores.index == chosen.name))
                    ceiling = best_possible(
                        reachable, network, deadline_ms=deadline, slowdown=slowdown
                    )
                    best.append({**scenario, **chosen.to_dict(), "best_possible": ceiling})

    every = pd.concat(options, ignore_index=True)
    every = every[SCENARIO_KEYS + [c for c in every.columns if c not in SCENARIO_KEYS]]
    return every, pd.DataFrame(best)
