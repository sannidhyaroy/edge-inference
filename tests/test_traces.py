import math

import numpy as np
import pandas as pd
import pytest

from builders import NETWORK, measurements
from edge_inference.offload import Network
from edge_inference.policies import Lookup, plan_family
from edge_inference.simulate import Plan, Scenario, View, frame_stream, simulate
from edge_inference.traces import DOWN, Bins, Trace, recorded_scenario


def probe_rows(rtt, rate, errors=None, link="net"):
    count = len(rtt)
    return pd.DataFrame(
        {
            "tick": np.arange(count),
            "t_s": np.arange(count) / 10,
            "link": link,
            "rtt_ms": rtt,
            "uplink_mbps": rate,
            "error": errors if errors is not None else [None] * count,
        }
    )


def test_a_failed_tick_is_a_dead_link_and_a_missing_rate_keeps_the_last_one():
    rows = probe_rows([5.0, 6.0, np.nan, 7.0], [10.0, np.nan, np.nan, -1.0],
                      [None, None, "TimeoutError", None])  # fmt: skip
    trace = Trace.from_probe(rows)
    assert trace.rtt_ms.tolist() == [5.0, 6.0, math.inf, 7.0]
    assert trace.uplink_mbps.tolist() == [10.0, 10.0, 10.0, 10.0]


def test_frames_meet_the_tick_the_recording_was_at_and_wrap_around():
    trace = Trace("net", np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]), 100.0)
    rtt, rate = trace.frames(4, period_ms=50)
    assert rtt.tolist() == [1, 1, 2, 2] and rate.tolist() == [10, 10, 20, 20]
    rtt, _ = trace.frames(5, period_ms=100)
    assert rtt.tolist() == [1, 2, 3, 1, 2]


def test_bins_cut_at_quantiles_of_the_cost_and_keep_a_state_for_a_dead_link():
    # 1000 bytes at 8 Mbps take 1 ms, so the cost is the round trip plus 1.
    rtt = np.array([*range(1, 601), math.inf], dtype=float)
    trace = Trace("net", rtt, np.full(601, 8.0), 100.0)
    bins = Bins.fit(trace, 1000, count=3, min_ticks=200)
    assert bins.names == ("fast", "middle", "slow", DOWN)
    states = bins.assign(trace.rtt_ms, trace.uplink_mbps)
    assert np.bincount(states).tolist() == [200, 200, 200, 1]
    assert bins.assign(np.array([1.0, 300.0, 599.0, math.inf]), np.full(4, 8.0)).tolist() == [
        0,
        1,
        2,
        3,
    ]
    assert bins.networks[0] == Network(100.5, 8.0)
    # Too few ticks for three states of 200: fewer states.
    assert Bins.fit(Trace("net", rtt[:300], np.full(300, 8.0), 100.0), 1000).names == (
        "steady",
        DOWN,
    )


def test_a_recording_is_refused_for_routes_on_another_link():
    trace = Trace("wifi", np.full(10, 10.0), np.full(10, 8.0), 100.0)
    bins = Bins.fit(trace, 1000, min_ticks=1)
    with pytest.raises(ValueError):
        recorded_scenario(measurements(), trace, bins, period_ms=100, deadline_ms=30, seed=0)


def test_a_steady_recording_runs_exactly_like_one_network_state():
    table = measurements()
    trace = Trace("net", np.full(40, NETWORK.rtt_ms), np.full(40, NETWORK.uplink_mbps), 100.0)
    bins = Bins.fit(trace, 1000, min_ticks=1, count=1)
    recorded = recorded_scenario(
        table, trace, bins, period_ms=100, deadline_ms=20, seed=3, passes=5
    )
    frames = frame_stream(len(table), 5 * len(table), 3)
    steady = Scenario(table, frames, np.zeros(len(frames), dtype=int), [NETWORK], ["net"],
                      slowdown=1.0, deadline_ms=20)  # fmt: skip
    for plan in plan_family(table, (0.5,), hedge=True, cascade=True):
        one, other = recorded.resolve(plan), steady.resolve(plan)
        for field in ("total_ms", "correct", "device_busy_ms", "server_at_ms", "local_ms"):
            assert np.array_equal(getattr(one, field), getattr(other, field)), (plan, field)


def test_the_lookup_plays_one_plan_per_bin_whatever_the_value_within_it():
    table = measurements()
    rtt = np.array([*np.full(50, 1.0), *np.full(50, 100.0)])
    trace = Trace("net", rtt, np.full(100, 8.0), 100.0)
    bins = Bins.fit(trace, 1000, count=2, min_ticks=10)
    rule = Lookup("ours", table, 1.0, bins=bins)
    near = rule.plan_for(Network(0.5, 8.0), 20)
    assert near == rule.plan_for(Network(2.0, 8.0), 20)
    # A near server answers in time, so the plan sends; a far or dead one does
    # not, so the plan answers locally.
    assert near.precision is None or near.otherwise != "answer"
    for network in (Network(90.0, 8.0), Network(math.inf, 8.0)):
        far = rule.plan_for(network, 20)
        assert far.precision is not None and far.otherwise == "answer" and not far.hedge


def test_rules_on_a_recording_see_the_measured_link_not_the_bin():
    table = measurements()
    trace = Trace("net", np.array([1.0, 2.0, 3.0, 4.0]), np.full(4, 8.0), 100.0)
    bins = Bins.fit(trace, 1000, count=1, min_ticks=1)
    scenario = recorded_scenario(table, trace, bins, period_ms=100, deadline_ms=20, seed=0)
    seen = []

    class Watcher:
        name = "watcher"

        def choose(self, view: View) -> Plan:
            seen.append(view.network.rtt_ms)
            return Plan()

        def observe(self, feedback) -> None:
            pass

    simulate(Watcher(), scenario)
    assert seen == [1.0, 2.0, 3.0, 4.0]


def test_a_once_a_second_recording_splits_into_halves():
    rows = pd.DataFrame({
        "second": [0, 1, 2, 3, 4], "label": "wifi", "rtt_ms": [5.0, 6.0, 7.0, np.nan, 9.0],
        "uplink_mbps": [10.0, np.nan, np.nan, np.nan, 20.0],
        "error": [None, None, None, "TimeoutError", None],
    })  # fmt: skip
    trace = Trace.from_seconds(rows, "public")
    assert (trace.link, trace.tick_ms) == ("public", 1000.0)
    first, second = trace.halves()
    assert first.rtt_ms.tolist() == [5.0, 6.0]
    assert second.rtt_ms.tolist() == [7.0, math.inf, 9.0]
    assert second.uplink_mbps.tolist() == [10.0, 10.0, 20.0]
