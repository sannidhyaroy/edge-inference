"""Scoreboards on a recorded link: rules tuned on one recording, scored on another.

The tuning half of the images meets the tuning recording; the test half meets
a recording made at another time over the same link. Network states are cut
from the tuning recording only. Every rule and yardstick plays the same test
frames over the same recorded values, so differences between them are paired.

The rules:

* every plain option, played on every frame;
* our lookup rule, playing per network state the best plan on the tuning half,
  from a family that includes cascades (answer locally unless unsure, then
  send the photo);
* the best single hedge on the tuning run, which knows nothing of the network;
* the lookup rule with hedged plans in its family.

The yardsticks are those of `yardsticks`, with cascades in rungs 3 and 3h.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from edge_inference.config import SEED
from edge_inference.decide import TIE_MARGIN
from edge_inference.policies import Always, Lookup, plan_family
from edge_inference.scoreboard import OURS, OURS_HEDGED, PROBE_INTERVALS, tuned_hedge
from edge_inference.simulate import (
    HORIZON_MS,
    NEVER,
    Measurements,
    Plan,
    Scenario,
    score,
    score_trace,
    simulate,
)
from edge_inference.traces import Bins, Trace, recorded_scenario
from edge_inference.yardsticks import ladder_runs, score_run

FAIR_TARGET = "3. best confidence rule per network"
MIN_STATE_MS = 20_000.0


@dataclass(frozen=True)
class RecordedSetup:
    """Everything that defines one run on a recorded link."""

    deadline_ms: float
    period_ms: float = 100.0
    horizon_ms: float = HORIZON_MS
    knowledge: str = "exact"
    cancel: str = NEVER
    bins: int = 3
    passes: int = 1
    seed: int = SEED
    price_busy: float = 0.0
    price_kb: float = 0.0
    # Rules play measured device times, so nothing is scaled.
    slowdown: float = 1.0

    @property
    def prices(self) -> tuple[float, float]:
        return (self.price_busy, self.price_kb)

    def tag(self) -> str:
        tag = f"{self.deadline_ms:g}ms"
        if self.knowledge != "exact":
            tag += f"_{self.knowledge.replace(':', '')}"
        if self.cancel != NEVER:
            tag += f"_cancel-{self.cancel}"
        if self.horizon_ms != HORIZON_MS:
            tag += f"_H{self.horizon_ms:g}"
        if self.price_busy or self.price_kb:
            tag += f"_price{self.price_busy:g}-{self.price_kb:g}"
        return tag


def reference_bytes(measurements: Measurements) -> float:
    """The median photo over every photo route: what network states are cut by."""
    return float(np.median(np.concatenate([r.bytes for r in measurements.routes.values()])))


def _lookup(
    name: str,
    tuning: Measurements,
    tuning_scenario: Scenario,
    setup: RecordedSetup,
    bins: Bins,
    *,
    hedge: bool,
) -> Lookup:
    rule = Lookup(
        name,
        tuning,
        setup.slowdown,
        plans=plan_family(tuning, hedge=hedge, cascade=True),
        tie_margin=TIE_MARGIN,
        prices=setup.prices,
        cancel=setup.cancel,
        bins=bins,
    )
    if setup.knowledge == "sends":
        useful = []
        for interval in PROBE_INTERVALS:
            rule.probe_every = interval
            trace = simulate(rule, tuning_scenario, setup.knowledge)
            useful.append(score_trace(trace, setup.deadline_ms, setup.horizon_ms)["useful"])
        rule.probe_every = PROBE_INTERVALS[int(np.argmax(useful))]
    return rule


def run_recorded(
    tuning: Measurements,
    test: Measurements,
    tuning_trace: Trace,
    test_trace: Trace,
    setup: RecordedSetup,
) -> pd.DataFrame:
    """Tune, run and score every rule and yardstick; one row each."""
    # Each state needs at least 20 s of the tuning recording, whatever its tick.
    min_ticks = max(1, round(MIN_STATE_MS / tuning_trace.tick_ms))
    bins = Bins.fit(tuning_trace, reference_bytes(tuning), count=setup.bins, min_ticks=min_ticks)

    def scenario_for(table: Measurements, trace: Trace, seed: int) -> Scenario:
        return recorded_scenario(
            table, trace, bins, period_ms=setup.period_ms, deadline_ms=setup.deadline_ms,
            seed=seed, passes=setup.passes, horizon_ms=setup.horizon_ms, cancel=setup.cancel,
        )  # fmt: skip

    scenario = scenario_for(test, test_trace, setup.seed)
    tuning_scenario = scenario_for(tuning, tuning_trace, setup.seed + 1)
    rows = []

    def plain(name: str, plan: Plan) -> None:
        outcome = scenario.resolve(plan)
        scores = score(
            outcome.total_ms, outcome.correct, outcome.device_busy_ms, outcome.sent_bytes,
            setup.deadline_ms, local_correct=outcome.local_correct,
            server_at_ms=outcome.server_at_ms, server_correct=outcome.server_correct,
            horizon_ms=setup.horizon_ms,
        )  # fmt: skip
        rows.append({"rule": name, "kind": "rule", "plans": plan.describe(), **scores})

    for variant in test.local:
        plain(f"always local {variant}", Plan(variant))
    for route in test.photo_routes:
        plain(f"always offload via {route}", Plan(route=route))

    def by_rule(rule: Lookup | Always, plans: str) -> None:
        trace = simulate(rule, scenario, setup.knowledge)
        scores = score_trace(trace, setup.deadline_ms, setup.horizon_ms)
        rows.append({"rule": rule.name, "kind": "rule", "plans": plans, **scores})

    def chosen(rule: Lookup) -> str:
        text = "; ".join(
            f"{name}: {rule.plan_for(network, setup.deadline_ms).describe()}"
            for name, network in zip(bins.names, bins.networks, strict=True)
        )
        if rule.probe_every is not None:
            text += f"; a frame is sent when the network was last seen {rule.probe_every} ago"
        return text

    ours = _lookup(OURS, tuning, tuning_scenario, setup, bins, hedge=False)
    by_rule(ours, chosen(ours))
    hedge = tuned_hedge(tuning_scenario)
    by_rule(hedge, hedge.plan.describe())
    hedged = _lookup(OURS_HEDGED, tuning, tuning_scenario, setup, bins, hedge=True)
    by_rule(hedged, chosen(hedged))

    for name, plans, run in ladder_runs(scenario, hedge=True, cascade=True):
        rows.append({"rule": name, "kind": "yardstick", "plans": plans,
                     **score_run(run, setup.deadline_ms, setup.horizon_ms)})  # fmt: skip

    board = pd.DataFrame(rows)
    fair = board.loc[board["rule"] == FAIR_TARGET, "useful"].iloc[0]
    board["behind_fair"] = fair - board["useful"]
    states = scenario.states
    return board.assign(
        **asdict(setup),
        link=test_trace.link,
        frames=len(scenario.frames),
        bin_edges_ms=";".join(f"{edge:.2f}" for edge in bins.edges),
        **{f"network_share_{name}": float(np.mean(states == i))
           for i, name in enumerate(bins.names)},
    )  # fmt: skip
