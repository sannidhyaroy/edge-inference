"""Run every rule on one scenario and line them up against the yardsticks.

A `Setup` names everything that defines a run: the server, the device speed,
the deadline, the two networks and how long each lasts, what rules know of the
network, and the seed. `run_setup` builds the frames from it, tunes each rule
on images and a network sequence the test never sees, runs them, and returns
the scoreboard with one row per rule and per yardstick.

`run_sweep` does that for many setups at once, one process per core, and
stacks the scoreboards. One scenario shows what happened; a sweep over device
speeds, deadlines and seeds shows where it holds.

With `hedge` set, two rules and two yardsticks join the board. "hedge, no
network knowledge" plays one hedged plan on every frame, chosen on the tuning
run without regard to the network. "lookup with hedging" is the lookup rule
over a family that includes hedged plans. Rung 3h is the best plan per
network over that family, and rung 4r the best possible record per frame.
"""

import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from itertools import product

import numpy as np
import pandas as pd

from edge_inference.bandit import offload_cost_in_layers, variants
from edge_inference.config import SEED
from edge_inference.decide import TIE_MARGIN
from edge_inference.offload import NETWORKS
from edge_inference.policies import Always, Lookup, plan_family, simple_rules
from edge_inference.simulate import (
    HORIZON_MS,
    Measurements,
    Plan,
    Rule,
    Scenario,
    View,
    frame_stream,
    markov_states,
    score_trace,
    simulate,
    split_images,
)
from edge_inference.yardsticks import ladder_runs, score_run

FAIR_TARGET = "3. best confidence rule per network"
OURS = "lookup (ours)"
HEDGE = "hedge, no network knowledge"
HEDGED_TARGET = "3h. best plan per network, hedges included"
OURS_HEDGED = "lookup with hedging"

# How old a sighting of the network may get before a frame is sent to refresh
# it, tried when sending is the only way the rule learns the network.
PROBE_INTERVALS = (None, 2, 5, 10, 20, 50)


@dataclass(frozen=True)
class Setup:
    """Everything that defines one run."""

    server: str = "t4"
    slowdown: float = 8.0
    deadline_ms: float = 30.0
    good: str = "wifi"
    bad: str = "4g"
    stay_good: float = 0.95
    stay_bad: float = 0.95
    knowledge: str = "exact"
    passes: int = 5
    tune_share: float = 0.5
    seed: int = SEED
    hedge: bool = False
    cancel: bool = False
    horizon_ms: float = HORIZON_MS
    price_busy: float = 0.0
    price_kb: float = 0.0

    @property
    def prices(self) -> tuple[float, float]:
        return (self.price_busy, self.price_kb)

    @property
    def names(self) -> list[str]:
        """The network states: one when the network is steady, else good then bad."""
        return [self.good] if self.good == self.bad else [self.good, self.bad]

    def tag(self) -> str:
        """A file-name fragment that tells runs apart."""
        tag = f"{self.server}_{self.slowdown:g}x_{self.deadline_ms:g}ms_{self.good}-{self.bad}"
        if len(self.names) > 1:
            tag += f"_stay{self.stay_good:g}-{self.stay_bad:g}"
        if self.knowledge != "exact":
            tag += f"_{self.knowledge.replace(':', '')}"
        return tag + self.extras()

    def extras(self) -> str:
        """The part of the tag for hedging, the horizon and prices; empty by default."""
        tag = ""
        if self.hedge:
            tag += "_hedge" + ("-cancel" if self.cancel else "")
        if self.horizon_ms != HORIZON_MS:
            tag += f"_H{self.horizon_ms:g}"
        if self.price_busy or self.price_kb:
            tag += f"_price{self.price_busy:g}-{self.price_kb:g}"
        return tag

    def scenario(self, images: Measurements, seed: int) -> Scenario:
        frames = frame_stream(len(images), self.passes * len(images), seed)
        if len(self.names) == 1:
            states = np.zeros(len(frames), dtype=int)
        else:
            states = markov_states(len(frames), (self.stay_good, self.stay_bad), seed)
        return Scenario(
            images,
            frames,
            states,
            [NETWORKS[name] for name in self.names],
            self.names,
            slowdown=self.slowdown,
            deadline_ms=self.deadline_ms,
            horizon_ms=self.horizon_ms,
            cancel=self.cancel,
        )


@dataclass(frozen=True)
class Result:
    """A scoreboard, the per-frame trace behind it, and the scenario it ran on."""

    board: pd.DataFrame
    traces: pd.DataFrame
    scenario: Scenario
    test_images: int
    held_out: bool


def tuned_lookup(
    tuning: Measurements,
    tuning_scenario: Scenario,
    setup: Setup,
    *,
    name: str = OURS,
    hedge: bool = False,
) -> Lookup:
    """Our rule, with its probing interval chosen where probing is needed.

    The plans themselves come from the tuning images. The interval matters
    only when the rule learns the network from its own sends; it is chosen by
    running the rule on the tuning scenario. With `hedge` the rule chooses
    from a family that includes hedged plans.
    """
    rule = Lookup(
        name,
        tuning,
        setup.slowdown,
        plans=plan_family(tuning, hedge=hedge),
        prices=setup.prices,
        cancel=setup.cancel,
    )
    if setup.knowledge != "sends":
        return rule
    useful = []
    for interval in PROBE_INTERVALS:
        rule.probe_every = interval
        trace = simulate(rule, tuning_scenario, setup.knowledge)
        useful.append(score_trace(trace, setup.deadline_ms, setup.horizon_ms)["useful"])
    rule.probe_every = PROBE_INTERVALS[int(np.argmax(useful))]
    return rule


def tuned_hedge(tuning_scenario: Scenario, tie_margin: float = TIE_MARGIN) -> Always:
    """The hedged plan that does best over the whole tuning run, whatever the network.

    It is chosen knowing nothing about which network each frame met: one plan,
    played on every frame. Near-ties go to the faster plan.
    """
    hedged = [plan for plan in plan_family(tuning_scenario.measurements, hedge=True) if plan.hedge]
    useful, mean_ms = [], []
    for plan in hedged:
        outcome = tuning_scenario.resolve(plan)
        useful.append((outcome.correct & (outcome.total_ms <= tuning_scenario.deadline_ms)).mean())
        mean_ms.append(outcome.total_ms.mean())
    useful, mean_ms = np.array(useful), np.array(mean_ms)
    tied = np.flatnonzero(useful >= useful.max() - tie_margin)
    plan: Plan = hedged[tied[mean_ms[tied].argmin()]]
    return Always(HEDGE, plan)


def decision_microseconds(rule: Rule, scenario: Scenario, repeats: int = 20000) -> float:
    """Median time the rule takes to decide on one frame, once its table is built."""
    views = [View(0, scenario.deadline_ms, network, network_age=0) for network in scenario.networks]
    for view in views:
        rule.choose(view)
    times = []
    for index in range(repeats):
        view = views[index % len(views)]
        start = time.perf_counter_ns()
        rule.choose(view)
        times.append(time.perf_counter_ns() - start)
    return float(np.median(times)) / 1000.0


def run_setup(table: Measurements, setup: Setup, *, every_variant: bool = True) -> Result:
    """Tune, run and score every rule for one setup."""
    if setup.tune_share > 0:
        held = split_images(table.labels, setup.tune_share, setup.seed)
        tuning, test = table.subset(held), table.subset(~held)
    else:
        # For reproducing whole-set figures. Settings are then chosen on the
        # images they are scored on, which flatters any tuned rule.
        tuning = test = table

    scenario = setup.scenario(test, setup.seed)
    # Rules that learn are tuned by running them, so they get a stream of their
    # own: the tuning images under a network sequence from another seed.
    tuning_scenario = setup.scenario(tuning, setup.seed + 1)
    frames, states = scenario.frames, scenario.states
    names = np.array(setup.names)

    def as_trace(name: str, run: dict[str, np.ndarray]) -> pd.DataFrame:
        correct = run["correct"].astype(bool)
        return pd.DataFrame(
            {
                "frame": np.arange(len(frames)),
                "image": scenario.measurements.images[frames],
                "network": names[states],
                "sent_bytes": run["sent_bytes"],
                "device_busy_ms": run["device_busy_ms"],
                "total_ms": run["total_ms"],
                "correct": correct,
                "local_correct": run["local_correct"].astype(bool),
                "server_at_ms": run["server_at_ms"],
                "server_correct": run["server_correct"].astype(bool),
                "useful": correct & (run["total_ms"] <= setup.deadline_ms),
                "rule": name,
            }
        )

    rows, traces = [], []

    def add_rule(rule: Rule, plans: str, **extra: object) -> None:
        trace = simulate(rule, scenario, setup.knowledge)
        traces.append(trace.assign(rule=rule.name))
        scores = score_trace(trace, setup.deadline_ms, setup.horizon_ms)
        rows.append({"rule": rule.name, "kind": "rule", "plans": plans, **scores, **extra})

    simple = simple_rules(
        tuning, scenario.networks[0], slowdown=setup.slowdown, deadline_ms=setup.deadline_ms
    )
    for rule in simple:
        add_rule(rule, rule.plan.describe())

    for variant in variants(tuning_scenario, scenario, every=every_variant):
        traces.append(as_trace(variant.name, variant.outcome).assign(plan=variant.played()))
        scores = score_run(variant.outcome, setup.deadline_ms, setup.horizon_ms)
        rows.append(
            {"rule": variant.name, "kind": variant.kind, "plans": variant.describe(), **scores}
        )

    ours = tuned_lookup(tuning, tuning_scenario, setup)
    chosen = "; ".join(
        f"{name}: {ours.plan_for(network, setup.deadline_ms).describe()}"
        for name, network in zip(setup.names, scenario.networks, strict=True)
    )
    if ours.probe_every is not None:
        chosen += f"; a frame is sent when the network was last seen {ours.probe_every} frames ago"
    add_rule(ours, chosen, decide_us=decision_microseconds(ours, scenario))

    if setup.hedge:
        hedge = tuned_hedge(tuning_scenario)
        add_rule(hedge, hedge.plan.describe())
        hedged = tuned_lookup(tuning, tuning_scenario, setup, name=OURS_HEDGED, hedge=True)
        chosen = "; ".join(
            f"{name}: {hedged.plan_for(network, setup.deadline_ms).describe()}"
            for name, network in zip(setup.names, scenario.networks, strict=True)
        )
        add_rule(hedged, chosen, decide_us=decision_microseconds(hedged, scenario))

    for name, plans, run in ladder_runs(scenario, hedge=setup.hedge):
        traces.append(as_trace(name, run))
        scores = score_run(run, setup.deadline_ms, setup.horizon_ms)
        rows.append({"rule": name, "kind": "yardstick", "plans": plans, **scores})

    board = pd.DataFrame(rows)
    fair = board.loc[board["rule"] == FAIR_TARGET, "useful"].iloc[0]
    board["behind_fair"] = fair - board["useful"]
    layers = offload_cost_in_layers(scenario)
    board = board.assign(
        **asdict(setup),
        frames=len(frames),
        **{f"network_share_{name}": float(np.mean(states == i)) for i, name in enumerate(names)},
        **{f"offload_layers_{name}": cost for name, cost in layers.items()},
    )
    return Result(
        board=board,
        traces=pd.concat(traces, ignore_index=True),
        scenario=scenario,
        test_images=len(test),
        held_out=setup.tune_share > 0,
    )


def grid(
    base: Setup,
    *,
    slowdowns: Sequence[float],
    deadlines: Sequence[float],
    seeds: int,
) -> list[Setup]:
    """Every device speed with every deadline, each under `seeds` different seeds.

    Seeds are spaced well apart: a setup tunes on the seed after its own, and
    neighbouring runs should not share a network sequence.
    """
    return [
        replace(base, slowdown=slowdown, deadline_ms=deadline, seed=base.seed + 1000 * index)
        for slowdown, deadline, index in product(slowdowns, deadlines, range(seeds))
    ]


_TABLE: Measurements | None = None


def _share_table(table: Measurements) -> None:
    global _TABLE
    _TABLE = table


def _board(setup: Setup) -> pd.DataFrame:
    return run_setup(_TABLE, setup, every_variant=False).board


def run_sweep(table: Measurements, setups: Sequence[Setup], workers: int = 1) -> pd.DataFrame:
    """Score every setup and stack the scoreboards, `workers` setups at a time.

    Only SplitEE as published and with every repair are run, since tuning all
    of its variants takes most of a scoreboard's time.
    """
    if workers <= 1:
        _share_table(table)
        boards = [_board(setup) for setup in setups]
    else:
        with ProcessPoolExecutor(workers, initializer=_share_table, initargs=(table,)) as pool:
            boards = list(pool.map(_board, setups))
    return pd.concat(boards, ignore_index=True)
