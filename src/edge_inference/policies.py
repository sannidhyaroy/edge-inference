"""Decision rules, and the sets of plans they and the yardsticks choose from.

A rule returns one `Plan` per frame. The simple rules here ignore everything
they are shown and always return the same plan; they are the baselines every
later rule has to beat.

Two sets of plans are defined:

* `fixed_plans`: the plain options, one per row of the per-image table. Run
  locally to one exit, send the feature map at one split point, or send the
  photo.
* `plan_family`: the fixed plans plus every plan that uses confidence. Stop at
  an exit when it is confident enough; and at the deepest exit run, if still
  not confident, answer anyway, send the feature map, or send the photo.

Whatever a plan in the family does to a frame, some fixed plan does the same
or better on that frame: stopping at an exit is "run locally to that exit",
and sending after a failed check is a fixed send plus wasted device time. That
is why choosing among fixed plans per frame, with the answer key, bounds every
rule.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from edge_inference.decide import TIE_MARGIN
from edge_inference.offload import Network
from edge_inference.simulate import (
    FEATURES,
    FINAL,
    PAYLOADS,
    PHOTO,
    PRECISIONS,
    Feedback,
    Measurements,
    Plan,
    View,
    resolve,
)
from edge_inference.split import SPLIT_NAMES

OFFLOAD = Plan()

# 0.05 to 0.95. Wide on purpose: the early heads are underconfident, so useful
# thresholds sit well below 0.5.
THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))


def fixed_plans() -> list[Plan]:
    """One plan per plain option: no confidence checks anywhere."""
    plans = [OFFLOAD]
    plans += [Plan(precision, depth) for precision in PRECISIONS for depth in range(FINAL + 1)]
    plans += [
        Plan("float32", depth, otherwise=FEATURES, payload=payload)
        for depth in range(len(SPLIT_NAMES))
        for payload in PAYLOADS
    ]
    return plans


def plan_family(thresholds: Sequence[float] = THRESHOLDS) -> list[Plan]:
    """Every fixed plan, and every plan that stops or sends on confidence."""
    levels = (*thresholds, math.inf)
    plans = [OFFLOAD]
    for precision in PRECISIONS:
        for depth in range(FINAL + 1):
            for earlier in product(levels, repeat=depth):
                plans.append(Plan(precision, depth, earlier))
                if depth == FINAL:
                    continue
                for last in levels:
                    plans.append(Plan(precision, depth, (*earlier, last), otherwise=PHOTO))
                    if precision == "float32" and depth < len(SPLIT_NAMES):
                        plans += [
                            Plan(precision, depth, (*earlier, last), FEATURES, payload)
                            for payload in PAYLOADS
                        ]
    return plans


@dataclass
class Always:
    """The same plan for every frame, whatever the network."""

    name: str
    plan: Plan

    def choose(self, view: View) -> Plan:
        return self.plan

    def observe(self, feedback: Feedback) -> None:
        pass


def best_plan(
    measurements: Measurements,
    plans: Sequence[Plan],
    network: Network,
    *,
    slowdown: float,
    deadline_ms: float,
    tie_margin: float = TIE_MARGIN,
) -> Plan:
    """The fastest plan among those within `tie_margin` of the top useful share.

    Every image in `measurements` meets the same `network`. This is how
    settings are chosen on the tuning images.
    """
    rows = np.arange(len(measurements))
    useful, mean_ms = [], []
    for plan in plans:
        outcome = resolve(measurements, plan, rows, network.rtt_ms, network.uplink_mbps, slowdown)
        useful.append((outcome.correct & (outcome.total_ms <= deadline_ms)).mean())
        mean_ms.append(outcome.total_ms.mean())
    useful, mean_ms = np.array(useful), np.array(mean_ms)
    tied = np.flatnonzero(useful >= useful.max() - tie_margin)
    return plans[tied[mean_ms[tied].argmin()]]


def tuned_threshold(
    measurements: Measurements,
    precision: str,
    network: Network,
    *,
    slowdown: float,
    deadline_ms: float,
    thresholds: Sequence[float] = THRESHOLDS,
) -> Always:
    """Early exit with one confidence threshold for every exit, chosen on tuning images.

    This is confidence-based early exit as it is usually described: a single
    threshold, the same at every exit, and no sending.
    """
    plans = [Plan(precision, FINAL, (t,) * FINAL) for t in thresholds]
    plan = best_plan(measurements, plans, network, slowdown=slowdown, deadline_ms=deadline_ms)
    return Always(f"early exit {precision}, threshold {plan.thresholds[0]:g}", plan)


def simple_rules(
    tuning: Measurements, network: Network, *, slowdown: float, deadline_ms: float
) -> list[Always]:
    """The baselines: always one plain option, or early exit with one threshold."""
    rules = [
        Always("always local float32", Plan("float32")),
        Always("always local int8", Plan("int8")),
        Always("always offload", OFFLOAD),
    ]
    rules += [
        tuned_threshold(tuning, precision, network, slowdown=slowdown, deadline_ms=deadline_ms)
        for precision in PRECISIONS
    ]
    return rules


@dataclass
class Lookup:
    """Measure first, then look the answer up.

    Before each frame the rule takes the network as it last saw it and plays
    the plan that did best under that network on the tuning images: the most
    useful, and among near-ties the fastest. The plan comes from the whole
    family, so it carries a threshold per exit and may fall back on sending
    the photo or the feature map.

    It needs what a bandit does not: a measuring phase on labelled images, on
    this device. In return it changes its choice the moment the network does.

    A rule that only learns the network by sending can get stuck: once it
    believes the network is bad it stops sending, and so never finds out that
    it recovered. With `probe_every`, a frame is sent anyway once the last
    sighting of the network is that many frames old. A frame is also sent when
    nothing is known at all, since sending both answers and reveals.
    """

    name: str
    tuning: Measurements
    slowdown: float
    plans: Sequence[Plan] = field(default_factory=plan_family)
    tie_margin: float = TIE_MARGIN
    probe_every: int | None = None
    table: dict[tuple[float, float, float], Plan] = field(default_factory=dict)

    def plan_for(self, network: Network, deadline_ms: float) -> Plan:
        """The plan to play under `network`, worked out once and then remembered."""
        key = (network.rtt_ms, network.uplink_mbps, deadline_ms)
        if key not in self.table:
            self.table[key] = best_plan(
                self.tuning,
                self.plans,
                network,
                slowdown=self.slowdown,
                deadline_ms=deadline_ms,
                tie_margin=self.tie_margin,
            )
        return self.table[key]

    def choose(self, view: View) -> Plan:
        stale = self.probe_every is not None and view.network_age >= self.probe_every
        if view.network is None or stale:
            return OFFLOAD
        return self.plan_for(view.network, view.deadline_ms)

    def observe(self, feedback: Feedback) -> None:
        pass
