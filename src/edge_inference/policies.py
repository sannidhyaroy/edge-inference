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
rule. A hedged plan is no exception: by its deadline it has either the
server's answer, which sending alone would have had as soon, or its own,
which running alone would have had no later.

Both sets are read off the table: its variants, how many exits each has, and
which photo and feature routes it holds.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np

from edge_inference.decide import TIE_MARGIN
from edge_inference.offload import Network
from edge_inference.simulate import (
    ANSWER,
    FEATURES,
    PHOTO,
    Feedback,
    Measurements,
    Plan,
    View,
    resolve,
)

OFFLOAD = Plan()

# 0.05 to 0.95. Wide on purpose: the early heads are underconfident, so useful
# thresholds sit well below 0.5.
THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))


def _photo_routes(measurements: Measurements) -> list[str | None]:
    """How plans name the photo routes: not at all when there is only one."""
    keys = measurements.photo_routes
    return [None] if len(keys) == 1 else keys


def _depths(measurements: Measurements, variant: str) -> list[int | None]:
    """A variant's exits, shallowest first, with the final exit as `None`."""
    final = measurements.final(variant)
    return [*range(final), None]


def _features_after(measurements: Measurements, variant: str, exit: int) -> list[str]:
    """The feature payloads profiled after `exit` of `variant`, in table order."""
    return [
        route.payload for route in measurements.routes.values() if route.after == (variant, exit)
    ]


def fixed_plans(measurements: Measurements) -> list[Plan]:
    """One plan per plain option: no confidence checks anywhere."""
    plans = [Plan(route=route) for route in _photo_routes(measurements)]
    plans += [
        Plan(variant, depth)
        for variant in measurements.local
        for depth in _depths(measurements, variant)
    ]
    plans += [
        Plan(route.after[0], route.after[1], otherwise=FEATURES, payload=route.payload)
        for route in measurements.routes.values()
        if route.after is not None
    ]
    return plans


def plan_family(
    measurements: Measurements, thresholds: Sequence[float] = THRESHOLDS, *, hedge: bool = False
) -> list[Plan]:
    """Every fixed plan, and every plan that stops or sends on confidence.

    With `hedge`, every plan that ends in a local answer also comes in a hedged
    form, sending the photo by each route as it starts.
    """
    levels = (*thresholds, math.inf)
    routes = _photo_routes(measurements)
    plans = [Plan(route=route) for route in routes]
    for variant in measurements.local:
        final = measurements.final(variant)
        for depth in range(final + 1):
            for earlier in product(levels, repeat=depth):
                plans.append(Plan(variant, None if depth == final else depth, earlier))
                if depth == final:
                    continue
                for last in levels:
                    checks = (*earlier, last)
                    plans += [Plan(variant, depth, checks, PHOTO, route=r) for r in routes]
                    plans += [
                        Plan(variant, depth, checks, FEATURES, payload)
                        for payload in _features_after(measurements, variant, depth)
                    ]
    if hedge:
        plans += [
            replace(plan, route=route, hedge=True)
            for plan in list(plans)
            if plan.precision is not None and plan.otherwise == ANSWER
            for route in routes
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
    prices: tuple[float, float] = (0.0, 0.0),
    cancel: bool = False,
) -> Plan:
    """The plan with the best useful share, net of `prices`, on the tuning images.

    Every image in `measurements` meets the same `network`. This is how
    settings are chosen on the tuning images. `prices` charges for device time
    (per ms) and for bytes sent (per kB), in units of useful share: with both
    at zero, which is the default, only the useful share counts. Plans within
    `tie_margin` of the best count as tied; among them the cheapest wins, or
    with no prices the fastest.
    """
    rows = np.arange(len(measurements))
    per_ms, per_kb = prices
    value, tiebreak = [], []
    for plan in plans:
        outcome = resolve(
            measurements,
            plan,
            rows,
            network.rtt_ms,
            network.uplink_mbps,
            slowdown,
            deadline_ms=deadline_ms,
            cancel=cancel,
        )
        useful = (outcome.correct & (outcome.total_ms <= deadline_ms)).mean()
        cost = per_ms * outcome.device_busy_ms.mean() + per_kb * outcome.sent_bytes.mean() / 1000
        value.append(useful - cost)
        tiebreak.append(cost if per_ms or per_kb else outcome.total_ms.mean())
    value, tiebreak = np.array(value), np.array(tiebreak)
    tied = np.flatnonzero(value >= value.max() - tie_margin)
    return plans[tied[tiebreak[tied].argmin()]]


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
    earlier = measurements.final(precision)
    plans = [Plan(precision, None, (t,) * earlier) for t in thresholds]
    plan = best_plan(measurements, plans, network, slowdown=slowdown, deadline_ms=deadline_ms)
    return Always(f"early exit {precision}, threshold {plan.thresholds[0]:g}", plan)


def simple_rules(
    tuning: Measurements, network: Network, *, slowdown: float, deadline_ms: float
) -> list[Always]:
    """The baselines: always one plain option, or early exit with one threshold."""
    rules = [Always(f"always local {variant}", Plan(variant)) for variant in tuning.local]
    rules += [
        Always("always offload" + (f" via {route}" if route else ""), Plan(route=route))
        for route in _photo_routes(tuning)
    ]
    rules += [
        tuned_threshold(tuning, variant, network, slowdown=slowdown, deadline_ms=deadline_ms)
        for variant in tuning.local
        if tuning.final(variant) > 0
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
    plans: Sequence[Plan] | None = None
    tie_margin: float = TIE_MARGIN
    probe_every: int | None = None
    table: dict[tuple[float, float, float], Plan] = field(default_factory=dict)
    prices: tuple[float, float] = (0.0, 0.0)
    cancel: bool = False

    def __post_init__(self) -> None:
        if self.plans is None:
            self.plans = plan_family(self.tuning)

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
                prices=self.prices,
                cancel=self.cancel,
            )
        return self.table[key]

    def choose(self, view: View) -> Plan:
        stale = self.probe_every is not None and view.network_age >= self.probe_every
        if view.network is None or stale:
            return Plan(route=_photo_routes(self.tuning)[0])
        return self.plan_for(view.network, view.deadline_ms)

    def observe(self, feedback: Feedback) -> None:
        pass
