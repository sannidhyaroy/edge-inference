"""Checks for the sets of plans and the simple rules, on the table in `builders.py`."""

import numpy as np

from builders import NETWORK, measurements
from edge_inference.offload import Network
from edge_inference.policies import (
    OFFLOAD,
    Always,
    Lookup,
    best_plan,
    fixed_plans,
    plan_family,
    simple_rules,
    tuned_threshold,
)
from edge_inference.simulate import Plan, View, resolve

ROWS = np.arange(4)
BAD = Network(rtt_ms=100, uplink_mbps=8)


def behaviour(plan):
    outcome = resolve(measurements(), plan, ROWS, NETWORK.rtt_ms, NETWORK.uplink_mbps, 2.0)
    return (*outcome.total_ms, *outcome.correct, *outcome.sent_bytes)


def test_there_is_one_fixed_plan_per_plain_option():
    # Offload, 3 exits x 2 precisions, 2 split points x 2 payloads.
    assert len(fixed_plans(measurements())) == 1 + 6 + 4
    assert OFFLOAD in fixed_plans(measurements())


def test_the_family_contains_every_fixed_plan():
    family = {behaviour(plan) for plan in plan_family(measurements(), (0.5,))}

    assert all(behaviour(plan) in family for plan in fixed_plans(measurements()))


def test_the_family_grows_with_the_thresholds():
    assert len(plan_family(measurements(), (0.5,))) < len(plan_family(measurements(), (0.3, 0.5)))


def test_a_simple_rule_always_returns_its_plan():
    rule = Always("always offload", OFFLOAD)

    assert rule.choose(view=None) is OFFLOAD


def test_the_best_plan_is_the_most_useful_then_the_fastest():
    table = measurements()
    plans = [Plan("int8", 0), Plan("int8", 1), Plan("int8"), OFFLOAD]

    # With 20 ms everything is in time. Offloading is right on all four
    # images, the final exit on three.
    chosen = best_plan(table, plans, NETWORK, slowdown=1.0, deadline_ms=20, tie_margin=0.0)
    assert chosen is OFFLOAD
    # A margin of a quarter lets the final exit tie, and it takes 3 ms, not 15.
    chosen = best_plan(table, plans, NETWORK, slowdown=1.0, deadline_ms=20, tie_margin=0.25)
    assert chosen == Plan("int8")


def test_the_threshold_is_chosen_on_the_tuning_images():
    rule = tuned_threshold(
        measurements(), "int8", NETWORK, slowdown=1.0, deadline_ms=20, thresholds=(0.1, 0.85)
    )

    # At 0.1 every image stops at the first exit, right once. At 0.85 images
    # stop where they are first right, and three of four are answered correctly.
    assert rule.plan == Plan("int8", None, (0.85, 0.85))
    assert rule.name == "early exit int8, threshold 0.85"


def test_the_baselines_are_named():
    rules = simple_rules(measurements(), NETWORK, slowdown=1.0, deadline_ms=20)

    assert [rule.name for rule in rules[:3]] == [
        "always local float32",
        "always local int8",
        "always offload",
    ]
    assert len(rules) == 5


def lookup(**settings):
    return Lookup("lookup", measurements(), slowdown=3.0, **settings)


def test_the_lookup_rule_plays_a_different_plan_per_network():
    rule = lookup()

    # At 3x with 16 ms the photo (15 ms) is in time on the good network and
    # right on all four images. On the bad network nothing sent is in time,
    # and running int8 to the end is right on three.
    assert rule.plan_for(NETWORK, deadline_ms=16) == OFFLOAD
    local = rule.plan_for(BAD, deadline_ms=16)
    assert local.precision == "int8" and local.depth is None

    outcome = resolve(measurements(), local, ROWS, BAD.rtt_ms, BAD.uplink_mbps, 3.0)
    assert outcome.correct.mean() == 0.75
    # Among plans that tie, the fastest wins: it stops early where it is sure.
    assert outcome.total_ms.mean() < 9.0
    assert (outcome.sent_bytes == 0).all()


def test_the_lookup_rule_works_each_network_out_once():
    rule = lookup()
    for _ in range(3):
        rule.choose(View(0, 16, NETWORK, network_age=0))
        rule.choose(View(0, 16, BAD, network_age=0))

    assert len(rule.table) == 2


def test_the_lookup_rule_sends_when_it_knows_nothing_or_too_little():
    rule = lookup(probe_every=5)
    local = rule.plan_for(BAD, deadline_ms=16)

    assert rule.choose(View(0, 16, None, network_age=0)) == OFFLOAD
    assert rule.choose(View(9, 16, BAD, network_age=4)) == local
    # The last sighting is five frames old: send a frame to look again.
    assert rule.choose(View(10, 16, BAD, network_age=5)) == OFFLOAD
    # Without probing it trusts an old sighting forever.
    assert lookup().choose(View(10, 16, BAD, network_age=500)) == local
