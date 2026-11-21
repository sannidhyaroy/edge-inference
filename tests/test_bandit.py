"""Checks for the SplitEE rule: the arms, UCB, the reward, and tuning.

Expected values come from the table in `builders.py`.
"""

import math

import numpy as np
import pytest

from builders import NETWORK, measurements
from edge_inference.bandit import (
    UCB,
    Arms,
    Bandit,
    Settings,
    arm_label,
    arm_plans,
    best_settings,
    offload_cost_in_layers,
    play,
    reward,
    run,
    variants,
)
from edge_inference.offload import Network
from edge_inference.simulate import FEATURES, PHOTO, Plan, Scenario, simulate

BAD = Network(rtt_ms=100, uplink_mbps=8)


def scenario(states, *, slowdown=1.0, deadline_ms=20.0):
    return Scenario(
        measurements(),
        frames=np.arange(len(states)) % 4,
        states=np.array(states),
        networks=[NETWORK, BAD],
        state_names=["good", "bad"],
        slowdown=slowdown,
        deadline_ms=deadline_ms,
    )


def test_each_arm_runs_to_one_exit_and_sends_if_unsure():
    first, second, last = arm_plans((0.5, 0.7))

    assert first == Plan("float32", 0, (0.5,), FEATURES)
    assert second == Plan("float32", 1, (math.inf, 0.7), FEATURES)
    assert last == Plan("float32")


def test_the_arms_can_be_widened():
    plans = arm_plans((0.5, 0.7), precision="int8", otherwise=PHOTO, offload_arm=True)

    assert plans[0] == Plan()
    assert plans[1] == Plan("int8", 0, (0.5,), PHOTO)
    assert plans[-1] == Plan("int8")
    assert [arm_label(plan) for plan in plans] == ["offload", "exit 1", "exit 2", "final exit"]


def test_every_arm_is_played_once_before_any_is_compared():
    ucb = UCB(arms=3)
    order = []
    for _ in range(3):
        order.append(ucb.select())
        ucb.update(order[-1], reward=0.0)

    assert order == [0, 1, 2]


def test_ucb_settles_on_the_arm_that_pays_best():
    rewards = np.tile([0.1, 0.9, 0.5], (300, 1))
    chosen = play(rewards)

    assert np.mean(chosen == 1) > 0.7
    # The others are still tried now and then: that is the exploring.
    assert (chosen[100:] != 1).any()


def test_a_sliding_window_follows_a_change():
    # Arm 1 always pays a half. Arm 0 pays in full for 1000 frames, then nothing.
    rewards = np.full((1300, 2), 0.5)
    rewards[:1000, 0] = 1.0
    rewards[1000:, 0] = 0.0
    remembering = np.mean(play(rewards)[1000:] == 1)
    forgetting = np.mean(play(rewards, window=50)[1000:] == 1)

    # A thousand good frames take a thousand bad ones to average away; a window
    # of 50 has forgotten them after 50.
    assert remembering < 0.2
    assert forgetting > 0.7


def test_the_published_reward_counts_layers_and_a_constant():
    settings = Settings((0.5, 0.5), layer_cost=0.1, offload_layers=5)

    # Nine layers to the first exit; five more if the frame is sent.
    first = settings.plans()[0]
    assert reward(settings, first, 0.9, False, total_ms=3, deadline_ms=40) == pytest.approx(0.0)
    assert reward(settings, first, 0.8, True, total_ms=99, deadline_ms=40) == pytest.approx(-0.6)


def test_the_measured_reward_counts_time_against_the_deadline():
    settings = Settings((0.5, 0.5), cost="measured", mu=2.0)

    first = settings.plans()[0]
    assert reward(settings, first, 0.9, True, total_ms=10, deadline_ms=40) == pytest.approx(0.4)


def test_arms_record_what_each_would_do_to_every_frame():
    arms = Arms.of(scenario([0, 0, 0, 0]), arm_plans((0.5, 0.5)))

    # Arm 1: image 0 answers after 2 ms; the rest pack (0.5), send 500 bytes
    # (10 + 0.5) and wait for the server (1.5).
    assert arms.total_ms[:, 0].tolist() == [2.0, 14.5, 14.5, 14.5]
    assert arms.total_ms[:, 1].tolist() == [4.0, 4.0, 16.5, 16.5]
    assert arms.total_ms[:, 2].tolist() == [6.0] * 4
    # A sent frame reports the server's confidence, an answered one the exit's.
    assert arms.confidence[:, 0].tolist() == [0.9, 0.9, 0.9, 0.4]
    assert arms.confidence[:, 1].tolist() == [0.9, 0.8, 0.9, 0.4]


def test_the_published_reward_is_blind_to_the_network():
    good = Arms.of(scenario([0, 0, 0, 0]), arm_plans((0.5, 0.5)))
    bad = Arms.of(scenario([1, 1, 1, 1]), arm_plans((0.5, 0.5)))
    published = Settings((0.5, 0.5))
    measured = Settings((0.5, 0.5), cost="measured")

    assert (good.rewards(published, 20) == bad.rewards(published, 20)).all()
    assert (good.rewards(measured, 20) > bad.rewards(measured, 20))[1:, 0].all()


def test_rewarding_by_label_replaces_confidence_with_correctness():
    arms = Arms.of(scenario([0, 0, 0, 0]), arm_plans((0.5, 0.5)))
    free = Settings((0.5, 0.5), layer_cost=0.0)

    # The final exit is right on images 0 to 2 and wrong on image 3.
    assert arms.rewards(free, 20, by_label=True)[:, 2].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_the_rule_and_the_fast_path_play_the_same_arms():
    states = [0] * 30 + [1] * 30
    for settings in (
        Settings((0.5, 0.5), layer_cost=0.02),
        Settings((0.3, 0.85), cost="measured", mu=0.5, window=10),
        Settings((0.5, 0.5), cost="measured", precision="int8", otherwise=PHOTO, offload_arm=True),
    ):
        run_scenario = scenario(states)
        arms = Arms.of(run_scenario, settings.plans())
        chosen = play(arms.rewards(settings, 20.0), settings.beta, settings.window)
        trace = simulate(Bandit("rule", settings), run_scenario)

        assert trace["total_ms"].tolist() == run(arms, chosen)["total_ms"].tolist()


def test_settings_are_chosen_by_useful_frames():
    # On the bad network a sent frame is late, so only local answers count.
    # The final exit is right on three images, the first exit on one.
    tuning = scenario([1] * 200)
    cheap_to_send = Settings((0.5, 0.5), layer_cost=0.01, offload_layers=1)
    dear_to_send = Settings((0.5, 0.5), layer_cost=0.01, offload_layers=20)

    # With sending priced at 20 layers the first arm stops looking attractive.
    assert best_settings(tuning, [cheap_to_send, dear_to_send]) == dear_to_send


def test_the_real_offloading_cost_is_many_layers():
    layers = offload_cost_in_layers(scenario([0, 0, 1, 1]))

    # A layer takes 6 / 17 ms. Sending after exit 1 adds 12.5 ms on the good
    # network and 102.5 ms on the bad one.
    assert layers["good"] == pytest.approx(12.5 / (6 / 17))
    assert layers["bad"] == pytest.approx(102.5 / (6 / 17))


def test_variants_change_one_assumption_at_a_time():
    states = ([0] * 10 + [1] * 10) * 2
    rows = variants(scenario(states), scenario(states))

    assert [row.name for row in rows] == [
        "SplitEE as published",
        "SplitEE, a threshold per exit",
        "SplitEE, measured cost",
        "SplitEE, measured cost, sliding window",
        "SplitEE, measured cost, INT8 and photo arms",
        "SplitEE, every repair at once",
        "SplitEE as published, told the answer",
    ]
    assert [row.kind for row in rows] == ["rule"] * 6 + ["diagnostic"]
    assert rows[0].settings.cost == "layers" and rows[0].settings.window is None
    assert rows[2].settings.cost == "measured" and rows[2].settings.window is None
    assert rows[3].settings.window is not None
    assert rows[4].settings.precision == "int8" and rows[4].settings.offload_arm
    assert rows[5].settings.offload_arm and rows[5].settings.window is not None
    assert rows[6].settings == rows[0].settings
    assert set(rows[4].played()) <= {"offload", "exit 1", "exit 2", "final exit"}
    assert all(len(row.chosen) == len(states) for row in rows)
