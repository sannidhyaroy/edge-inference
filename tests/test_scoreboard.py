"""Checks that a scoreboard holds every rule and yardstick, consistently."""

import numpy as np

from builders import measurements
from edge_inference.scoreboard import FAIR_TARGET, OURS, Setup, grid, run_setup, run_sweep


def board_for(**settings):
    setup = Setup(slowdown=1.0, deadline_ms=20.0, passes=10, seed=0, **settings)
    return run_setup(measurements(), setup, every_variant=False)


def test_runs_are_told_apart_by_their_tag():
    assert Setup().tag() == "t4_8x_30ms_wifi-4g_stay0.95-0.95"
    assert Setup(good="wifi", bad="wifi").tag() == "t4_8x_30ms_wifi-wifi"
    assert Setup(knowledge="probe:5").tag().endswith("_probe5")
    assert Setup(slowdown=2.5, deadline_ms=12.5).tag().startswith("t4_2.5x_12.5ms_")


def test_a_steady_network_has_one_state():
    assert Setup(good="5g", bad="5g").names == ["5g"]
    assert Setup().names == ["wifi", "4g"]


def test_the_scoreboard_lists_rules_then_yardsticks():
    result = board_for()
    board = result.board

    # Half of the four images are held out, one from each class.
    assert result.test_images == 2 and result.held_out
    assert len(result.scenario.frames) == 20
    assert board["kind"].tolist() == ["rule"] * 8 + ["yardstick"] * 4
    assert board["rule"].tolist()[5:8] == [
        "SplitEE as published",
        "SplitEE, every repair at once",
        OURS,
    ]
    assert len(result.traces) == len(board) * 20
    assert set(result.traces["rule"]) == set(board["rule"])


def test_distances_are_measured_from_the_fair_target():
    board = board_for().board.set_index("rule")
    yardsticks = board[board["kind"] == "yardstick"]

    assert board.loc[FAIR_TARGET, "behind_fair"] == 0.0
    assert (np.diff(yardsticks["useful"]) >= 0).all()
    assert board["useful"].max() == yardsticks["useful"].iloc[-1]
    behind = board.loc[FAIR_TARGET, "useful"] - board["useful"]
    assert np.allclose(board["behind_fair"], behind)


def test_only_our_rule_reports_a_decision_time():
    board = board_for().board.set_index("rule")

    assert board.loc[OURS, "decide_us"] > 0
    assert board["decide_us"].drop(OURS).isna().all()


def test_the_setup_is_recorded_on_every_row():
    board = board_for(stay_good=0.8).board

    assert (board["stay_good"] == 0.8).all()
    assert (board["seed"] == 0).all()
    assert np.isclose(board["network_share_wifi"] + board["network_share_4g"], 1.0).all()
    assert {"offload_layers_wifi", "offload_layers_4g"} <= set(board.columns)


def test_learning_the_network_only_by_sending_still_scores():
    board = board_for(knowledge="sends").board.set_index("rule")

    assert 0.0 <= board.loc[OURS, "useful"] <= board["useful"].max()
    assert (board["knowledge"] == "sends").all()


def test_scoring_every_image_is_flagged():
    result = board_for(tune_share=0.0)

    assert not result.held_out
    assert result.test_images == 4


def test_a_grid_crosses_speeds_deadlines_and_seeds():
    setups = grid(Setup(seed=7), slowdowns=(2, 8), deadlines=(30, 50, 100), seeds=2)

    assert len(setups) == 2 * 3 * 2
    assert {(s.slowdown, s.deadline_ms) for s in setups} == {
        (2, 30),
        (2, 50),
        (2, 100),
        (8, 30),
        (8, 50),
        (8, 100),
    }
    assert sorted({s.seed for s in setups}) == [7, 1007]


def test_a_sweep_stacks_one_scoreboard_per_setup():
    base = Setup(slowdown=1.0, deadline_ms=20.0, passes=10, seed=0)
    setups = grid(base, slowdowns=(1, 3), deadlines=(20,), seeds=1)
    boards = run_sweep(measurements(), setups)

    # 5 simple rules, 2 SplitEE variants, ours and 4 yardsticks in each.
    assert len(boards) == 2 * 12
    assert sorted(boards["slowdown"].unique()) == [1, 3]
    assert (boards.groupby("slowdown")["rule"].nunique() == 12).all()
