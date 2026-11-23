import numpy as np
import pandas as pd

from edge_inference.live_check import answer_agreement, live_plan, simulate_live
from edge_inference.menu_table import menu_measurements, route_key
from edge_inference.simulate import Plan
from edge_inference.traces import Trace
from test_menu_table import tables

VARIANT, ROUTE = "m@8/float32", route_key("jpeg8q50", "edge", "big@16/int8")


def live_rows(plan, correct, local_correct, server_correct, local_ms, server_at_ms):
    return pd.DataFrame({
        "plan": plan, "image": [0, 1, 2, 3], "variant": VARIANT, "photo": "jpeg8q50",
        "server_model": "big@16/int8", "deadline_ms": 1000.0, "horizon_ms": 2000.0,
        "cancel": "never", "correct": correct, "local_correct": local_correct,
        "server_correct": server_correct, "local_ms": local_ms, "server_at_ms": server_at_ms,
        "total_ms": np.minimum(local_ms, server_at_ms),
    })  # fmt: skip


def test_plan_names_map_to_the_simulators_plans():
    assert live_plan("local", VARIANT, ROUTE) == Plan(VARIANT)
    assert live_plan("offload", VARIANT, ROUTE) == Plan(route=ROUTE)
    assert live_plan("hedge", VARIANT, ROUTE) == Plan(VARIANT, route=ROUTE, hedge=True)


def test_answers_given_live_are_checked_against_the_table():
    table = menu_measurements(**tables(), link="wifi", seed=1)
    # The table: local right on images 0 and 2; the server right on all four.
    live = live_rows("hedge", [True] * 4, [True, False, True, True], [True] * 4,
                     [1.0, 2.0, 3.0, np.inf], [5.0, 6.0, np.inf, 8.0])  # fmt: skip
    found = answer_agreement(table, live, VARIANT, ROUTE)
    assert found == {"local_answers": 3, "local_agree": 3, "server_answers": 3,
                     "server_agree": 3}  # fmt: skip
    live.loc[3, "local_ms"] = 4.0
    assert answer_agreement(table, live, VARIANT, ROUTE)["local_agree"] == 3


def test_the_simulator_plays_the_same_images_over_the_recording():
    table = menu_measurements(**tables(), link="wifi", seed=1)
    live = live_rows("local", [True, False, True, False], [True, False, True, False],
                     [False] * 4, [1.0] * 4, [np.inf] * 4)  # fmt: skip
    live["useful"], live["record"] = live["correct"], live["local_correct"]
    trace = Trace("wifi", np.full(10, 5.0), np.full(10, 8.0), 100.0)
    found = simulate_live(table, live, trace, "edge", draws=3).iloc[0]
    assert found["sim_useful"] == found["live_useful"] == 0.5
    assert found["local_agree"] == found["local_answers"] == 4
