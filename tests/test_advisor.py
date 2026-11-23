import pandas as pd

from edge_inference.advisor import advise
from edge_inference.recorded import BEST_FIXED
from edge_inference.scoreboard import HEDGE, OURS, OURS_HEDGED


def board(**gains):
    """A scoreboard where each rule's useful share and interval low end are given."""
    rows = []
    for rule, useful, low, record in [
        (BEST_FIXED, 0.60, 0.0, 0.60),
        (HEDGE, *gains.get("hedge", (0.60, -0.01)), 0.70),
        (OURS, *gains.get("ours", (0.60, -0.01)), 0.61),
        (OURS_HEDGED, *gains.get("ours_hedged", (0.60, -0.01)), 0.66),
        ("always local m", 0.60, 0.0, 0.60),
    ]:
        rows.append({"rule": rule, "plans": f"plan of {rule}", "useful": useful,
                     "useful_vs_fixed_lo": low, "useful_vs_fixed_hi": low + 0.02,
                     "behind_fair": 0.65 - useful, "record": record, "device_busy_ms": 5.0,
                     "sent_share": 0.0, "config": "c", "link": "l", "deadline_ms": 100.0})  # fmt: skip
    return pd.DataFrame(rows)


def test_without_a_clear_gain_the_advice_is_the_fixed_option():
    advice = advise(board(ours=(0.61, -0.001)))
    assert (advice["verdict"], advice["rule"]) == ("fixed", BEST_FIXED)
    assert advice["fair_target"] == 0.65


def test_the_best_clear_gain_wins_and_a_tie_goes_to_the_simpler_rule():
    advice = advise(board(hedge=(0.63, 0.01), ours=(0.64, 0.02), ours_hedged=(0.6405, 0.02)))
    assert (advice["verdict"], advice["rule"]) == ("adapt", OURS)


def test_a_record_is_advised_by_its_own_score():
    advice = advise(board())
    assert (advice["record_rule"], advice["record"]) == (HEDGE, 0.70)
