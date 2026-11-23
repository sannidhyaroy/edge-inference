"""What to deploy, read off a scoreboard: the advice the measurements support.

A scoreboard scores every rule on one recorded link, one device and one
deadline. The advisor answers the question an engineer has for that setting:
is it worth adapting at all, and if so, how?

For an application that acts once on each answer, the candidates are, from
simplest to most demanding:

1. play one fixed option (the best one, even picked with hindsight);
2. hedge: send every photo and run the device's model at once, without
   watching the network;
3. watch the network and look a plan up per network state;
4. the same lookup, with hedged plans among those it may pick.

A candidate beats the fixed option only if its paired 95% interval over the
fixed option lies wholly above zero. If none does, the advice is the fixed
option. Otherwise it is the best-scoring candidate that does, the simpler one
when two score the same. The advice says what it costs: how long the device
is busy per frame and the share of frames it sends.

For an application that keeps a record, every rule is ranked by its record
share instead, and the advice is the best of them, with the fixed option's
record for comparison.
"""

import pandas as pd

from edge_inference.recorded import BEST_FIXED
from edge_inference.scoreboard import HEDGE, OURS, OURS_HEDGED

CANDIDATES = (
    (BEST_FIXED, "one fixed option"),
    (HEDGE, "hedge without watching the network"),
    (OURS, "watch the network, look the plan up"),
    (OURS_HEDGED, "watch the network, look the plan up, hedges allowed"),
)
SAME = 1e-3


def advise(board: pd.DataFrame) -> dict[str, object]:
    """The advice for one scoreboard, as one row."""
    rows = board.set_index("rule")
    fixed = rows.loc[BEST_FIXED]
    beat = [
        (name, what) for name, what in CANDIDATES[1:] if rows.loc[name, "useful_vs_fixed_lo"] > 0
    ]
    if beat:
        best = max(rows.loc[name, "useful"] for name, _ in beat)
        name, what = next((n, w) for n, w in beat if rows.loc[n, "useful"] >= best - SAME)
        verdict = "adapt"
    else:
        (name, what), verdict = CANDIDATES[0], "fixed"
    chosen = rows.loc[name]
    plain = board[board["rule"].str.startswith(("always ", "hedge", "lookup"))]
    recorder = plain.loc[plain["record"].idxmax()]
    first = board.iloc[0]
    return {
        "config": first["config"],
        "link": first["link"],
        "deadline_ms": first["deadline_ms"],
        "verdict": verdict,
        "advice": what,
        "rule": name,
        "plans": chosen["plans"],
        "useful": chosen["useful"],
        "fixed_useful": fixed["useful"],
        "gain_lo": chosen["useful_vs_fixed_lo"],
        "gain_hi": chosen["useful_vs_fixed_hi"],
        "fair_target": chosen["useful"] + chosen["behind_fair"],
        "device_busy_ms": chosen["device_busy_ms"],
        "fixed_busy_ms": fixed["device_busy_ms"],
        "sent_share": chosen["sent_share"],
        "record_rule": recorder["rule"],
        "record": recorder["record"],
        "fixed_record": fixed["record"],
    }
