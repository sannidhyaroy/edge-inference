"""Four yardsticks to hold a rule against, each knowing a little more.

A score means nothing alone: 74% of frames useful is good or bad only next to
what was achievable. These are computed after a run, from the same frames and
the same networks the rules met.

1. **Best fixed option.** One plain option for the whole run. This is what a
   bandit's regret is usually measured against.
2. **Best option per network.** A plain option for each network state. It
   knows the network exactly, and nothing about the image.
3. **Best confidence rule per network.** The best plan in the whole family for
   each network state, found by trying every one on these very frames. It
   knows the network and each exit's confidence. This is the fair target: the
   best that playing one plan of the family per network state can do.
4. **Best possible per frame.** The best plain option for each frame, chosen
   knowing which options answer it correctly. Unreachable, because it needs
   the answer in advance.

Each is at least as good as the one before, by construction. The gaps are the
finding: 1 to 2 is what watching the network is worth, 2 to 3 what confidence
is worth, and 3 to 4 the price of not knowing the answer.

When hedged plans are in play, one rung goes between 3 and 4:

3h. **Best plan per network, hedges included.** The same search as 3 over a
    family that also sends the photo while computing. 3 to 3h is what doing
    both at once is worth to a rule that knows the network.

And one more for applications that keep a record rather than act once:

4r. **Best possible record per frame.** For each frame, whether any plain
    option's final answer, the server's if it arrives by the horizon, is right.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from edge_inference.policies import THRESHOLDS, fixed_plans, plan_family
from edge_inference.simulate import HORIZON_MS, Outcome, Plan, Scenario, score

FIELDS = (
    "total_ms",
    "correct",
    "device_busy_ms",
    "sent_bytes",
    "local_correct",
    "server_at_ms",
    "server_correct",
)


def _best_per_group(
    scenario: Scenario, plans: Sequence[Plan], groups: np.ndarray
) -> tuple[list[Plan], dict[str, np.ndarray]]:
    """The best plan for each group of frames, and the run that results.

    Best means the most useful frames; plans that tie exactly are separated
    by total time.
    """
    count = int(groups.max()) + 1
    useful = np.zeros((len(plans), count))
    time = np.zeros((len(plans), count))
    for index, plan in enumerate(plans):
        outcome = scenario.resolve(plan)
        hit = outcome.correct & (outcome.total_ms <= scenario.deadline_ms)
        useful[index] = np.bincount(groups, weights=hit, minlength=count)
        time[index] = np.bincount(groups, weights=outcome.total_ms, minlength=count)

    chosen = []
    run = {name: np.zeros(len(groups)) for name in FIELDS}
    for group in range(count):
        tied = np.flatnonzero(useful[:, group] == useful[:, group].max())
        plan = plans[tied[time[tied, group].argmin()]]
        chosen.append(plan)
        at = np.flatnonzero(groups == group)
        outcome = scenario.resolve(plan, at)
        for name in FIELDS:
            run[name][at] = getattr(outcome, name)
    return chosen, run


def _best_per_frame(scenario: Scenario) -> dict[str, np.ndarray]:
    """For each frame, the fastest plain option that is useful, else the fastest."""
    plans = fixed_plans(scenario.measurements)
    outcomes: list[Outcome] = [scenario.resolve(plan) for plan in plans]
    total = np.stack([o.total_ms for o in outcomes])
    correct = np.stack([o.correct for o in outcomes])
    useful = correct & (total <= scenario.deadline_ms)
    pick = np.where(
        useful.any(axis=0), np.where(useful, total, np.inf).argmin(axis=0), total.argmin(axis=0)
    )
    frames = np.arange(total.shape[1])
    return {name: np.stack([getattr(o, name) for o in outcomes])[pick, frames] for name in FIELDS}


def _best_record_per_frame(scenario: Scenario) -> dict[str, np.ndarray]:
    """For each frame, a plain option whose final answer is right, if any is."""
    plans = fixed_plans(scenario.measurements)
    outcomes = [scenario.resolve(plan) for plan in plans]
    final = np.stack(
        [
            np.where(o.server_at_ms <= scenario.horizon_ms, o.server_correct, o.local_correct)
            for o in outcomes
        ]
    )
    total = np.stack([o.total_ms for o in outcomes])
    pick = np.where(final.any(axis=0), np.where(final, total, np.inf).argmin(axis=0), 0)
    frames = np.arange(total.shape[1])
    return {name: np.stack([getattr(o, name) for o in outcomes])[pick, frames] for name in FIELDS}


def ladder_runs(
    scenario: Scenario,
    thresholds: Sequence[float] = THRESHOLDS,
    *,
    hedge: bool = False,
    cascade: bool = False,
) -> list[tuple[str, str, dict[str, np.ndarray]]]:
    """The yardsticks, weakest first: a name, what was chosen, and the run per frame.

    With `hedge`, rung 3h and the record rung 4r join the four. With `cascade`,
    rungs 3 and 3h also search the plans that send the photo when the final
    exit is unsure, so they bound rules that play them.
    """
    whole_run = np.zeros(len(scenario.frames), dtype=int)
    table = scenario.measurements

    def named(plans: Sequence[Plan]) -> str:
        return "; ".join(
            f"{scenario.state_names[state]}: {plan.describe()}" for state, plan in enumerate(plans)
        )

    (plan,), fixed = _best_per_group(scenario, fixed_plans(table), whole_run)
    per_state, by_network = _best_per_group(scenario, fixed_plans(table), scenario.states)
    family, by_confidence = _best_per_group(
        scenario, plan_family(table, thresholds, cascade=cascade), scenario.states
    )
    runs = [
        ("1. best fixed option", plan.describe(), fixed),
        ("2. best option per network", named(per_state), by_network),
        ("3. best confidence rule per network", named(family), by_confidence),
    ]
    if hedge:
        hedged, by_hedge = _best_per_group(
            scenario, plan_family(table, thresholds, hedge=True, cascade=cascade), scenario.states
        )
        runs.append(("3h. best plan per network, hedges included", named(hedged), by_hedge))
    runs.append(("4. best possible per frame", "chosen per frame", _best_per_frame(scenario)))
    if hedge:
        runs.append(
            (
                "4r. best possible record per frame",
                "chosen per frame",
                _best_record_per_frame(scenario),
            )
        )
    return runs


def score_run(
    run: dict[str, np.ndarray], deadline_ms: float, horizon_ms: float = HORIZON_MS
) -> dict[str, float]:
    """`score` for a run held as one array per field."""
    events = {}
    if "server_at_ms" in run:
        events = {
            "local_correct": run["local_correct"].astype(bool),
            "server_at_ms": run["server_at_ms"],
            "server_correct": run["server_correct"].astype(bool),
        }
    return score(
        run["total_ms"],
        run["correct"].astype(bool),
        run["device_busy_ms"],
        run["sent_bytes"],
        deadline_ms,
        horizon_ms=horizon_ms,
        **events,
    )


def ladder(
    scenario: Scenario, thresholds: Sequence[float] = THRESHOLDS, *, hedge: bool = False
) -> pd.DataFrame:
    """Score the yardsticks on a scenario, weakest first."""
    return pd.DataFrame(
        [
            {
                "rule": name,
                "kind": "yardstick",
                "plans": plans,
                **score_run(run, scenario.deadline_ms, scenario.horizon_ms),
            }
            for name, plans, run in ladder_runs(scenario, thresholds, hedge=hedge)
        ]
    )
