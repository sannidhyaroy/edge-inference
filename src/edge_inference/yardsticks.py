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
   knows the network and each exit's confidence. This is the fair target: no
   rule that decides from the network and confidence can do better.
4. **Best possible per frame.** The best plain option for each frame, chosen
   knowing which options answer it correctly. Unreachable, because it needs
   the answer in advance.

Each is at least as good as the one before, by construction. The gaps are the
finding: 1 to 2 is what watching the network is worth, 2 to 3 what confidence
is worth, and 3 to 4 the price of not knowing the answer.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from edge_inference.policies import THRESHOLDS, fixed_plans, plan_family
from edge_inference.simulate import Outcome, Plan, Scenario, score

FIELDS = ("total_ms", "correct", "device_busy_ms", "sent_bytes")


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
    outcomes: list[Outcome] = [scenario.resolve(plan) for plan in fixed_plans()]
    total = np.stack([o.total_ms for o in outcomes])
    correct = np.stack([o.correct for o in outcomes])
    useful = correct & (total <= scenario.deadline_ms)
    pick = np.where(
        useful.any(axis=0), np.where(useful, total, np.inf).argmin(axis=0), total.argmin(axis=0)
    )
    frames = np.arange(total.shape[1])
    return {name: np.stack([getattr(o, name) for o in outcomes])[pick, frames] for name in FIELDS}


def ladder(scenario: Scenario, thresholds: Sequence[float] = THRESHOLDS) -> pd.DataFrame:
    """Score the four yardsticks on a scenario, weakest first."""
    whole_run = np.zeros(len(scenario.frames), dtype=int)

    def named(plans: Sequence[Plan]) -> str:
        return "; ".join(
            f"{scenario.state_names[state]}: {plan.describe()}" for state, plan in enumerate(plans)
        )

    def row(name: str, plans: str, run: dict[str, np.ndarray]) -> dict[str, object]:
        scores = score(
            run["total_ms"],
            run["correct"].astype(bool),
            run["device_busy_ms"],
            run["sent_bytes"],
            scenario.deadline_ms,
        )
        return {"rule": name, "kind": "yardstick", "plans": plans, **scores}

    rows = []
    (plan,), run = _best_per_group(scenario, fixed_plans(), whole_run)
    rows.append(row("1. best fixed option", plan.describe(), run))
    plans, run = _best_per_group(scenario, fixed_plans(), scenario.states)
    rows.append(row("2. best option per network", named(plans), run))
    plans, run = _best_per_group(scenario, plan_family(thresholds), scenario.states)
    rows.append(row("3. best confidence rule per network", named(plans), run))
    rows.append(row("4. best possible per frame", "chosen per frame", _best_per_frame(scenario)))
    return pd.DataFrame(rows)
