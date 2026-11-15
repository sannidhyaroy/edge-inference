"""SplitEE's bandit rule, as published and with its assumptions repaired one by one.

SplitEE (Bajpai et al., 2023) treats "how deep should the device run before
deciding" as a multi-armed bandit. Each arm is an exit. Playing arm *i* means:
run to exit *i*; if its confidence reaches a threshold, answer there; otherwise
send the feature map and let the server finish. The last arm runs the whole
network on the device. The rule keeps the average reward of each arm and plays
the one with the highest upper confidence bound (UCB), which favours arms that
have paid well and arms that have been tried little.

The reward, from the paper's equation 1, never uses a label:

    confidence of whoever answered  -  mu * (cost of the layers run  +  o if it sent)

where the cost of reaching an exit is proportional to its depth in layers and
`o` is one constant offloading cost. I-SplitEE uses the same rule for images.

Five of its assumptions can be switched here, each on its own:

* **cost**: counted in layers with a constant `o`, or the measured time of the
  frame as a share of the deadline;
* **memory**: all of history, or a sliding window of recent frames, the
  standard fix for a bandit whose surroundings change;
* **threshold**: one for every exit, or one per exit;
* **reward signal**: confidence, or, as a diagnostic that no deployed rule
  could use, whether the answer was actually right;
* **arms**: float32 exits that fall back on the feature map, as published, or
  another precision, another fallback, and one more arm that sends the photo
  at once.

A sample that exits or is offloaded costs the same in the paper whichever
network it meets. That is the assumption the measured cost removes.
"""

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np

from edge_inference.decide import TIE_MARGIN
from edge_inference.policies import THRESHOLDS
from edge_inference.simulate import FEATURES, FINAL, PHOTO, Feedback, Plan, Scenario, View

# Convolution layers up to each exit of ResNet-18: the stem, then four per stage.
EXIT_LAYERS = (9, 13, 17)


def arm_plans(
    thresholds: Sequence[float],
    precision: str = "float32",
    otherwise: str = FEATURES,
    offload_arm: bool = False,
) -> list[Plan]:
    """One plan per arm: run to that exit, answer if confident, else do `otherwise`.

    `thresholds` has one entry per early exit. The final exit always answers.
    With `offload_arm` there is one more arm, first, that sends the photo at once.
    """
    plans = [Plan()] if offload_arm else []
    plans += [
        Plan(precision, depth, (*[math.inf] * depth, thresholds[depth]), otherwise)
        for depth in range(FINAL)
    ]
    return [*plans, Plan(precision, FINAL)]


def arm_label(plan: Plan) -> str:
    if plan.precision is None:
        return "offload"
    return "final exit" if plan.depth == FINAL else f"exit {plan.depth + 1}"


class UCB:
    """Upper confidence bound over a fixed set of arms, optionally over a sliding window.

    Each arm is played once before any comparison. With a window, only the
    last `window` plays count, so an arm that drops out of the window is
    played again: that is what lets the rule notice a change.
    """

    def __init__(self, arms: int, beta: float = 1.0, window: int | None = None) -> None:
        self.beta = beta
        self.window = window
        self.total = [0.0] * arms
        self.count = [0] * arms
        self.plays = 0
        self.recent: deque[tuple[int, float]] = deque()

    def select(self) -> int:
        for arm, count in enumerate(self.count):
            if count == 0:
                return arm
        horizon = self.plays + 1 if self.window is None else min(self.plays + 1, self.window)
        log = math.log(horizon)
        best, best_index = 0, -math.inf
        for arm, count in enumerate(self.count):
            index = self.total[arm] / count + self.beta * math.sqrt(log / count)
            if index > best_index:
                best, best_index = arm, index
        return best

    def update(self, arm: int, reward: float) -> None:
        self.plays += 1
        self.total[arm] += reward
        self.count[arm] += 1
        if self.window is None:
            return
        self.recent.append((arm, reward))
        if len(self.recent) > self.window:
            old_arm, old_reward = self.recent.popleft()
            self.total[old_arm] -= old_reward
            self.count[old_arm] -= 1


@dataclass(frozen=True)
class Settings:
    """Everything that defines one variant of the rule."""

    thresholds: tuple[float, ...]
    # "layers": mu * layer_cost * (layers run + offload_layers if it sent).
    # "measured": mu * (the frame's measured time / the deadline).
    cost: str = "layers"
    mu: float = 1.0
    layer_cost: float = 0.1
    offload_layers: float = 5.0
    beta: float = 1.0
    window: int | None = None
    precision: str = "float32"
    otherwise: str = FEATURES
    offload_arm: bool = False

    def plans(self) -> list[Plan]:
        return arm_plans(self.thresholds, self.precision, self.otherwise, self.offload_arm)

    def describe(self) -> str:
        stops = "/".join(f"{t:g}" for t in self.thresholds)
        if self.cost == "layers":
            cost = f"{self.layer_cost:g} per layer, o = {self.offload_layers:g} layers"
        else:
            cost = f"measured time, mu = {self.mu:g}"
        memory = "all history" if self.window is None else f"last {self.window} frames"
        return f"{self.precision}, else {self.otherwise}; threshold {stops}; {cost}; {memory}"


def layers_run(plan: Plan) -> int:
    """Layers the device runs under a plan: none when the photo is sent at once."""
    return 0 if plan.precision is None else EXIT_LAYERS[plan.depth]


def reward(
    settings: Settings,
    plan: Plan,
    confidence: float,
    sent: bool,
    total_ms: float,
    deadline_ms: float,
) -> float:
    """The paper's reward for one frame, with the cost model the settings name."""
    if settings.cost == "layers":
        layers = layers_run(plan) + (settings.offload_layers if sent else 0.0)
        return confidence - settings.mu * settings.layer_cost * layers
    return confidence - settings.mu * total_ms / deadline_ms


@dataclass
class Bandit:
    """The rule itself, behind the interface every rule shares."""

    name: str
    settings: Settings
    plans: list[Plan] = field(init=False)
    ucb: UCB = field(init=False)
    deadline_ms: float = field(init=False, default=math.inf)
    arm: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.plans = self.settings.plans()
        self.ucb = UCB(len(self.plans), self.settings.beta, self.settings.window)

    def choose(self, view: View) -> Plan:
        self.deadline_ms = view.deadline_ms
        self.arm = self.ucb.select()
        return self.plans[self.arm]

    def observe(self, feedback: Feedback) -> None:
        plan = self.plans[self.arm]
        confidence = (
            feedback.server_confidence if feedback.sent else feedback.confidences[plan.depth]
        )
        self.ucb.update(
            self.arm,
            reward(
                self.settings, plan, confidence, feedback.sent, feedback.total_ms, self.deadline_ms
            ),
        )


@dataclass(frozen=True)
class Arms:
    """What each arm would do to every frame of a scenario; arrays are (frames, arms)."""

    total_ms: np.ndarray
    correct: np.ndarray
    device_busy_ms: np.ndarray
    sent_bytes: np.ndarray
    confidence: np.ndarray
    # Layers the device runs under each arm, one entry per arm.
    layers: np.ndarray

    @classmethod
    def of(cls, scenario: Scenario, plans: Sequence[Plan]) -> Arms:
        outcomes = [scenario.resolve(plan) for plan in plans]
        frames = np.arange(len(scenario.frames))
        confidence = [
            np.where(o.sent_bytes > 0, o.server_confidence, o.confidences[frames, plan.depth])
            for plan, o in zip(plans, outcomes, strict=True)
        ]

        def stack(name: str) -> np.ndarray:
            return np.stack([getattr(o, name) for o in outcomes], axis=1)

        return cls(
            total_ms=stack("total_ms"),
            correct=stack("correct"),
            device_busy_ms=stack("device_busy_ms"),
            sent_bytes=stack("sent_bytes"),
            confidence=np.stack(confidence, axis=1),
            layers=np.array([layers_run(plan) for plan in plans]),
        )

    def rewards(
        self, settings: Settings, deadline_ms: float, *, by_label: bool = False
    ) -> np.ndarray:
        """Reward of every arm on every frame.

        `by_label` swaps confidence for whether the answer was right. No
        deployed rule could do that; it measures what trusting confidence costs.
        """
        signal = self.correct.astype(float) if by_label else self.confidence
        if settings.cost == "layers":
            layers = self.layers + settings.offload_layers * (self.sent_bytes > 0)
            return signal - settings.mu * settings.layer_cost * layers
        return signal - settings.mu * self.total_ms / deadline_ms


def play(rewards: np.ndarray, beta: float = 1.0, window: int | None = None) -> np.ndarray:
    """The arm UCB plays on each frame, given what every arm would have paid.

    This is the same rule as `Bandit`, run without the simulator in between,
    which is what makes trying hundreds of settings affordable.
    """
    ucb = UCB(rewards.shape[1], beta, window)
    chosen = []
    for row in rewards.tolist():
        arm = ucb.select()
        ucb.update(arm, row[arm])
        chosen.append(arm)
    return np.array(chosen)


def run(arms: Arms, chosen: np.ndarray) -> dict[str, np.ndarray]:
    """The frames' outcomes when `chosen` names the arm played on each."""
    frames = np.arange(len(chosen))
    return {
        name: getattr(arms, name)[frames, chosen]
        for name in ("total_ms", "correct", "device_busy_ms", "sent_bytes")
    }


def evaluate(
    scenario: Scenario, settings: Settings, *, by_label: bool = False, arms: Arms | None = None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run one variant over a scenario; return the frames' outcomes and the arms played."""
    arms = arms or Arms.of(scenario, settings.plans())
    rewards = arms.rewards(settings, scenario.deadline_ms, by_label=by_label)
    chosen = play(rewards, settings.beta, settings.window)
    return run(arms, chosen), chosen


def best_settings(
    scenario: Scenario, candidates: Sequence[Settings], tie_margin: float = TIE_MARGIN
) -> Settings:
    """The candidate that answers the most frames correctly in time on `scenario`.

    The rule is run in full for every candidate, on tuning images and a tuning
    network sequence. Near-ties go to the lower mean time. Scoring by useful
    frames is generous to the rule: its own reward knows no deadline.
    """
    cache: dict[tuple[Plan, ...], Arms] = {}
    useful, mean_ms = [], []
    for settings in candidates:
        plans = tuple(settings.plans())
        if plans not in cache:
            cache[plans] = Arms.of(scenario, plans)
        outcome, _ = evaluate(scenario, settings, arms=cache[plans])
        hit = outcome["correct"] & (outcome["total_ms"] <= scenario.deadline_ms)
        useful.append(hit.mean())
        mean_ms.append(outcome["total_ms"].mean())
    useful, mean_ms = np.array(useful), np.array(mean_ms)
    tied = np.flatnonzero(useful >= useful.max() - tie_margin)
    return candidates[tied[mean_ms[tied].argmin()]]


# The paper sets the offloading cost to between one and five layers' worth, and
# its code charges 0.1 per layer. Smaller per-layer costs are offered too, since
# 0.1 over 17 layers outweighs any confidence.
LAYER_COSTS = (0.01, 0.02, 0.05, 0.1)
OFFLOAD_LAYERS = (1.0, 2.0, 3.0, 4.0, 5.0)
MUS = (0.25, 0.5, 1.0, 2.0, 4.0)
WINDOWS = (25, 50, 100, 200, 400)
COARSE_THRESHOLDS = tuple(round(0.1 * step, 1) for step in range(1, 10))


def published_candidates(thresholds: Sequence[float] = THRESHOLDS) -> list[Settings]:
    return [
        Settings((alpha,) * FINAL, layer_cost=layer_cost, offload_layers=offload)
        for alpha in thresholds
        for layer_cost in LAYER_COSTS
        for offload in OFFLOAD_LAYERS
    ]


def measured_candidates(
    thresholds: Sequence[float] = THRESHOLDS,
    windows: Sequence[int | None] = (None,),
    **arms: object,
) -> list[Settings]:
    return [
        Settings((alpha,) * FINAL, cost="measured", mu=mu, window=window, **arms)
        for alpha in thresholds
        for mu in MUS
        for window in windows
    ]


def per_exit_candidates(
    base: Settings, thresholds: Sequence[float] = COARSE_THRESHOLDS
) -> list[Settings]:
    return [replace(base, thresholds=pair) for pair in product(thresholds, repeat=FINAL)]


@dataclass(frozen=True)
class Variant:
    """One row for the scoreboard: a tuned variant and what it did on the test frames."""

    name: str
    kind: str
    settings: Settings
    outcome: dict[str, np.ndarray]
    chosen: np.ndarray

    def describe(self) -> str:
        plans = self.settings.plans()
        share = np.bincount(self.chosen, minlength=len(plans)) / len(self.chosen)
        played = ", ".join(
            f"{arm_label(plan)} {part * 100:.0f}%" for plan, part in zip(plans, share, strict=True)
        )
        return f"{self.settings.describe()}; played {played}"

    def played(self) -> list[str]:
        """The arm played on each frame, by name."""
        labels = [arm_label(plan) for plan in self.settings.plans()]
        return [labels[arm] for arm in self.chosen]


def variants(tuning: Scenario, test: Scenario, *, every: bool = True) -> list[Variant]:
    """SplitEE as published, then with one assumption changed at a time.

    Every variant's settings are chosen on `tuning` and it is then run once,
    from scratch, on `test`. Without `every`, only the published rule and the
    one with every repair are run, which is what a sweep over scenarios needs.
    """
    # The published arms are float32 and fall back on the feature map. The
    # wide arms are INT8 exits, the photo as the fallback, and an arm that
    # sends the photo at once: the options the decision map found best.
    wide_arms = {"precision": "int8", "otherwise": PHOTO, "offload_arm": True}
    published = best_settings(tuning, published_candidates())
    repaired = best_settings(tuning, measured_candidates(windows=WINDOWS, **wide_arms))
    rows = [("SplitEE as published", "rule", published, False)]
    if every:
        per_exit = best_settings(tuning, per_exit_candidates(published))
        measured = best_settings(tuning, measured_candidates())
        windowed = best_settings(tuning, measured_candidates(windows=WINDOWS))
        wider = best_settings(tuning, measured_candidates(**wide_arms))
        rows += [
            ("SplitEE, a threshold per exit", "rule", per_exit, False),
            ("SplitEE, measured cost", "rule", measured, False),
            ("SplitEE, measured cost, sliding window", "rule", windowed, False),
            ("SplitEE, measured cost, INT8 and photo arms", "rule", wider, False),
        ]
    rows.append(("SplitEE, every repair at once", "rule", repaired, False))
    if every:
        # Not a rule a device could run: it is rewarded with the right answer.
        rows.append(("SplitEE as published, told the answer", "diagnostic", published, True))
    return [
        Variant(name, kind, settings, *evaluate(test, settings, by_label=by_label))
        for name, kind, settings, by_label in rows
    ]


def offload_cost_in_layers(scenario: Scenario) -> dict[str, float]:
    """What sending features after the first exit really costs, in layers of device work.

    The paper takes the offloading cost to be one to five layers' worth,
    whatever the network. Here it is measured: the median time to pack, send
    and get an answer, over the median time the device needs per layer.
    """
    send_all = Plan("float32", 0, (math.inf,), FEATURES)
    final = scenario.resolve(Plan("float32", FINAL))
    per_layer = np.median(final.total_ms) / EXIT_LAYERS[FINAL]
    sent = scenario.resolve(send_all)
    reach = scenario.resolve(Plan("float32", 0))
    extra = sent.total_ms - reach.total_ms
    costs = {}
    for state, name in enumerate(scenario.state_names):
        met = scenario.states == state
        # A short run may never meet one of the networks.
        costs[name] = float(np.median(extra[met]) / per_layer) if met.any() else math.nan
    return costs
