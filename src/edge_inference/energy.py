"""Energy from the processor's own counters (RAPL), read around batches of work.

Linux exposes a running energy total in microjoules for the whole package and
for the cores. The counters are readable only by root unless permission is
given, they wrap around at a published maximum, and on AMD processors they are
estimates from a model of the chip rather than a meter. They update about every
millisecond, so one inference is too short to measure: work is run in batches,
and idle power, measured just before and after, is taken away.

Marginal energy per inference = (energy over the batch - idle power x its
duration) / inferences.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POWERCAP = Path("/sys/class/powercap")
DOMAINS = {"package": "intel-rapl:0", "core": "intel-rapl:0:0"}


@dataclass(frozen=True)
class Counter:
    """One RAPL domain: where its total is read, and where it wraps."""

    path: Path
    wraps_at: int

    def read(self) -> int:
        return int(self.path.read_text())

    def since(self, before: int, after: int) -> int:
        """Microjoules between two readings, across one wrap if there was one."""
        return after - before if after >= before else after + self.wraps_at - before


def counters(root: Path = POWERCAP) -> dict[str, Counter]:
    """The readable domains; empty when the counters are absent or not readable."""
    found = {}
    for name, folder in DOMAINS.items():
        total = root / folder / "energy_uj"
        limit = root / folder / "max_energy_range_uj"
        try:
            total.read_text()
            found[name] = Counter(total, int(limit.read_text()))
        except OSError, ValueError:
            continue
    return found


def measure(
    work: Callable[[], None],
    repeats: int,
    found: dict[str, Counter],
    *,
    idle_s: float = 1.0,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, float]:
    """Run `work` `repeats` times and return marginal joules per run for each domain.

    Idle power is measured for `idle_s` before and after, and its mean is taken
    away for the time the batch took.
    """

    def idle() -> dict[str, float]:
        before = {name: counter.read() for name, counter in found.items()}
        started = clock()
        sleep(idle_s)
        seconds = clock() - started
        return {
            name: counter.since(before[name], counter.read()) / 1e6 / seconds
            for name, counter in found.items()
        }

    idle_before = idle()
    before = {name: counter.read() for name, counter in found.items()}
    started = clock()
    for _ in range(repeats):
        work()
    seconds = clock() - started
    used = {
        name: counter.since(before[name], counter.read()) / 1e6 for name, counter in found.items()
    }
    idle_after = idle()
    result = {"seconds": seconds, "repeats": repeats}
    for name in found:
        idle_w = (idle_before[name] + idle_after[name]) / 2
        result[f"{name}_j"] = (used[name] - idle_w * seconds) / repeats
        result[f"{name}_idle_w"] = idle_w
        result[f"{name}_busy_w"] = used[name] / seconds
    return result


def summarise(runs: list[dict[str, float]], domain: str) -> dict[str, float]:
    """Mean joules per run over repeated batches, with a 95% interval."""
    values = np.array([run[f"{domain}_j"] for run in runs])
    half = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
    return {"mean_j": float(values.mean()), "ci_low": float(values.mean() - half),
            "ci_high": float(values.mean() + half)}  # fmt: skip
