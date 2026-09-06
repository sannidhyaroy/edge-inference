"""Sanity checks for the latency harness.

These are not comprehensive tests. They exist to catch the failure mode that
would quietly invalidate every number in the report: a timing harness that
reports plausible looking values without measuring anything.
"""

import time

from edge_inference.bench import machine_info, summarise, time_callable


def test_timing_reflects_actual_work():
    """A function that sleeps 5 ms must be timed at roughly 5 ms."""

    def sleep_briefly() -> None:
        time.sleep(0.005)

    times_ms = time_callable(sleep_briefly, warmup=1, runs=5)

    assert len(times_ms) == 5
    # Generous lower bound: sleep guarantees at least the requested duration,
    # while the upper bound stays loose because a shared CPU can overshoot.
    assert all(4.0 <= t < 500.0 for t in times_ms)


def test_warmup_runs_are_not_reported():
    calls = []

    def record() -> None:
        calls.append(1)

    times_ms = time_callable(record, warmup=3, runs=4)

    assert len(calls) == 7
    assert len(times_ms) == 4


def test_summary_statistics_are_ordered():
    stats = summarise([1.0, 2.0, 3.0, 4.0, 100.0])

    assert stats["min_ms"] <= stats["median_ms"] <= stats["p95_ms"] <= stats["max_ms"]
    assert stats["median_ms"] == 3.0
    # The median must ignore the outlier that drags the mean upward. This is
    # the whole reason results report the median rather than the mean.
    assert stats["median_ms"] < stats["mean_ms"]


def test_machine_info_is_populated():
    info = machine_info()

    for key in ("machine", "cpu", "system", "arch", "torch_version"):
        assert info[key], f"{key} should not be empty"

    # Logical cores are always knowable. Physical cores are best effort per
    # platform, so only their type is asserted, not their presence.
    assert isinstance(info["cores_logical"], int)
    assert info["cores_logical"] >= 1
    assert info["cores_physical"] is None or isinstance(info["cores_physical"], int)


def test_machine_info_excludes_hostname():
    """Results files are committed to a public repo, so no personal identifiers."""
    import platform

    info = machine_info()

    assert "hostname" not in info
    assert platform.node() not in info.values()
