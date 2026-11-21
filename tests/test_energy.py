import pytest

from edge_inference.energy import Counter, counters, measure, summarise


def fake_domain(tmp_path, folder, start, wraps_at=1000):
    path = tmp_path / folder
    path.mkdir()
    (path / "energy_uj").write_text(str(start))
    (path / "max_energy_range_uj").write_text(str(wraps_at))
    return path / "energy_uj"


def test_the_difference_survives_a_wrap(tmp_path):
    counter = Counter(fake_domain(tmp_path, "intel-rapl:0", 0), wraps_at=1000)
    assert counter.since(100, 400) == 300
    assert counter.since(900, 50) == 150


def test_unreadable_domains_are_left_out(tmp_path):
    fake_domain(tmp_path, "intel-rapl:0", 5)
    assert set(counters(tmp_path)) == {"package"}


def test_idle_power_is_taken_away(tmp_path):
    total = fake_domain(tmp_path, "intel-rapl:0", 0, wraps_at=10**9)
    found = counters(tmp_path)
    state = {"energy": 0.0, "time": 0.0}

    def spend(seconds, watts):
        state["time"] += seconds
        state["energy"] += seconds * watts
        total.write_text(str(int(state["energy"] * 1e6)))

    # Idle draws 2 W. Each run of work takes 0.1 s at 7 W: 0.5 J above idle.
    result = measure(
        lambda: spend(0.1, 7.0),
        10,
        found,
        idle_s=1.0,
        clock=lambda: state["time"],
        sleep=lambda seconds: spend(seconds, 2.0),
    )
    assert result["package_idle_w"] == pytest.approx(2.0)
    assert result["package_j"] == pytest.approx(0.5)
    assert summarise([result, result], "package")["mean_j"] == pytest.approx(0.5)
