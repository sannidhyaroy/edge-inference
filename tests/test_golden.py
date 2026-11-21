"""The committed scoreboards must reproduce exactly from the committed profiles.

These tests guard the published results: whatever changes inside the
simulator, the rules or the yardsticks, the numbers and plans already
published for the early-exit model must come out the same.

The quick test runs on every commit. The full one, with every SplitEE variant
and the scoreboard where rules learn the network from their own sends, takes
about a minute; set EDGE_GOLDEN_FULL=1 to run it.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edge_inference.offload import load_components
from edge_inference.scoreboard import Setup, run_setup
from edge_inference.simulate import Measurements

RESULTS = Path(__file__).resolve().parents[1] / "results"
STEM = "resnet18_early_exit_imagenette"
SERVERS = {"t4": "pytorch_float32_cuda", "cpu_t6": "onnxruntime_float32_t6"}
SCENARIO = "t4_8x_30ms_wifi-4g_stay0.95-0.95"

pytestmark = pytest.mark.skipif(
    not (RESULTS / f"offload_components_{STEM}.csv.gz").exists()
    and not (RESULTS / f"exit_profile_{STEM}_onnxruntime_float32_t2.csv.gz").exists(),
    reason="the early-exit profiles are not present",
)


def committed(tag: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"decide_scoreboard_{STEM}_{tag}.csv")


def table() -> Measurements:
    return Measurements.from_components(
        load_components(RESULTS, STEM, threads=2, servers=SERVERS), "t4"
    )


def assert_same(old: pd.DataFrame, new: pd.DataFrame) -> None:
    """Same rules, same plans, same numbers, for every rule the old board has."""
    new = new.set_index("rule").loc[old["rule"]].reset_index()
    assert new["plans"].astype(str).tolist() == old["plans"].astype(str).tolist()
    numbers = [
        column
        for column in old.columns
        if column != "decide_us" and pd.api.types.is_numeric_dtype(old[column])
    ]
    np.testing.assert_allclose(
        new[numbers].to_numpy(float), old[numbers].to_numpy(float), rtol=0, atol=1e-12
    )


def test_main_scoreboard_reproduces():
    """Every rule except the SplitEE variants a sweep skips."""
    old = committed(SCENARIO)
    new = run_setup(table(), Setup(), every_variant=False).board
    assert_same(old[old["rule"].isin(new["rule"])], new)


@pytest.mark.skipif(not os.environ.get("EDGE_GOLDEN_FULL"), reason="set EDGE_GOLDEN_FULL=1")
@pytest.mark.parametrize("knowledge", ["exact", "sends"])
def test_full_scoreboards_reproduce(knowledge):
    tag = SCENARIO if knowledge == "exact" else f"{SCENARIO}_{knowledge}"
    old = committed(tag)
    new = run_setup(table(), Setup(knowledge=knowledge)).board
    assert len(new) == len(old)
    assert_same(old, new)
