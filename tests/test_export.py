"""Sanity checks for ONNX export.

A tiny network is used rather than the real backbone. These assert properties
of the export path itself, which do not depend on the model being large, and
keeping it small keeps the suite fast enough to run on every commit.
"""

import torch
from torch import nn

from edge_inference.export import DEFAULT_OPSET, export_onnx, exported_opset, verify_parity

IMAGE_SIZE = 32


def build_tiny_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 4),
    )


def test_onnx_output_matches_pytorch(tmp_path):
    """The whole point of the export step: the two engines must agree."""
    model = build_tiny_model()
    path = export_onnx(model, tmp_path / "tiny.onnx", image_size=IMAGE_SIZE)

    result = verify_parity(model, path, image_size=IMAGE_SIZE, samples=4)

    assert result["predictions_match"], "ONNX Runtime predicted a different class"
    assert result["within_tolerance"], f"max error {result['max_abs_error']:.3e} too large"


def test_export_is_self_contained(tmp_path):
    """Weights belong in the file, not in a sibling blob.

    torch defaults to external data, which leaves a small graph plus a
    `.onnx.data` file. Measuring the graph alone would report a model as a
    fraction of its real size and make the quantization comparison meaningless.
    """
    model = build_tiny_model()
    path = export_onnx(model, tmp_path / "tiny.onnx", image_size=IMAGE_SIZE)

    assert not path.with_suffix(".onnx.data").exists()
    assert not list(tmp_path.glob("*.data"))

    parameter_bytes = sum(p.numel() * 4 for p in model.parameters())
    assert path.stat().st_size >= parameter_bytes


def test_exported_opset_is_read_back(tmp_path):
    """A requested opset that silently did not apply would misdescribe the graph."""
    model = build_tiny_model()
    path = export_onnx(model, tmp_path / "tiny.onnx", image_size=IMAGE_SIZE)

    assert exported_opset(path) == DEFAULT_OPSET


def test_batch_axis_is_dynamic(tmp_path):
    """Calibration feeds batches through the same file that inference uses."""
    from edge_inference.export import INPUT_NAME, build_session

    model = build_tiny_model()
    path = export_onnx(model, tmp_path / "tiny.onnx", image_size=IMAGE_SIZE)
    session = build_session(path)

    for batch in (1, 5):
        inputs = torch.randn(batch, 3, IMAGE_SIZE, IMAGE_SIZE).numpy()
        outputs = session.run(None, {INPUT_NAME: inputs})[0]
        assert outputs.shape[0] == batch


def test_chained_stages_match_every_exit(tmp_path):
    """Stage graphs fed each other's outputs must reproduce all three exits.

    This is the deployed path for early exit, so a stage that exported wrongly
    or a features tensor handed over in the wrong shape must fail here.
    """
    from edge_inference.export import export_exit_stages, verify_stage_parity
    from edge_inference.models import EXIT_NAMES, build_early_exit

    torch.manual_seed(0)
    model = build_early_exit(num_classes=4)
    paths = export_exit_stages(model, tmp_path / "tiny", image_size=IMAGE_SIZE)

    rows = verify_stage_parity(model, paths, image_size=IMAGE_SIZE, samples=4)

    assert [row["exit_name"] for row in rows] == list(EXIT_NAMES)
    for row in rows:
        assert row["predictions_match"], f"{row['exit_name']} predicted a different class"
        assert row["within_tolerance"], f"{row['exit_name']} max error {row['max_abs_error']:.3e}"


SHARED_POOL_CHECK = """
import sys
from pathlib import Path

import torch

from edge_inference.export import (
    FEATURES_NAME, OUTPUT_NAME, build_session, build_stage_sessions, export_exit_stages, run_stage,
)
from edge_inference.models import build_early_exit

torch.manual_seed(0)
paths = export_exit_stages(build_early_exit(num_classes=4), Path(sys.argv[1]) / "tiny", image_size=32)
sessions = build_stage_sessions(paths, threads=1)

x = torch.randn(1, 3, 32, 32).numpy()
for session in sessions:
    outputs = run_stage(session, x)
    x = outputs.get(FEATURES_NAME)
assert outputs[OUTPUT_NAME].shape == (1, 4)

build_session(paths[0], threads=1)  # joins the shared pool instead of failing

try:
    build_stage_sessions(paths, threads=2)
except RuntimeError as error:
    assert "separate process" in str(error)
else:
    raise AssertionError("a second pool size was accepted")
"""


def test_shared_pool_stages_chain_and_refuse_a_second_size(tmp_path):
    """Stages on the shared pool must chain like any others, at one size only.

    A second thread count in the same process cannot take effect, since the
    pool is sized once, so it must raise rather than quietly time stages at
    the first count and record them as the second. Runs in its own process
    because the pool, once created, outlives the test and would force every
    later session onto it.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", SHARED_POOL_CHECK, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
