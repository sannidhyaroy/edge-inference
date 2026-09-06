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
