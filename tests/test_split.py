"""Sanity checks for split payload profiling."""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from edge_inference.export import build_session, export_exit_stages
from edge_inference.models import build_early_exit
from edge_inference.split import (
    SPLIT_NAMES,
    UINT8_HEADER_BYTES,
    dequantize_uint8,
    profile_splits,
    quantize_uint8,
)

IMAGE_SIZE = 32


def test_uint8_round_trip_is_within_half_a_level():
    """Each value comes back within half a quantization step of where it was."""
    rng = np.random.default_rng(0)
    features = np.maximum(rng.normal(size=(1, 8, 4, 4)), 0).astype(np.float32)

    levels, scale, zero_point = quantize_uint8(features)
    restored = dequantize_uint8(levels, scale, zero_point)

    assert levels.dtype == np.uint8
    assert np.abs(restored - features).max() <= scale / 2 + 1e-6
    # Post-ReLU maps are non-negative, so zero must survive exactly: it is the
    # most common value, and the reason the payload compresses at all.
    assert np.all(restored[features == 0] == 0)


def test_one_row_per_image_per_split(tmp_path):
    torch.manual_seed(0)
    model = build_early_exit(num_classes=4)
    paths = export_exit_stages(model, tmp_path / "tiny", image_size=IMAGE_SIZE)
    sessions = [build_session(path) for path in paths]
    dataset = TensorDataset(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE), torch.tensor([0, 1]))

    rows = profile_splits(sessions, dataset, threads=1, warmup=1, precision="float32")

    assert [(row["image"], row["split_name"]) for row in rows] == [
        (image, name) for image in range(2) for name in SPLIT_NAMES
    ]
    for row in rows:
        assert row["float32_bytes"] == row["elements"] * 4
        assert row["uint8_bytes"] == row["elements"] + UINT8_HEADER_BYTES
        assert row["jpeg_bytes"] is None
