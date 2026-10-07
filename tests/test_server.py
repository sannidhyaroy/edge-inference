"""Sanity checks for server-side profiling."""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from edge_inference.config import IMAGE_SIZE
from edge_inference.export import build_session, export_exit_stages
from edge_inference.models import build_early_exit
from edge_inference.server import OnnxServer, TorchServer, profile_server
from edge_inference.split import SPLIT_NAMES


def make_images(tmp_path, count):
    """A dataset of preprocessed tensors and matching JPEG files on disk."""
    rng = np.random.default_rng(0)
    files = []
    for index in range(count):
        path = tmp_path / f"{index}.jpg"
        pixels = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path, quality=90)
        files.append(path)
    images = torch.randn(count, 3, IMAGE_SIZE, IMAGE_SIZE)
    return TensorDataset(images, torch.arange(count)), files


def test_every_arrival_mode_is_timed(tmp_path):
    torch.manual_seed(0)
    dataset, files = make_images(tmp_path, 2)
    server = TorchServer(build_early_exit(num_classes=4), "cpu")

    rows = profile_server(server, dataset, files, warmup=1, labels={"runtime": "pytorch"})

    expected = [("offload", None, "jpeg")] + [
        ("split", name, payload) for name in SPLIT_NAMES for payload in ("float32", "uint8_zstd")
    ]
    for image in range(2):
        mine = [row for row in rows if row["image"] == image]
        assert [(row["mode"], row["split_name"], row["payload"]) for row in mine] == expected
        assert mine[0]["payload_bytes"] == files[image].stat().st_size
        for row in mine:
            assert row["total_ms"] == row["receive_ms"] + row["compute_ms"]
            assert row["runtime"] == "pytorch"


def test_backends_agree(tmp_path):
    """A GPU server and a CPU server must give the same answers to the same frames.

    Their timings are what is being compared later, so a backend that ran a
    different network, or started from the wrong stage, must fail here.
    """
    torch.manual_seed(0)
    dataset, files = make_images(tmp_path, 2)
    model = build_early_exit(num_classes=4)
    sessions = [build_session(path) for path in export_exit_stages(model, tmp_path / "tiny")]

    by_torch = profile_server(TorchServer(model, "cpu"), dataset, files, warmup=0, labels={})
    by_onnx = profile_server(OnnxServer(sessions), dataset, files, warmup=0, labels={})

    assert [row["predicted"] for row in by_onnx] == [row["predicted"] for row in by_torch]
