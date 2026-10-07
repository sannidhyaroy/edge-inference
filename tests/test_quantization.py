"""Sanity checks for quantizing early-exit stages."""

import torch

from edge_inference.export import (
    FEATURES_NAME,
    OUTPUT_NAME,
    build_session,
    export_exit_stages,
    run_stage,
    stage_paths,
)
from edge_inference.models import EXIT_NAMES, build_early_exit
from edge_inference.quantization import quantize_exit_stages

IMAGE_SIZE = 32
NUM_CLASSES = 4


def test_quantized_stages_chain_and_shrink(tmp_path):
    """Each later stage must accept the features of the quantized stage before it.

    Calibration feeds every stage its own kind of input, so a stage calibrated
    on the wrong tensor would fail here rather than in a long profiling run.
    """
    torch.manual_seed(0)
    model = build_early_exit(num_classes=NUM_CLASSES)
    base = tmp_path / "tiny"
    sources = export_exit_stages(model, base, image_size=IMAGE_SIZE)
    batches = [torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE).numpy() for _ in range(2)]

    destinations = quantize_exit_stages(sources, stage_paths(base, suffix=".int8.onnx"), batches)

    for source, destination in zip(sources, destinations, strict=True):
        assert destination.stat().st_size < source.stat().st_size / 2

    x = batches[0][:1]
    for name, path in zip(EXIT_NAMES, destinations, strict=True):
        outputs = run_stage(build_session(path), x)
        assert outputs[OUTPUT_NAME].shape == (1, NUM_CLASSES), name
        x = outputs.get(FEATURES_NAME)
