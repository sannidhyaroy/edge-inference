"""Pretrained ImageNet classifiers, exported for a device or a server.

Each model is named once here, with where its weights come from. Its input
recipe is read from the weights' own metadata, never typed by hand: torchvision
publishes a resize, crop, filter and normalisation with every set of weights,
and timm publishes a crop fraction and filter.

`export_variant` writes one ONNX graph per model and resolution (the graph's
height and width are fixed at export), the recipe beside it as JSON for the
device to read without torch, and optionally an INT8 copy calibrated on images
the caller chooses.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from edge_inference.export import export_onnx
from edge_inference.preprocess import Preprocess, load_frame
from edge_inference.quantization import ArrayCalibrationReader, quantize_onnx


@dataclass(frozen=True)
class Entry:
    """Where a model comes from: a torchvision builder and weights, or a timm name."""

    source: str
    name: str
    weights: str | None = None


ZOO = {
    "resnet18": Entry("torchvision", "resnet18", "IMAGENET1K_V1"),
    "mnv3l": Entry("torchvision", "mobilenet_v3_large", "IMAGENET1K_V2"),
    "mnv4s": Entry("timm", "mobilenetv4_conv_small.e2400_r224_in1k"),
    "mnv4m": Entry("timm", "mobilenetv4_conv_medium.e500_r224_in1k"),
    "resnet50": Entry("torchvision", "resnet50", "IMAGENET1K_V2"),
    "convnext_t": Entry("torchvision", "convnext_tiny", "IMAGENET1K_V1"),
}


def _weights(entry: Entry):
    import torchvision

    enum = type(torchvision.models.get_model_weights(entry.name).DEFAULT).__name__
    return torchvision.models.get_weight(f"{enum}.{entry.weights}")


def recipe(name: str) -> Preprocess:
    """A model's input recipe at its own resolution, read from its metadata.

    Nothing is downloaded: torchvision keeps the recipe with the weights' name,
    and timm with the model's configuration.
    """
    entry = ZOO[name]
    if entry.source == "torchvision":
        transform = _weights(entry).transforms()
        return Preprocess(
            resolution=int(transform.crop_size[0]),
            resize=int(transform.resize_size[0]),
            interpolation=transform.interpolation.value,
            mean=tuple(float(v) for v in transform.mean),
            std=tuple(float(v) for v in transform.std),
        )
    import timm

    config = timm.data.resolve_data_config({}, model=timm.create_model(entry.name))
    size = int(config["input_size"][-1])
    return Preprocess(
        resolution=size,
        resize=math.floor(size / config["crop_pct"]),
        interpolation=config["interpolation"],
        mean=tuple(float(v) for v in config["mean"]),
        std=tuple(float(v) for v in config["std"]),
    )


def reference_transform(name: str):
    """The transform the model's makers publish, for checking `recipe` against."""
    entry = ZOO[name]
    if entry.source == "torchvision":
        return _weights(entry).transforms()
    import timm

    model = timm.create_model(entry.name)
    return timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))


def build(name: str) -> nn.Module:
    """A pretrained model in evaluation mode, its weights downloaded if needed."""
    entry = ZOO[name]
    if entry.source == "torchvision":
        import torchvision

        model = torchvision.models.get_model(entry.name, weights=_weights(entry))
    else:
        import timm

        model = timm.create_model(entry.name, pretrained=True)
    return model.eval()


def variant_path(directory: Path, name: str, resolution: int, precision: str) -> Path:
    """Where a model's graph at one resolution and precision is kept."""
    suffix = ".int8.onnx" if precision == "int8" else ".onnx"
    return directory / f"{name}@{resolution}{suffix}"


def recipe_path(directory: Path, name: str, resolution: int) -> Path:
    return directory / f"{name}@{resolution}.json"


def export_variant(
    name: str,
    resolution: int,
    directory: Path,
    *,
    calibration: Sequence[Path] = (),
    model: nn.Module | None = None,
) -> dict[str, Path]:
    """Export `name` at `resolution`, its recipe, and an INT8 copy if calibration images are given.

    Files already present are kept, so an interrupted run resumes.
    """
    prepare = recipe(name).at(resolution)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"recipe": recipe_path(directory, name, resolution)}
    prepare.save(paths["recipe"])
    paths["float32"] = variant_path(directory, name, resolution, "float32")
    if not paths["float32"].exists():
        export_onnx(
            model if model is not None else build(name), paths["float32"], image_size=resolution
        )
    if calibration:
        paths["int8"] = variant_path(directory, name, resolution, "int8")
        if not paths["int8"].exists():
            batches = [prepare(load_frame(path)) for path in calibration]
            quantize_onnx(paths["float32"], paths["int8"], ArrayCalibrationReader(batches))
    return paths
