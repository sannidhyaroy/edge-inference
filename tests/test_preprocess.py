"""The device's Pillow and NumPy preprocessing matches what each model's makers publish."""

import numpy as np
import pytest
from PIL import Image

from edge_inference.preprocess import (
    Preprocess,
    centre_crop,
    decode_photo,
    encode_photo,
    resize_shorter,
)

torch = pytest.importorskip("torch")
v2 = pytest.importorskip("torchvision.transforms.v2")
zoo = pytest.importorskip("edge_inference.zoo")


def frames():
    rng = np.random.default_rng(0)
    # Landscape, portrait, square, and odd sizes that make the crop round.
    for width, height in ((500, 375), (333, 500), (256, 256), (641, 427)):
        yield Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.parametrize("name", sorted(zoo.ZOO))
def test_the_recipe_matches_the_published_transform(name):
    reference = zoo.reference_transform(name)
    recipe = zoo.recipe(name)
    for frame in frames():
        expected = reference(frame).numpy()[None]
        np.testing.assert_allclose(recipe(frame), expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize("name", ["resnet18", "mnv4s"])
@pytest.mark.parametrize("resolution", [160, 112])
def test_other_resolutions_keep_the_resize_ratio(name, resolution):
    native = zoo.recipe(name)
    recipe = native.at(resolution)
    assert recipe.resize == round(resolution * native.resize / native.resolution)
    interpolation = {
        "bilinear": v2.InterpolationMode.BILINEAR,
        "bicubic": v2.InterpolationMode.BICUBIC,
    }[recipe.interpolation]
    reference = v2.Compose(
        [
            v2.Resize(recipe.resize, interpolation=interpolation, antialias=True),
            v2.CenterCrop(resolution),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(recipe.mean, recipe.std),
        ]
    )
    for frame in frames():
        np.testing.assert_allclose(recipe(frame), reference(frame).numpy()[None], rtol=0, atol=1e-5)


def test_resize_keeps_the_aspect_ratio_and_truncates():
    assert resize_shorter(Image.new("RGB", (500, 375)), 256, "bilinear").size == (341, 256)
    assert resize_shorter(Image.new("RGB", (333, 500)), 160, "bicubic").size == (160, 240)


def test_the_crop_is_centred():
    image = Image.new("RGB", (10, 7))
    assert centre_crop(image, 4).size == (4, 4)


def test_a_recipe_survives_a_round_trip_through_json(tmp_path):
    recipe = Preprocess(160, 183, "bicubic", (0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    recipe.save(tmp_path / "r.json")
    assert Preprocess.load(tmp_path / "r.json") == recipe


def test_an_encoded_photo_decodes_at_the_requested_size():
    frame = next(frames())
    photo = encode_photo(frame, 256, 85)
    assert decode_photo(photo).size == (341, 256)
    assert encode_photo(frame, 256, 85) == photo
