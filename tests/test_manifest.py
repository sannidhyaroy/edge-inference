from pathlib import Path

import pandas as pd
import pytest

from edge_inference.manifest import (
    IMAGENET_TRAIN,
    IMAGENET_VAL,
    accuracy_by_origin,
    manifest,
    origin_of,
    wilson,
)


def test_origin_is_read_from_the_imagenet_file_name():
    assert origin_of("val/n01440764/ILSVRC2012_val_00009111.JPEG") == IMAGENET_VAL
    assert origin_of(Path("val/n01440764/n01440764_10040.JPEG")) == IMAGENET_TRAIN


def test_manifest_keeps_index_order_and_relative_paths(tmp_path):
    files = [tmp_path / "val" / "a" / "ILSVRC2012_val_1.JPEG", tmp_path / "val" / "b" / "b_2.JPEG"]
    table = manifest(files, [3, 7], root=tmp_path)
    assert table["image"].tolist() == [0, 1]
    assert table["label"].tolist() == [3, 7]
    assert table["file"].tolist() == ["val/a/ILSVRC2012_val_1.JPEG", "val/b/b_2.JPEG"]
    assert table["origin"].tolist() == [IMAGENET_VAL, IMAGENET_TRAIN]


def test_manifest_refuses_mismatched_lengths():
    with pytest.raises(ValueError):
        manifest([Path("a.JPEG")], [1, 2])


def test_wilson_interval_contains_the_estimate_and_stays_in_bounds():
    low, high = wilson(97, 100)
    assert low < 0.97 < high <= 1.0
    low, high = wilson(10, 10)
    assert low < 1.0 and high == pytest.approx(1.0)
    # 131 of 134 right, as for the smallest group here: about 93.6% to 99.2%.
    low, high = wilson(131, 134)
    assert low == pytest.approx(0.9363, abs=1e-3)
    assert high == pytest.approx(0.9923, abs=1e-3)


def test_accuracy_by_origin_groups_by_exit_and_origin():
    images = pd.DataFrame(
        {
            "image": [0, 1, 2, 3],
            "label": [0] * 4,
            "file": list("abcd"),
            "origin": [IMAGENET_VAL, IMAGENET_TRAIN, IMAGENET_TRAIN, IMAGENET_TRAIN],
        }
    )
    profile = pd.DataFrame(
        {
            "image": [0, 1, 2, 3, 0, 1, 2, 3],
            "exit_name": ["layer2"] * 4 + ["layer4"] * 4,
            "correct": [True, False, True, False, True, True, True, False],
        }
    )
    table = accuracy_by_origin(profile, images).set_index(["exit_name", "origin"])
    assert table.loc[("layer2", IMAGENET_TRAIN), "accuracy"] == pytest.approx(1 / 3)
    assert table.loc[("layer4", IMAGENET_TRAIN), "correct"] == 2
    assert table.loc[("layer4", IMAGENET_VAL), "images"] == 1


def test_accuracy_by_origin_refuses_images_missing_from_the_manifest():
    images = pd.DataFrame({"image": [0], "label": [0], "file": ["a"], "origin": [IMAGENET_VAL]})
    profile = pd.DataFrame({"image": [0, 5], "exit_name": ["layer4"] * 2, "correct": [True, True]})
    with pytest.raises(ValueError):
        accuracy_by_origin(profile, images)
