import numpy as np
import pandas as pd
import pytest
from PIL import Image

from edge_inference.menu import TEST, TUNE, check_manifest, imagenetv2_manifest


@pytest.fixture
def root(tmp_path):
    # Folders named as ImageNetV2 names them: by class index, so "10" sorts
    # before "2" as text.
    rng = np.random.default_rng(0)
    for label in ("0", "1", "10", "2"):
        folder = tmp_path / label
        folder.mkdir()
        for index in range(10):
            pixels = rng.integers(0, 256, (12 + index, 16, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"{index:02d}.jpeg")
    return tmp_path


def test_labels_are_the_folder_numbers_and_images_are_indexed_in_label_order(root):
    table = imagenetv2_manifest(root)
    assert table["label"].tolist() == sorted(table["label"].tolist())
    assert set(table["label"]) == {0, 1, 2, 10}
    assert table.loc[table["file"].str.startswith("10/"), "label"].eq(10).all()
    assert table["image"].tolist() == list(range(40))


def test_each_class_is_split_in_half_and_calibration_comes_from_tuning(root):
    table = imagenetv2_manifest(root)
    counts = table.groupby(["label", "split"]).size().unstack()
    assert (counts[TUNE] == 5).all() and (counts[TEST] == 5).all()
    assert table.loc[table["calibration"], "split"].eq(TUNE).all()
    assert table.groupby("label")["calibration"].sum().eq(1).all()
    check_manifest(table, root)


def test_the_split_is_fixed_by_the_seed(root):
    first, second = imagenetv2_manifest(root), imagenetv2_manifest(root)
    pd.testing.assert_frame_equal(first, second)
    other = imagenetv2_manifest(root, seed=7)
    assert not first["split"].equals(other["split"])


def test_a_changed_file_is_caught(root):
    table = imagenetv2_manifest(root)
    (root / table["file"].iloc[0]).write_bytes(b"not the same image")
    with pytest.raises(ValueError):
        check_manifest(table, root)
