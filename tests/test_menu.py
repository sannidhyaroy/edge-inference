import numpy as np
import onnx
import pandas as pd
import pytest
from onnx import TensorProto, helper
from PIL import Image

from edge_inference.menu import (
    TEST,
    TUNE,
    answer_photos,
    check_manifest,
    imagenetv2_manifest,
    profile_preparation,
    timing_sample,
)


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


def test_preparation_times_every_recipe_on_every_image_and_rotates_their_order():
    order = []

    def recipe(name):
        return lambda frame: order.append((frame, name))

    ticks = iter(range(100))
    rows = profile_preparation(
        {"a": recipe("a"), "b": recipe("b"), "c": recipe("c")},
        [(7, "x.jpg"), (8, "y.jpg")],
        load=lambda path: path,
        clock=lambda: next(ticks) / 1000.0,
        cpus=[],
    )
    assert len(rows) == 6
    assert set(zip(rows["image"], rows["step"], strict=True)) == {
        (i, r) for i in (7, 8) for r in "abc"
    }
    # Each image is loaded once and handed to every recipe; the order shifts by one.
    assert order == [("x.jpg", "a"), ("x.jpg", "b"), ("x.jpg", "c"),
                     ("y.jpg", "b"), ("y.jpg", "c"), ("y.jpg", "a")]  # fmt: skip
    assert np.allclose(rows["ms"], 1.0)


def batched_tiny_model(path, resolution=16, classes=5):
    """Average the image to three numbers, then score five classes; any batch size."""
    weights = np.array([[1, 0, 0, 2, 0], [0, 1, 0, 0, 2], [0, 0, 1, 1, 1]], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"]),
            helper.make_node("MatMul", ["flat", "w"], ["logits"]),
        ],
        "tiny",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["n", 3, resolution, resolution]
            )
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n", classes])],
        [helper.make_tensor("w", TensorProto.FLOAT, weights.shape, weights.flatten())],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)


def test_photo_answers_match_what_the_menu_server_answers_for_the_same_bytes(tmp_path):
    from edge_inference.preprocess import Preprocess, encode_photo
    from edge_inference.serving import MenuServer, open_model

    batched_tiny_model(tmp_path / "tiny@16.onnx")
    Preprocess(16, 18, "bilinear", (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)).save(
        tmp_path / "tiny@16.json"
    )
    photos = tmp_path / "jpeg20q90"
    photos.mkdir()
    colours = [(250, 10, 10), (10, 250, 10), (10, 10, 250)]
    for image, colour in enumerate(colours):
        frame = Image.fromarray(np.full((30, 40, 3), colour, dtype=np.uint8))
        (photos / f"{image}.jpg").write_bytes(encode_photo(frame, 20, 90))
    index = pd.DataFrame({"image": [0, 1, 2], "label": [0, 1, 4], "split": [TUNE, TEST, TEST]})

    found = answer_photos(
        (tmp_path, "tiny", 16, {"float32": tmp_path / "tiny@16.onnx"}, photos, index, 1)
    )

    served = open_model(tmp_path / "tiny@16.onnx", tmp_path / "tiny@16.json", threads=1)
    server = MenuServer({"tiny": served}, "tiny")
    expected = [
        server.handle({"kind": "photo"}, (photos / f"{i}.jpg").read_bytes())["predicted"]
        for i in range(3)
    ]
    assert found["predicted"].tolist() == expected
    assert found["correct"].tolist() == [
        p == label for p, label in zip(expected, [0, 1, 4], strict=True)
    ]
    assert set(found["photo"]) == {"jpeg20q90"} and set(found["model"]) == {"tiny"}


def test_timing_uses_the_same_test_images_whatever_order_the_table_comes_in():
    table = pd.DataFrame({"image": range(40), "split": [TUNE, TEST] * 20})
    first = timing_sample(table, 5)
    again = timing_sample(table.sample(frac=1, random_state=3), 5)
    assert first["image"].tolist() == again["image"].tolist()
    assert (first["split"] == TEST).all() and first["image"].is_monotonic_increasing
    assert len(timing_sample(table, 0)) == 20
