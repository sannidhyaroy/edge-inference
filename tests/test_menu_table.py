import numpy as np
import pandas as pd
import pytest

from edge_inference.menu_table import donors, menu_measurements, route_key

IMAGES = pd.DataFrame({"image": [0, 1, 2, 3], "label": [5, 6, 7, 8]})


def tables(**changes):
    answers = pd.DataFrame({
        "model": "m", "resolution": 8, "precision": "float32", "image": [0, 1, 2, 3],
        "confidence": [0.9, 0.8, 0.7, 0.6], "correct": [True, False, True, False],
    })  # fmt: skip
    # Graph and server times on a sample: images 1 and 3 only.
    device = pd.DataFrame({"variant": "m@8/float32", "image": [1, 3], "device_ms": [10.0, 30.0]})
    prepare = pd.concat([
        pd.DataFrame({"step": "m@8", "image": [0, 1, 2, 3], "ms": [1.0, 2.0, 3.0, 4.0]}),
        pd.DataFrame({"step": "jpeg8q50", "image": [0, 1, 2, 3], "ms": [0.1, 0.2, 0.3, 0.4]}),
    ])  # fmt: skip
    photo_answers = pd.DataFrame({
        "model": "big", "resolution": 16, "precision": "int8", "photo": "jpeg8q50",
        "image": [0, 1, 2, 3], "confidence": [0.5] * 4, "correct": [True] * 4,
    })  # fmt: skip
    server = pd.DataFrame({
        "model": "big@16/int8", "photo": "jpeg8q50", "image": [1, 3],
        "receive_ms": [1.0, 3.0], "compute_ms": [100.0, 300.0],
    })  # fmt: skip
    payloads = {"jpeg8q50": pd.DataFrame({"image": [3, 2, 1, 0], "bytes": [40, 30, 20, 10]})}
    parts = {"images": IMAGES, "answers": answers, "device": device, "prepare": prepare,
             "photo_answers": photo_answers, "servers": {"edge": server}, "payloads": payloads}  # fmt: skip
    return {**parts, **changes}


def test_untimed_images_borrow_one_donor_for_every_variant_and_server():
    table = menu_measurements(**tables(), link="wifi", seed=1)
    local = table.local["m@8/float32"]
    route = table.routes[route_key("jpeg8q50", "edge", "big@16/int8")]

    # Timed images keep their own times, on the device and on the server.
    assert local.ms[[1, 3], 0].tolist() == [12.0, 34.0]
    assert route.server_ms[[1, 3]].tolist() == [101.0, 303.0]
    # An untimed image borrows a donor's graph time, but keeps its own preparation,
    # and the same donor serves it on the server.
    picks = donors(np.arange(4), np.array([1, 3]), seed=1)
    for image in (0, 2):
        donor = [1, 3][picks[image]]
        assert local.ms[image, 0] == (image + 1) + {1: 10.0, 3: 30.0}[donor]
        assert route.server_ms[image] == {1: 101.0, 3: 303.0}[donor]


def test_answers_bytes_and_encodings_follow_the_image_not_the_row_order():
    table = menu_measurements(**tables(), link="wifi")
    route = table.routes["jpeg8q50>edge/big@16/int8"]
    assert route.bytes.tolist() == [10, 20, 30, 40]
    assert route.prepare_ms.tolist() == [0.1, 0.2, 0.3, 0.4]
    assert route.link == "wifi" and route.server == "edge"
    local = table.local["m@8/float32"]
    assert local.correct[:, 0].tolist() == [True, False, True, False]
    assert local.confidence[:, 0].tolist() == [0.9, 0.8, 0.7, 0.6]
    assert table.labels.tolist() == [5, 6, 7, 8]


def test_a_block_missing_an_image_or_holding_one_twice_is_refused():
    parts = tables()
    with pytest.raises(ValueError, match="missing"):
        menu_measurements(**tables(answers=parts["answers"].iloc[:3]), link="wifi")
    doubled = pd.concat([parts["payloads"]["jpeg8q50"]] * 2)
    with pytest.raises(ValueError, match="twice"):
        menu_measurements(**tables(payloads={"jpeg8q50": doubled}), link="wifi")
