"""A hand-built table shared by the simulator, rule and yardstick tests.

Four images, round numbers throughout, so expected values can be worked out on
paper. Both precisions share confidences and answers; int8 is twice as fast.

* Device time to each exit: 2, 4, 6 ms in float32; 1, 2, 3 ms in int8.
* Confidence per exit: image 0 is sure from the start, image 1 from the second
  exit, image 2 only at the end, image 3 never.
* Local answers: image 0 is right everywhere, image 1 from the second exit,
  image 2 only at the final exit, image 3 nowhere.
* The server is right on every image.
* The photo is 1000 bytes and takes the server 4 ms. Feature maps take the
  server 1.5 ms: 4000 bytes raw, or 500 bytes after 0.5 ms of packing.

Over `NETWORK` (10 ms round trip, 8 Mbps) 1000 bytes take 1 ms, so the photo
costs 10 + 1 + 4 = 15 ms from a standing start.
"""

import numpy as np

from edge_inference.offload import Network
from edge_inference.simulate import Measurements

NETWORK = Network(rtt_ms=10, uplink_mbps=8)
IMAGES = 4

CONFIDENCE = np.array([[0.9, 0.9, 0.9], [0.2, 0.8, 0.9], [0.2, 0.3, 0.9], [0.2, 0.3, 0.4]])
CORRECT = np.array(
    [[True, True, True], [False, True, True], [False, False, True], [False, False, False]]
)


def arrays(rows: slice = slice(None)) -> dict:
    """The table's arrays, as keyword arguments for `Measurements.from_arrays`.

    `rows` keeps only some images, so a test can change a few arrays and build
    a smaller table from the rest.
    """

    def per_image(*values: float) -> np.ndarray:
        return np.tile(np.array(values, dtype=float), (IMAGES, 1))

    whole = {
        "images": np.arange(IMAGES) + 100,
        "labels": np.array([0, 0, 1, 1]),
        "device_ms": {"float32": per_image(2, 4, 6), "int8": per_image(1, 2, 3)},
        "confidence": {"float32": CONFIDENCE.copy(), "int8": CONFIDENCE.copy()},
        "correct": {"float32": CORRECT.copy(), "int8": CORRECT.copy()},
        "photo_bytes": np.full(IMAGES, 1000.0),
        "photo_server_ms": np.full(IMAGES, 4.0),
        "photo_correct": np.ones(IMAGES, dtype=bool),
        "features_bytes": {"float32": per_image(4000, 4000), "uint8_zstd": per_image(500, 500)},
        "features_prepare_ms": {"float32": per_image(0, 0), "uint8_zstd": per_image(0.5, 0.5)},
        "features_server_ms": {"float32": per_image(1.5, 1.5), "uint8_zstd": per_image(1.5, 1.5)},
        "features_correct": {
            "float32": np.ones((IMAGES, 2), dtype=bool),
            "uint8_zstd": np.ones((IMAGES, 2), dtype=bool),
        },
    }
    return {
        name: {key: array[rows] for key, array in value.items()}
        if isinstance(value, dict)
        else value[rows]
        for name, value in whole.items()
    }


def measurements(**changes) -> Measurements:
    """The table, with any of its arrays replaced by `changes`."""
    return Measurements.from_arrays(**{**arrays(), **changes})
