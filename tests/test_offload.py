"""Checks for joining profiles into per-image costs and applying a network.

Profiles are built by hand with round numbers, so every expected value below
can be worked out on paper.
"""

import math

import pandas as pd

from edge_inference.models import EXIT_NAMES
from edge_inference.offload import (
    Network,
    breakeven_mbps,
    breakeven_slowdown,
    build_components,
    summarise,
    total_ms,
)
from edge_inference.split import SPLIT_NAMES

IMAGES = (0, 1)


def exit_profile(scale):
    """Reaching exit k takes k * scale ms; every answer is right."""
    return pd.DataFrame(
        [
            {
                "image": image,
                "label": image,
                "exit_name": name,
                "cumulative_ms": (index + 1) * scale,
                "confidence": 0.5,
                "correct": True,
            }
            for image in IMAGES
            for index, name in enumerate(EXIT_NAMES)
        ]
    )


def split_profile():
    return pd.DataFrame(
        [
            {
                "image": image,
                "label": image,
                "split_name": name,
                "float32_bytes": 4000,
                "uint8_zstd_bytes": 500,
                "uint8_quantize_ms": 0.25,
                "uint8_zstd_ms": 0.25,
            }
            for image in IMAGES
            for name in SPLIT_NAMES
        ]
    )


def server_profile():
    rows = []
    for image in IMAGES:
        rows.append(
            {
                "image": image,
                "split_name": None,
                "payload": "jpeg",
                "payload_bytes": 1000,
                "receive_ms": 1.0,
                "compute_ms": 3.0,
                "correct": True,
                "label": image,
            }
        )
        for name in SPLIT_NAMES:
            for payload in ("float32", "uint8_zstd"):
                rows.append(
                    {
                        "image": image,
                        "split_name": name,
                        "payload": payload,
                        "payload_bytes": 0,
                        "receive_ms": 0.5,
                        "compute_ms": 1.0,
                        # The server gets image 1 wrong after an 8-bit payload.
                        "correct": not (image == 1 and payload == "uint8_zstd"),
                        "label": image,
                    }
                )
    return pd.DataFrame(rows)


def components():
    return build_components(
        {"float32": exit_profile(2.0), "int8": exit_profile(1.0)},
        split_profile(),
        {"t4": server_profile()},
    )


def test_every_mode_appears_once_per_image():
    frame = components()

    # 3 exits x 2 precisions locally, 2 split points x 2 payloads, 1 offload.
    assert len(frame) == len(IMAGES) * (6 + 4 + 1)
    keys = ["image", "mode", "point", "device_precision", "payload", "server"]
    assert not frame.duplicated(keys).any()


def test_split_rows_join_device_payload_and_server():
    frame = components()
    row = frame[
        (frame["image"] == 1)
        & (frame["mode"] == "split")
        & (frame["point"] == "layer3")
        & (frame["payload"] == "uint8_zstd")
    ].iloc[0]

    assert row["device_ms"] == 4.0  # float32 device, second exit
    assert row["prepare_ms"] == 0.5
    # The device's payload size, not the server's copy, crosses the network.
    assert row["payload_bytes"] == 500
    assert row["server_receive_ms"] + row["server_compute_ms"] == 1.5
    assert not row["correct"]


def test_network_is_added_only_when_something_is_sent():
    frame = components()
    network = Network(rtt_ms=10, uplink_mbps=8)
    times = frame.assign(total=total_ms(frame, network)).set_index(
        ["image", "mode", "point", "device_precision", "payload", "server"]
    )["total"]

    # Local, int8, final exit: 3 ms of device time and nothing else.
    assert times[(0, "local", "layer4", "int8", "none", "none")] == 3.0
    # Offload: 10 ms round trip + 1000 bytes at 8 Mbps (1 ms) + 4 ms server.
    assert times[(0, "offload", "input", "none", "jpeg", "t4")] == 15.0
    # Split after layer2 as float32: 2 ms device + 10 + 4 ms transfer + 1.5 ms server.
    assert times[(0, "split", "layer2", "float32", "float32", "t4")] == 17.5


def test_summary_counts_deadlines_and_accuracy():
    summary = summarise(components(), {"lan": Network(rtt_ms=10, uplink_mbps=8)}, (16,))
    offload = summary[summary["mode"] == "offload"].iloc[0]
    split = summary[(summary["payload"] == "uint8_zstd") & (summary["point"] == "layer2")].iloc[0]

    assert offload["within_16ms"] == 1.0  # 15 ms
    assert split["accuracy"] == 0.5  # image 1 answered wrongly


def test_breakeven_bandwidth_matches_the_formula():
    table = breakeven_mbps(components(), rtt_ms=10, local_ms=20).set_index(
        ["mode", "point", "payload", "server"]
    )
    offload = table.loc[("offload", "input", "jpeg", "t4")]

    # 20 = 10 + 4 + 8000 bits / bandwidth, so bandwidth = 8000 bits / 6 ms.
    assert math.isclose(offload["breakeven_mbps"], 8000 / 6 / 1000)

    hopeless = breakeven_mbps(components(), rtt_ms=10, local_ms=12)
    assert math.isinf(hopeless.set_index("mode").loc["offload", "breakeven_mbps"])


def test_slowdown_compares_the_fastest_send_with_local():
    summary = summarise(components(), {"lan": Network(rtt_ms=10, uplink_mbps=8)}, ())
    factors = breakeven_slowdown(summary).set_index("against")

    # The 8-bit split after layer2 (2 + 0.5 + 10 + 0.5 + 1.5 = 14.5 ms) beats the
    # 15 ms offload; local finals take 6 ms in float32 and 3 ms in int8.
    assert set(factors["best_mode"]) == {"split"}
    assert set(factors["best_point"]) == {"layer2"}
    assert factors.loc["local float32", "slowdown"] == 14.5 / 6.0
    assert factors.loc["local int8", "slowdown"] == 14.5 / 3.0
