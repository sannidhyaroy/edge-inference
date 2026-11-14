"""Checks for scoring options under a device speed, a network and a deadline.

The table is built by hand with round numbers, so every expected value below
can be worked out on paper. Two images, four options:

* local float32 to the final exit: 10 ms, right on both images;
* local float32 to the first exit: 4 ms, right on image 1 only;
* local int8 to the first exit: 2 ms, right on image 0 only;
* offload: nothing on the device, 1000 bytes sent, 4 ms on the server, right
  on both.

Over the test network (10 ms round trip, 8 Mbps) the 1000 bytes take 1 ms, so
offloading takes 10 + 1 + 4 = 15 ms whatever the device's speed.
"""

import pandas as pd

from edge_inference.decide import (
    best_possible,
    decision_map,
    describe_option,
    for_server,
    frame_ms,
    pick_best,
    score_options,
)
from edge_inference.offload import Network

NETWORK = Network(rtt_ms=10, uplink_mbps=8)


def option(mode, point, precision, payload, server, device_ms, payload_bytes, server_ms, correct):
    """Both images' rows for one option; `correct` lists image 0 then image 1."""
    return [
        {
            "image": image,
            "label": image,
            "mode": mode,
            "point": point,
            "device_precision": precision,
            "payload": payload,
            "server": server,
            "device_ms": device_ms,
            "prepare_ms": 0.0,
            "server_receive_ms": 0.0,
            "server_compute_ms": server_ms,
            "payload_bytes": payload_bytes,
            "confidence": 0.5,
            "correct": right,
        }
        for image, right in enumerate(correct)
    ]


def components():
    return pd.DataFrame(
        option("local", "layer4", "float32", "none", "none", 10.0, 0, 0.0, (True, True))
        + option("local", "layer2", "float32", "none", "none", 4.0, 0, 0.0, (False, True))
        + option("local", "layer2", "int8", "none", "none", 2.0, 0, 0.0, (True, False))
        + option("offload", "input", "none", "jpeg", "srv", 0.0, 1000, 4.0, (True, True))
    )


def useful(scores):
    return scores.set_index("option")["useful"]


def test_a_slower_device_slows_only_the_device():
    frame = components()
    times = frame.assign(total=frame_ms(frame, NETWORK, slowdown=3.0))
    first = times[times["image"] == 0].set_index("mode")["total"]

    assert list(first["local"]) == [30.0, 12.0, 6.0]
    assert first["offload"] == 15.0


def test_useful_means_correct_and_in_time():
    scores = useful(score_options(components(), NETWORK, deadline_ms=12))

    assert scores["local float32 to layer4"] == 1.0  # 10 ms, both right
    assert scores["offload"] == 0.0  # right, but 15 ms
    assert scores["local int8 to layer2"] == 0.5  # in time, right once


def test_a_slower_device_flips_the_best_option():
    frame = components()
    fast = pick_best(score_options(frame, NETWORK, deadline_ms=20, slowdown=1.0))
    slow = pick_best(score_options(frame, NETWORK, deadline_ms=20, slowdown=3.0))

    assert fast["option"] == "local float32 to layer4"
    # At 3x the final exit takes 30 ms and misses; offloading still takes 15.
    assert slow["option"] == "offload"
    assert slow["useful"] == 1.0


def test_ties_go_to_the_faster_option():
    scores = score_options(components(), NETWORK, deadline_ms=100)

    # Local (10 ms) and offload (15 ms) both score 1.0.
    assert pick_best(scores, tie_margin=0.0)["option"] == "local float32 to layer4"
    # A margin of a half lets in the first exit, which is right once in 2 ms.
    assert pick_best(scores, tie_margin=0.5)["option"] == "local int8 to layer2"


def test_best_possible_can_beat_every_single_option():
    frame = components()
    # At 3x with 13 ms, only the two first exits fit: 12 ms and 6 ms. Each is
    # right on a different image, so choosing per frame gets both.
    scores = score_options(frame, NETWORK, deadline_ms=13, slowdown=3.0)

    assert scores["useful"].max() == 0.5
    assert best_possible(frame, NETWORK, deadline_ms=13, slowdown=3.0) == 1.0


def test_only_the_reachable_server_is_offered():
    other = option("offload", "input", "none", "jpeg", "other", 0.0, 1000, 1.0, (True, True))
    frame = pd.concat([components(), pd.DataFrame(other)], ignore_index=True)

    assert set(for_server(frame, "srv")["server"]) == {"none", "srv"}


def test_map_has_one_best_option_per_scenario():
    other = option("offload", "input", "none", "jpeg", "other", 0.0, 1000, 1.0, (True, True))
    frame = pd.concat([components(), pd.DataFrame(other)], ignore_index=True)
    networks = {"lan": NETWORK, "slow": Network(rtt_ms=50, uplink_mbps=1)}
    every, best = decision_map(frame, slowdowns=(1, 3), networks=networks, deadlines=(13, 20))
    scenario = ["server", "slowdown", "network", "deadline_ms"]

    # 2 servers x 2 speeds x 2 networks x 2 deadlines, 4 options in each.
    assert len(best) == 16
    assert len(every) == 16 * 4
    assert (every.groupby(scenario)["best"].sum() == 1).all()
    assert (best["best_possible"] >= best["useful"]).all()

    chosen = best.set_index(scenario)["option"]
    # The other server answers in 1 ms, so at 3x it is in time where srv is not.
    assert chosen[("other", 3, "lan", 13)] == "offload"
    assert chosen[("srv", 3, "lan", 13)] == "local int8 to layer2"


def test_options_are_named_readably():
    assert describe_option("local", "layer4", "int8", "none") == "local int8 to layer4"
    assert describe_option("split", "layer3", "float32", "uint8_zstd") == (
        "split after layer3, uint8_zstd"
    )
    assert describe_option("offload", "input", "none", "jpeg") == "offload"
