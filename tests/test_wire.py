"""Messages, the device's connection, the menu server and link probing, over loopback."""

import socket
import threading
import time

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image

from edge_inference.preprocess import Preprocess, encode_photo
from edge_inference.probe import probe
from edge_inference.serving import MenuServer, open_model
from edge_inference.wire import Connection, receive_message, send_message, serve


def start(handle):
    """Serve `handle` on a free loopback port; return the port and a stop switch."""
    ready, stop = threading.Event(), threading.Event()
    port = []

    def on_ready(bound):
        port.append(bound)
        ready.set()

    thread = threading.Thread(
        target=serve, args=(handle, "127.0.0.1", 0), kwargs={"on_ready": on_ready, "stop": stop}
    )
    thread.start()
    ready.wait(5)
    return port[0], stop, thread


@pytest.fixture
def echo_server():
    port, stop, thread = start(lambda header, body: {"kind": header["kind"], "size": len(body)})
    yield port
    stop.set()
    thread.join()


def test_a_message_round_trips():
    left, right = socket.socketpair()
    send_message(left, {"kind": "photo", "id": 3}, b"abc")
    assert receive_message(right) == ({"kind": "photo", "id": 3}, b"abc")


def test_replies_reach_the_request_that_asked(echo_server):
    connection = Connection.open("127.0.0.1", echo_server)
    first = connection.request({"kind": "photo"}, b"x" * 10)
    second = connection.request({"kind": "photo"}, b"y" * 20)
    assert second.wait(5)["size"] == 20
    assert first.wait(5)["size"] == 10
    assert first.reply["id"] == first.id and second.reply["id"] == second.id
    connection.close()


def test_out_of_order_replies_are_dispatched_by_id():
    # A server that answers its two requests in reverse order.
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]

    def reverse():
        sock, _ = listener.accept()
        requests = [receive_message(sock)[0] for _ in range(2)]
        for header in reversed(requests):
            send_message(sock, {"id": header["id"], "tag": header["tag"]})
        sock.close()

    thread = threading.Thread(target=reverse)
    thread.start()
    connection = Connection.open("127.0.0.1", port)
    a = connection.request({"kind": "photo", "tag": "a"})
    b = connection.request({"kind": "photo", "tag": "b"})
    assert a.wait(5)["tag"] == "a" and b.wait(5)["tag"] == "b"
    thread.join()
    connection.close()
    listener.close()


def test_a_late_reply_never_answers_the_next_request():
    slow = threading.Event()

    def handle(header, body):
        if header.get("tag") == "slow":
            slow.wait(5)
        return {"tag": header.get("tag")}

    port, stop, thread = start(handle)
    connection = Connection.open("127.0.0.1", port)
    late = connection.request({"kind": "photo", "tag": "slow"})
    assert late.wait(0.05) is None
    slow.set()
    following = connection.request({"kind": "photo", "tag": "next"})
    assert following.wait(5)["tag"] == "next"
    assert late.wait(5)["tag"] == "slow"
    connection.close()
    stop.set()
    thread.join()


def test_an_emulated_link_holds_the_reply_back(echo_server):
    connection = Connection.open("127.0.0.1", echo_server, link=lambda size: 60.0)
    pending = connection.request({"kind": "echo"}, b"z")
    assert pending.wait(5) is not None
    assert pending.round_trip_ms >= 60.0
    connection.close()


def test_the_arrival_callback_runs_when_the_reply_lands(echo_server):
    seen = []
    connection = Connection.open("127.0.0.1", echo_server)
    pending = connection.request({"kind": "echo"}, on_arrival=lambda p: seen.append(p.id))
    pending.wait(5)
    time.sleep(0.05)
    assert seen == [pending.id]
    connection.close()


def test_probing_records_the_round_trip_and_the_uplink(echo_server):
    connection = Connection.open("127.0.0.1", echo_server)
    rows = probe(connection, link="loopback", seconds=0.3, tick_ms=50, photo_every=2)
    connection.close()
    assert len(rows) >= 4
    assert rows["rtt_ms"].notna().all() and (rows["rtt_ms"] < 100).all()
    assert rows.loc[rows["tick"] % 2 == 0, "photo_ms"].notna().all()
    assert rows["error"].isna().all()


def tiny_model(path, resolution=16, classes=5):
    """Average the image to three numbers, then score five classes."""
    weights = np.arange(3 * classes, dtype=np.float32).reshape(3, classes) / 10
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"]),
            helper.make_node("MatMul", ["flat", "w"], ["logits"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, resolution, resolution])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, classes])],
        [helper.make_tensor("w", TensorProto.FLOAT, weights.shape, weights.flatten())],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(model, path)


def test_the_menu_server_answers_a_photo_with_the_model_asked_for(tmp_path):
    tiny_model(tmp_path / "tiny.onnx")
    Preprocess(16, 18, "bilinear", (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)).save(tmp_path / "t.json")
    server = MenuServer(
        {"tiny": open_model(tmp_path / "tiny.onnx", tmp_path / "t.json", threads=1)}, "tiny"
    )
    port, stop, thread = start(server.handle)
    frame = Image.fromarray(np.full((30, 40, 3), 200, dtype=np.uint8))
    connection = Connection.open("127.0.0.1", port)
    reply = connection.request({"kind": "photo", "model": "tiny"}, encode_photo(frame, 20, 90))
    reply = reply.wait(5)
    connection.close()
    stop.set()
    thread.join()
    assert reply["model"] == "tiny"
    assert 0 <= reply["predicted"] < 5 and 0 < reply["confidence"] <= 1
    assert reply["server_ms"] >= reply["compute_ms"] >= 0
