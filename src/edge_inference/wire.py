"""Messages between a device and a server, and a connection that can wait for several at once.

A message is a small JSON header and a body, each prefixed with its length.
Every request carries an `id` and every reply repeats it. One reader thread per
connection hands each reply to the request that asked, so a device can send a
photo and keep computing, and a reply that arrives late is never mistaken for
the answer to the next frame.

When device and server share a machine, a link can be emulated: each reply is
held back until the link would have delivered it, by the reader thread, so the
device's own work is never paused by it.
"""

import contextlib
import json
import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

_LENGTHS = struct.Struct("!II")


def send_message(connection: socket.socket, header: dict, body: bytes = b"") -> None:
    """Write one message: two lengths, a JSON header, then the body."""
    head = json.dumps(header).encode()
    connection.sendall(_LENGTHS.pack(len(head), len(body)) + head + body)


def _exactly(connection: socket.socket, count: int) -> bytes:
    chunks = []
    while count:
        chunk = connection.recv(count)
        if not chunk:
            raise ConnectionError("the other end closed the connection")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


def receive_message(connection: socket.socket) -> tuple[dict, bytes]:
    """Read one message written by `send_message`."""
    head_length, body_length = _LENGTHS.unpack(_exactly(connection, _LENGTHS.size))
    header = json.loads(_exactly(connection, head_length))
    return header, _exactly(connection, body_length)


@dataclass
class Pending:
    """A request on its way: when it left, and its reply once in."""

    id: int
    sent_at: float
    # Where the reply would land on the emulated link, or 0 for a real link.
    due_at: float = 0.0
    arrived: threading.Event = field(default_factory=threading.Event)
    reply: dict | None = None
    arrived_at: float = float("nan")
    on_arrival: Callable[[Pending], None] | None = None

    def wait(self, timeout: float | None = None) -> dict | None:
        """The reply, or None if it is not in within `timeout` seconds."""
        return self.reply if self.arrived.wait(timeout) else None

    @property
    def round_trip_ms(self) -> float:
        return (self.arrived_at - self.sent_at) * 1000.0


class Connection:
    """The device's end: send requests, and receive replies on a thread of its own.

    `link`, when given, turns a request's body size into the milliseconds an
    emulated link takes to deliver it and bring the answer back; the reply is
    then released no earlier than that after it was sent.
    """

    def __init__(
        self,
        sock: socket.socket,
        *,
        link: Callable[[int], float] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.sock = sock
        self.link = link
        self.clock = clock
        self._next = 0
        self._pending: dict[int, Pending] = {}
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    @classmethod
    def open(cls, host: str, port: int, **settings) -> Connection:
        sock = socket.create_connection((host, port))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return cls(sock, **settings)

    def request(
        self,
        header: dict,
        body: bytes = b"",
        *,
        on_arrival: Callable[[Pending], None] | None = None,
    ) -> Pending:
        """Send a request without waiting; the returned `Pending` fills in when the reply lands."""
        with self._lock:
            self._next += 1
            pending = Pending(self._next, self.clock(), on_arrival=on_arrival)
            if self.link is not None:
                pending.due_at = pending.sent_at + self.link(len(body)) / 1000.0
            self._pending[pending.id] = pending
        with self._send_lock:
            send_message(self.sock, {**header, "id": pending.id}, body)
        return pending

    def _read(self) -> None:
        while True:
            try:
                header, _ = receive_message(self.sock)
            except ConnectionError, OSError:
                return
            with self._lock:
                pending = self._pending.pop(header.get("id"), None)
            if pending is None:
                continue
            wait = pending.due_at - self.clock()
            if wait > 0:
                time.sleep(wait)
            pending.reply = header
            pending.arrived_at = self.clock()
            pending.arrived.set()
            if pending.on_arrival is not None:
                pending.on_arrival(pending)

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()
        self._reader.join(timeout=2)


def serve(
    handle: Callable[[dict, bytes], dict],
    host: str,
    port: int,
    *,
    on_ready: Callable[[int], None] | None = None,
    stop: threading.Event | None = None,
) -> None:
    """Answer requests until `stop` is set, one connection at a time, in order.

    `handle` turns a request into a reply. An `echo` request is answered at
    once with nothing computed, which is how a link is probed. The reply
    repeats the request's id.
    """
    stop = stop or threading.Event()
    with socket.create_server((host, port)) as listener:
        listener.settimeout(0.2)
        if on_ready is not None:
            on_ready(listener.getsockname()[1])
        while not stop.is_set():
            try:
                sock, _ = listener.accept()
            except TimeoutError:
                continue
            with sock:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while not stop.is_set():
                    try:
                        header, body = receive_message(sock)
                    except ConnectionError:
                        break
                    if header.get("kind") == "echo":
                        reply = {"server_ms": 0.0, "received": len(body)}
                    else:
                        reply = handle(header, body)
                    send_message(sock, {**reply, "id": header.get("id")})
