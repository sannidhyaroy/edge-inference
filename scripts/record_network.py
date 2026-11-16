"""Record how a real link behaves over time, for replaying in the simulator.

Once a second, over one kept-open HTTPS connection to a public speed-test
endpoint, this times:

* a one-byte upload, which is the round trip;
* an upload the size of a photo (8 KB);
* every fifth second, an upload the size of the smallest feature map (30 KB);
* every tenth second, a larger upload (250 KB), from which the uplink speed
  is estimated as its bytes over the time beyond one round trip.

Every timing is an upload to the same endpoint, so whatever the endpoint adds
is the same in each and cancels when two sizes are compared.

Uploads are random bytes. Each row is written as it is measured, so stopping
early loses nothing. No clock time or network name is recorded, only seconds
since the start and whatever `--label` says. Standard library only.

    python scripts/record_network.py --label shared_wifi --out results/network_trace_shared_wifi.csv
"""

import argparse
import csv
import http.client
import os
import time
from pathlib import Path

HOST = "speed.cloudflare.com"
HEADERS = {"User-Agent": "netprobe/1.0"}
PHOTO, FEATURES, LARGE = 8_000, 30_000, 250_000


class Link:
    def __init__(self) -> None:
        self.connection: http.client.HTTPSConnection | None = None

    def timed(self, size: int) -> float:
        """Milliseconds to upload `size` random bytes and get the reply."""
        if self.connection is None:
            self.connection = http.client.HTTPSConnection(HOST, timeout=10)
            # Open the connection outside the timing: a device keeps one open.
            self.connection.request("GET", "/__down?bytes=0", headers=HEADERS)
            self.connection.getresponse().read()
        start = time.perf_counter()
        self.connection.request(
            "POST",
            "/__up",
            body=os.urandom(size),
            headers={**HEADERS, "Content-Type": "application/octet-stream"},
        )
        response = self.connection.getresponse()
        response.read()
        if response.status != 200:
            raise OSError(f"HTTP {response.status}")
        return (time.perf_counter() - start) * 1000.0

    def reset(self) -> None:
        if self.connection is not None:
            self.connection.close()
        self.connection = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--out", default="results/network_trace.csv")
    parser.add_argument("--label", default="", help="what kind of link this is")
    args = parser.parse_args()

    fields = [
        "second",
        "label",
        "rtt_ms",
        "photo_ms",
        "features_ms",
        "large_ms",
        "uplink_mbps",
        "error",
    ]
    link = Link()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for second in range(args.seconds):
            row = {"second": second, "label": args.label}
            try:
                row["rtt_ms"] = round(link.timed(1), 2)
                row["photo_ms"] = round(link.timed(PHOTO), 2)
                if second % 5 == 0:
                    row["features_ms"] = round(link.timed(FEATURES), 2)
                if second % 10 == 0:
                    large = link.timed(LARGE)
                    row["large_ms"] = round(large, 2)
                    beyond = max(large - row["rtt_ms"], 1e-3)
                    row["uplink_mbps"] = round(LARGE * 8 / beyond / 1000, 2)
            except (OSError, http.client.HTTPException) as error:
                row["error"] = type(error).__name__
                link.reset()
            writer.writerow(row)
            handle.flush()
            time.sleep(max(0.0, began + second + 1 - time.monotonic()))


if __name__ == "__main__":
    main()
