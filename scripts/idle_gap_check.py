"""Does the device run slower when it idles between frames?

Live, the device handles one frame every 100 ms and does nothing in between;
the profile runs images back to back. This times one device variant on the
same test images both ways, in alternating blocks so that drift in clock or
temperature falls on both equally: preparing the input from the decoded
frame, then running the graph, as the live device does.

    uv run python scripts/idle_gap_check.py [--variant mnv4m@160/float32]

Writes results/device_idle_gap_<config>.csv, one row per timed image.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from edge_inference.cli import IMAGENETV2_ROOT, MENU_DIR, MENU_MANIFEST, parse_variant
from edge_inference.menu import timing_sample
from edge_inference.preprocess import load_frame
from edge_inference.serving import open_model

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--variant", default="mnv4m@160/float32")
parser.add_argument("--config", default="laptop-asis-t2")
parser.add_argument("--images", type=int, default=400)
parser.add_argument("--block", type=int, default=50)
parser.add_argument("--gap-ms", type=float, default=90.0)
parser.add_argument("--threads", type=int, default=2)
args = parser.parse_args()

table = timing_sample(pd.read_csv(MENU_MANIFEST), args.images)
frames = [(row.image, load_frame(Path(IMAGENETV2_ROOT) / row.file)) for row in table.itertuples()]
served = open_model(*parse_variant(args.variant, Path(MENU_DIR)), threads=args.threads)
for _, frame in frames[:20]:
    served.infer(served.recipe(frame))

rows = []
for block in range(0, len(frames), args.block):
    for mode in (
        ("back to back", "idle between")
        if block % (2 * args.block)
        else ("idle between", "back to back")
    ):
        for image, frame in frames[block : block + args.block]:
            if mode == "idle between":
                time.sleep(args.gap_ms / 1000.0)
            start = time.perf_counter()
            x = served.recipe(frame)
            prepared = time.perf_counter()
            served.infer(x)
            done = time.perf_counter()
            rows.append({"image": image, "mode": mode,
                         "preprocess_ms": (prepared - start) * 1000.0,
                         "infer_ms": (done - prepared) * 1000.0})  # fmt: skip

found = pd.DataFrame(rows).assign(
    variant=args.variant, config=args.config, threads=args.threads, gap_ms=args.gap_ms
)
found["total_ms"] = found["preprocess_ms"] + found["infer_ms"]
out = Path("results") / f"device_idle_gap_{args.config}.csv"
found.to_csv(out, index=False)
print(found.groupby("mode")[["preprocess_ms", "infer_ms", "total_ms"]].median().round(2))
print("wrote", out)
