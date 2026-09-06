"""Command line entry point.

Subcommands are kept thin. Anything worth reusing lives in a module so it can
be imported by later experiments rather than shelled out to.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from edge_inference.bench import benchmark_model
from edge_inference.config import (
    DATA_DIR,
    IMAGE_SIZE,
    RESULTS_DIR,
    seed_everything,
)
from edge_inference.data import load_split
from edge_inference.models import SUPPORTED_BACKBONES, build_backbone

console = Console()


def default_thread_list() -> str:
    """Thread counts to sweep by default: a constrained device, then this host."""
    available = os.cpu_count() or 1
    return ",".join(str(t) for t in sorted({1, 2, 4, available}))


def parse_thread_list(value: str) -> list[int]:
    """Parse "1,2,4" into a sorted, deduplicated list of thread counts."""
    try:
        threads = sorted({int(part) for part in value.split(",") if part.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a comma separated list of integers, got {value!r}"
        ) from None

    if not threads or threads[0] < 1:
        raise argparse.ArgumentTypeError("thread counts must be positive integers")
    return threads


def directory_size_mb(path: Path) -> float:
    """Total size of a directory tree in mebibytes."""
    total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return total / (1024 * 1024)


def cmd_data_prepare(args: argparse.Namespace) -> int:
    """Download Imagenette if needed and report what landed on disk."""
    root = Path(args.root)
    console.print(f"Preparing Imagenette under [bold]{root}[/bold]")
    console.print("Downloading roughly 94 MB if it is not already present.")

    splits = {split: load_split(split, root=root, download=True) for split in ("train", "val")}

    table = Table(title="Imagenette")
    table.add_column("split")
    table.add_column("images", justify="right")
    table.add_column("classes", justify="right")
    for split, dataset in splits.items():
        table.add_row(split, str(len(dataset)), str(len(dataset.classes)))
    console.print(table)

    console.print(f"On disk: [bold]{directory_size_mb(root):.1f} MiB[/bold]")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Sweep thread counts and record forward pass latency for each."""
    seed_everything()

    # Weights are left randomly initialised. Latency depends on tensor shapes,
    # not on the values in them, so loading pretrained weights would download
    # 45 MB to produce identical timings.
    model = build_backbone(args.backbone, num_classes=args.num_classes)

    console.print(
        f"Benchmarking [bold]{args.backbone}[/bold] "
        f"at {args.image_size}px, batch {args.batch_size}, on {args.device}"
    )

    rows = []
    for threads in args.threads:
        row = benchmark_model(
            model,
            threads=threads,
            warmup=args.warmup,
            runs=args.runs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=args.device,
        )
        row["backbone"] = args.backbone
        rows.append(row)
        console.print(
            f"  {threads:>3} thread(s): "
            f"median {row['median_ms']:.2f} ms, p95 {row['p95_ms']:.2f} ms"
        )

    frame = pd.DataFrame(rows)

    table = Table(title=f"{args.backbone} forward pass latency, batch {args.batch_size}")
    for column in ("threads", "effective", "median ms", "p95 ms", "mean ms", "std ms"):
        table.add_column(column, justify="right")
    for row in rows:
        table.add_row(
            str(row["threads_requested"]),
            str(row["threads_effective"]),
            f"{row['median_ms']:.2f}",
            f"{row['p95_ms']:.2f}",
            f"{row['mean_ms']:.2f}",
            f"{row['std_ms']:.2f}",
        )
    console.print(table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    console.print(f"Wrote [bold]{out}[/bold]")

    # A thread cap that is silently ignored would make every constrained-core
    # number in the report meaningless, so say so loudly rather than filing it.
    requested = [row["threads_requested"] for row in rows]
    effective = [row["threads_effective"] for row in rows]
    if requested != effective:
        console.print(
            "[bold red]Warning:[/bold red] requested and effective thread counts "
            "differ. Set OMP_NUM_THREADS before launching, or run each thread "
            "count in a separate process."
        )
    elif len(rows) > 1 and rows[0]["median_ms"] <= rows[-1]["median_ms"]:
        console.print(
            "[bold yellow]Warning:[/bold yellow] the lowest thread count was not "
            "slower than the highest. Thread pinning may not be taking effect."
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge",
        description="Early exit and INT8 quantization profiling for edge inference.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("data", help="dataset management")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    prepare = data_sub.add_parser("prepare", help="download and verify Imagenette")
    prepare.add_argument("--root", default=str(DATA_DIR), help="dataset directory")
    prepare.set_defaults(func=cmd_data_prepare)

    bench = subparsers.add_parser("bench", help="measure forward pass latency")
    bench.add_argument("--backbone", default="resnet18", choices=SUPPORTED_BACKBONES)
    bench.add_argument("--num-classes", type=int, default=None, help="replace the classifier head")
    bench.add_argument(
        "--threads",
        type=parse_thread_list,
        default=parse_thread_list(default_thread_list()),
        help=f"comma separated thread counts (default: {default_thread_list()})",
    )
    bench.add_argument("--runs", type=int, default=50, help="timed runs per thread count")
    bench.add_argument("--warmup", type=int, default=10, help="discarded runs before timing")
    bench.add_argument("--batch-size", type=int, default=1)
    bench.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    bench.add_argument(
        "--device",
        default="cpu",
        help="cpu, mps, or cuda. Defaults to cpu because that is the measurement target",
    )
    bench.add_argument("--out", default=str(RESULTS_DIR / "latency_backbone.csv"))
    bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


# Guarded so that dataloader worker processes, which start by re-importing the
# entry module on Windows and macOS, do not re-run the command.
if __name__ == "__main__":
    raise SystemExit(main())
