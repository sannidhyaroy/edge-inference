"""Command line entry point.

Subcommands are kept thin. Anything worth reusing lives in a module so it can
be imported by later experiments rather than shelled out to.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from edge_inference.bench import benchmark_model, benchmark_session, resolve_device
from edge_inference.config import (
    CHECKPOINT_DIR,
    DATA_DIR,
    IMAGE_SIZE,
    NUM_CLASSES,
    RESULTS_DIR,
    seed_everything,
)
from edge_inference.data import build_dataloader, load_split
from edge_inference.export import (
    DEFAULT_OPSET,
    build_session,
    evaluate_session,
    export_onnx,
    verify_parity,
)
from edge_inference.models import SUPPORTED_BACKBONES, build_backbone
from edge_inference.profiling import profile_model
from edge_inference.quantization import DataLoaderCalibrationReader, quantize_onnx
from edge_inference.training import fine_tune

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


def cmd_train(args: argparse.Namespace) -> int:
    """Fine-tune a pretrained backbone on Imagenette and save the weights."""
    seed_everything()
    device = resolve_device(args.device)

    train_dataset = load_split("train", root=Path(args.root))
    val_dataset = load_split("val", root=Path(args.root))

    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_backbone(args.backbone, num_classes=NUM_CLASSES, pretrained=True)

    console.print(
        f"Fine-tuning [bold]{args.backbone}[/bold] on {len(train_dataset)} images, "
        f"validating on {len(val_dataset)}, for {args.epochs} epoch(s) on {device}"
    )

    history = fine_tune(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    checkpoint = Path(args.out)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Only the weights are saved, not the whole model object. A state dict is a
    # plain mapping of tensors, so it loads without needing the original class
    # definition to be importable and survives refactoring of the code.
    torch.save(model.state_dict(), checkpoint)
    console.print(f"Saved weights to [bold]{checkpoint}[/bold]")

    frame = pd.DataFrame(history)
    frame["backbone"] = args.backbone
    frame["device"] = device
    frame["batch_size"] = args.batch_size

    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(history_path, index=False)
    console.print(f"Wrote [bold]{history_path}[/bold]")

    # Report the final epoch, because that is the model just written to disk.
    # Quoting the best epoch instead would describe weights that were never
    # saved, and every downstream measurement would be against a different
    # model than the headline number.
    #
    # The best epoch is deliberately not saved either. Imagenette ships only
    # train and val splits, so val doubles as the test set. Keeping whichever
    # epoch scored highest on it would mean choosing a model using the same
    # data the accuracy is reported on, and the maximum of several noisy
    # measurements is biased upward. A fixed epoch budget with the last epoch
    # reported makes no such choice.
    final = history[-1]
    best = max(history, key=lambda row: row["val_accuracy"])
    console.print(
        f"Final validation accuracy: [bold]{final['val_accuracy'] * 100:.2f}%[/bold] "
        f"after {int(final['epoch'])} epoch(s), which is what was saved"
    )
    if best["epoch"] != final["epoch"]:
        console.print(
            f"  (epoch {int(best['epoch'])} peaked higher at "
            f"{best['val_accuracy'] * 100:.2f}%, neither saved nor reported)"
        )
    return 0


def load_checkpoint(model: torch.nn.Module, path: Path | None) -> torch.nn.Module:
    """Load weights into `model` if a checkpoint exists, warning loudly if not."""
    if path is None:
        return model
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        console.print(f"Loaded weights from [bold]{path}[/bold]")
    else:
        console.print(f"[yellow]No checkpoint at {path}, using untrained weights.[/yellow]")
    return model


def cmd_export(args: argparse.Namespace) -> int:
    """Export a trained model to ONNX and verify it matches PyTorch."""
    seed_everything()

    model = build_backbone(args.backbone, num_classes=NUM_CLASSES)
    load_checkpoint(model, Path(args.checkpoint) if args.checkpoint else None)

    out = Path(args.out)
    export_onnx(model, out, image_size=args.image_size, opset=args.opset)
    size_mib = out.stat().st_size / (1024 * 1024)
    console.print(f"Exported to [bold]{out}[/bold] ({size_mib:.2f} MiB, opset {args.opset})")

    row = verify_parity(
        model, out, image_size=args.image_size, samples=args.samples, atol=args.atol
    )

    table = Table(title="PyTorch against ONNX Runtime")
    table.add_column("check")
    table.add_column("value", justify="right")
    table.add_row("samples", str(int(row["samples"])))
    table.add_row("max absolute error", f"{row['max_abs_error']:.3e}")
    table.add_row("mean absolute error", f"{row['mean_abs_error']:.3e}")
    table.add_row("tolerance", f"{row['tolerance']:.0e}")
    table.add_row("prediction agreement", f"{row['prediction_agreement'] * 100:.1f}%")
    console.print(table)

    row["backbone"] = args.backbone
    row["opset"] = args.opset
    row["onnx_mib"] = size_mib

    results = Path(args.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(results, index=False)
    console.print(f"Wrote [bold]{results}[/bold]")

    # A failed parity check is an error, not a warning. Every latency and
    # accuracy number measured through this file afterwards would describe a
    # model that was never evaluated.
    if not row["predictions_match"]:
        console.print(
            "[bold red]Parity failed:[/bold red] the exported graph predicts a "
            "different class from PyTorch on at least one input. Do not measure "
            "against this file."
        )
        return 1

    if not row["within_tolerance"]:
        console.print(
            f"[bold yellow]Note:[/bold yellow] max error {row['max_abs_error']:.3e} "
            f"exceeds the {row['tolerance']:.0e} tolerance, but every prediction "
            "still agrees. Worth understanding before relying on it."
        )

    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    """Quantize an exported model to INT8 and measure what it cost."""
    seed_everything()

    source = Path(args.source)
    if not source.exists():
        console.print(f"[bold red]No ONNX model at {source}.[/bold red] Run `edge export` first.")
        return 1

    # Training images with evaluation preprocessing. Calibration measures the
    # range of values inference will actually see, and random crops would
    # measure a distribution that never occurs at inference time.
    calibration_dataset = load_split("train", root=Path(args.root), augment=False)
    calibration_loader = build_dataloader(
        calibration_dataset, batch_size=args.batch_size, shuffle=True
    )
    reader = DataLoaderCalibrationReader(calibration_loader, limit=args.calibration_images)

    console.print(
        f"Calibrating on [bold]{args.calibration_images}[/bold] training images, "
        f"then quantizing to INT8"
    )
    destination = Path(args.out)
    quantize_onnx(source, destination, reader, per_channel=not args.per_tensor)

    val_dataset = load_split("val", root=Path(args.root))
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    rows = []
    for label, path in (("float32", source), ("int8", destination)):
        size_mib = path.stat().st_size / (1024 * 1024)
        session = build_session(path, threads=args.threads)
        console.print(f"Evaluating {label} on {len(val_dataset)} images")
        metrics = evaluate_session(session, val_loader)

        timed = benchmark_session(
            session,
            threads=args.threads,
            warmup=args.warmup,
            runs=args.runs,
            image_size=args.image_size,
        )
        rows.append(
            {
                "precision": label,
                "accuracy": metrics["accuracy"],
                "size_mib": size_mib,
                **timed,
            }
        )

    float_row, int8_row = rows
    accuracy_drop = (float_row["accuracy"] - int8_row["accuracy"]) * 100
    size_ratio = float_row["size_mib"] / int8_row["size_mib"]
    speedup = float_row["median_ms"] / int8_row["median_ms"]

    table = Table(title=f"float32 against INT8, {args.threads} thread(s)")
    for column in ("precision", "accuracy", "size MiB", "median ms", "p95 ms"):
        table.add_column(column, justify="right")
    for row in rows:
        table.add_row(
            row["precision"],
            f"{row['accuracy'] * 100:.2f}%",
            f"{row['size_mib']:.2f}",
            f"{row['median_ms']:.2f}",
            f"{row['p95_ms']:.2f}",
        )
    console.print(table)

    console.print(
        f"Accuracy drop [bold]{accuracy_drop:+.2f}[/bold] points, "
        f"size [bold]{size_ratio:.2f}x[/bold] smaller, "
        f"latency [bold]{speedup:.2f}x[/bold]"
    )
    if speedup < 1.2:
        console.print(
            "[yellow]Modest speedup is expected on a CPU without VNNI.[/yellow] "
            "The size and accuracy results are hardware-independent and still hold."
        )

    frame = pd.DataFrame(rows)
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    console.print(f"Wrote [bold]{out}[/bold]")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """Report parameters, operations, and checkpoint size for a model."""
    seed_everything()

    model = build_backbone(args.backbone, num_classes=NUM_CLASSES)

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint is not None and checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        console.print(f"Loaded weights from [bold]{checkpoint}[/bold]")
    elif checkpoint is not None:
        console.print(
            f"[yellow]No checkpoint at {checkpoint}, profiling untrained weights.[/yellow]"
        )

    row = profile_model(model, image_size=args.image_size, checkpoint=checkpoint)
    row["backbone"] = args.backbone

    table = Table(title=f"{args.backbone} cost at {args.image_size}px, batch 1")
    table.add_column("measure")
    table.add_column("value", justify="right")
    table.add_row("parameters", f"{int(row['parameters_total']):,}")
    table.add_row("MFLOPs", f"{row['mflops']:.1f}")
    table.add_row("MMACs", f"{row['mmacs']:.1f}")
    if "checkpoint_mib" in row:
        table.add_row("checkpoint MiB", f"{row['checkpoint_mib']:.2f}")
    console.print(table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)
    console.print(f"Wrote [bold]{out}[/bold]")
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

    train = subparsers.add_parser("train", help="fine-tune a backbone on Imagenette")
    train.add_argument("--backbone", default="resnet18", choices=SUPPORTED_BACKBONES)
    train.add_argument("--root", default=str(DATA_DIR), help="dataset directory")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=0.01, help="initial learning rate")
    train.add_argument(
        "--device",
        default="cpu",
        help="cpu, cuda, or mps. Training may run anywhere; latency measurement may not",
    )
    train.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help=(
            "background data loading processes. On CPU these compete with the "
            "compute threads, so more is not always better"
        ),
    )
    train.add_argument("--out", default=str(CHECKPOINT_DIR / "resnet18_imagenette.pt"))
    train.add_argument("--history", default=str(RESULTS_DIR / "training_history.csv"))
    train.set_defaults(func=cmd_train)

    export = subparsers.add_parser("export", help="export to ONNX and verify against PyTorch")
    export.add_argument("--backbone", default="resnet18", choices=SUPPORTED_BACKBONES)
    export.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "resnet18_imagenette.pt"))
    export.add_argument("--out", default=str(CHECKPOINT_DIR / "resnet18_imagenette.onnx"))
    export.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    export.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version")
    export.add_argument("--samples", type=int, default=8, help="inputs compared for parity")
    export.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="absolute tolerance on raw outputs; predictions must match regardless",
    )
    export.add_argument("--results", default=str(RESULTS_DIR / "onnx_parity.csv"))
    export.set_defaults(func=cmd_export)

    quantize = subparsers.add_parser(
        "quantize", help="quantize to INT8 and compare against float32"
    )
    quantize.add_argument("--root", default=str(DATA_DIR), help="dataset directory")
    quantize.add_argument("--source", default=str(CHECKPOINT_DIR / "resnet18_imagenette.onnx"))
    quantize.add_argument("--out", default=str(CHECKPOINT_DIR / "resnet18_imagenette.int8.onnx"))
    quantize.add_argument(
        "--calibration-images",
        type=int,
        default=256,
        help="training images used to measure activation ranges",
    )
    quantize.add_argument(
        "--per-tensor",
        action="store_true",
        help="one scale per layer instead of per convolution filter, usually less accurate",
    )
    quantize.add_argument("--batch-size", type=int, default=32)
    quantize.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    quantize.add_argument("--threads", type=int, default=6, help="threads for both timed models")
    quantize.add_argument("--runs", type=int, default=50)
    quantize.add_argument("--warmup", type=int, default=10)
    quantize.add_argument("--results", default=str(RESULTS_DIR / "quantization.csv"))
    quantize.set_defaults(func=cmd_quantize)

    profile = subparsers.add_parser("profile", help="report parameters, operations, and size")
    profile.add_argument("--backbone", default="resnet18", choices=SUPPORTED_BACKBONES)
    profile.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    profile.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT_DIR / "resnet18_imagenette.pt"),
        help="weights to load and measure on disk; operations do not depend on them",
    )
    profile.add_argument("--out", default=str(RESULTS_DIR / "model_cost.csv"))
    profile.set_defaults(func=cmd_profile)

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


# Guarded so that dataloader worker processes do not re-run the command. Those
# workers re-import the entry module on every platform now: spawn on Windows and
# macOS, and forkserver on Linux since Python 3.14. Without this guard they
# recurse and die with a BrokenPipeError.
if __name__ == "__main__":
    raise SystemExit(main())
