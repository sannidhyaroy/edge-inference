"""Command line entry point.

Subcommands are kept thin. Anything worth reusing lives in a module so it can
be imported by later experiments rather than shelled out to.
"""

import argparse
import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from edge_inference.bandit import Bandit, variants
from edge_inference.bench import (
    benchmark_model,
    benchmark_session,
    cpu_model,
    cpu_state,
    machine_model,
    power_source,
    resolve_device,
    set_thread_count,
    wait_for_cooldown,
)
from edge_inference.config import (
    CHECKPOINT_DIR,
    DATA_DIR,
    IMAGE_SIZE,
    NUM_CLASSES,
    RESULTS_DIR,
    SEED,
    seed_everything,
)
from edge_inference.data import build_dataloader, image_files, load_split
from edge_inference.decide import DEADLINES_MS, SLOWDOWNS, TIE_MARGIN, decision_map
from edge_inference.exit_analysis import default_thresholds, exit_vectors, sweep_thresholds
from edge_inference.exit_profiler import profile_exit_sessions, profile_exits
from edge_inference.export import (
    DEFAULT_OPSET,
    build_session,
    build_stage_sessions,
    evaluate_session,
    evaluate_stages,
    export_exit_stages,
    export_onnx,
    sized_base,
    stage_paths,
    verify_parity,
    verify_stage_parity,
)
from edge_inference.manifest import IMAGENET_TRAIN, IMAGENET_VAL, accuracy_by_origin, manifest
from edge_inference.models import (
    EXIT_NAMES,
    SUPPORTED_BACKBONES,
    build_backbone,
    build_early_exit,
)
from edge_inference.offload import (
    NETWORKS,
    breakeven_mbps,
    breakeven_slowdown,
    load_components,
    summarise,
)
from edge_inference.policies import OFFLOAD, Always, Lookup
from edge_inference.profiling import operations_per_exit, profile_model
from edge_inference.quantization import (
    INT8_RECIPES,
    DataLoaderCalibrationReader,
    calibration_batches,
    quantize_exit_stages,
    quantize_onnx,
)
from edge_inference.runtime import Device, LiveRun, Responder, compare, connect, serve
from edge_inference.scoreboard import (
    FAIR_TARGET,
    HEDGE,
    HEDGED_TARGET,
    OURS,
    OURS_HEDGED,
    Setup,
    grid,
    run_setup,
    run_sweep,
)
from edge_inference.server import OnnxServer, TorchServer, profile_server
from edge_inference.simulate import (
    AT_EXIT,
    AT_ONCE,
    HORIZON_MS,
    NEVER,
    Measurements,
    Plan,
    Scenario,
    local_from_profile,
    simulate,
    split_images,
)
from edge_inference.split import profile_splits
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


def parse_weight_list(value: str) -> list[float]:
    """Parse "1,1,1" into a list of non-negative loss weights."""
    try:
        weights = [float(part) for part in value.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a comma separated list of numbers, got {value!r}"
        ) from None

    if not weights or any(w < 0 for w in weights) or sum(weights) == 0:
        raise argparse.ArgumentTypeError("weights must be non-negative and not all zero")
    return weights


def parse_number_list(value: str) -> list[float]:
    """Parse "1,2,4" into a list of positive numbers."""
    try:
        numbers = [float(part) for part in value.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a comma separated list of numbers, got {value!r}"
        ) from None

    if not numbers or any(n <= 0 for n in numbers):
        raise argparse.ArgumentTypeError("numbers must be positive")
    return numbers


def cool_down(target_c: float | None) -> None:
    """Wait for the CPU to cool to `target_c` before a timed run, if one was given."""
    if target_c is None:
        return
    start = cpu_state()["cpu_temp_c"]
    if start is None:
        console.print("[yellow]No CPU temperature is readable here; not waiting.[/yellow]")
        return
    if start <= target_c:
        return
    console.print(f"Waiting for the CPU to cool from {start:.0f} C to {target_c:.0f} C")
    reached = wait_for_cooldown(target_c)
    if reached is not None and reached > target_c:
        console.print(f"[yellow]Still {reached:.0f} C after 10 minutes; timing anyway.[/yellow]")


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


def cmd_data_origin(args: argparse.Namespace) -> int:
    """Where each test image came from in ImageNet, and accuracy by origin.

    The backbone is ImageNet-pretrained and most Imagenette test images are
    ImageNet training images. If the pretraining had inflated accuracy, the
    few images ImageNet kept for validation would score lower than the rest.
    """
    root = Path(args.root)
    dataset = load_split("val", root=root)
    images = manifest(image_files(dataset), [label for _, label in dataset._samples], root=root)
    out = Path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    images.to_csv(out, index=False)
    counts = images["origin"].value_counts()
    console.print(
        f"Wrote [bold]{out}[/bold]: {len(images)} images, "
        + ", ".join(f"{count} from {where}" for where, count in counts.items())
    )

    results = Path(args.results)
    tables = []
    for profile in args.profiles:
        frame = pd.read_csv(profile)
        stem = Path(profile).name.removesuffix(".gz").removesuffix(".csv")
        stem = stem.removeprefix("exit_profile_")
        tables.append(accuracy_by_origin(frame, images).assign(profile=stem))
    if not tables:
        return 0
    table = pd.concat(tables, ignore_index=True)
    by_origin = results / "accuracy_by_origin.csv"
    table.to_csv(by_origin, index=False)

    view = Table(title="Accuracy by ImageNet origin, 95% interval")
    for column in ("profile", "exit", "ImageNet train", "ImageNet val", "val minus train"):
        view.add_column(column, justify="left" if column == "profile" else "right")
    for (stem, exit_name), group in table.groupby(["profile", "exit_name"], sort=False):
        rows = group.set_index("origin")
        cells = [
            f"{rows.at[where, 'accuracy'] * 100:.2f} "
            f"({rows.at[where, 'ci_low'] * 100:.1f} to {rows.at[where, 'ci_high'] * 100:.1f})"
            for where in (IMAGENET_TRAIN, IMAGENET_VAL)
        ]
        gap = (rows.at[IMAGENET_VAL, "accuracy"] - rows.at[IMAGENET_TRAIN, "accuracy"]) * 100
        view.add_row(stem, exit_name, *cells, f"{gap:+.2f}")
    console.print(view)
    console.print(f"Wrote [bold]{by_origin}[/bold]")
    return 0


IMAGENETV2_ROOT = DATA_DIR / "imagenetv2" / "imagenetv2-matched-frequency-format-val"
MENU_DIR = CHECKPOINT_DIR / "menu"
ONNXRUNTIME, TORCH_CUDA, TORCH_CPU = "onnxruntime", "torch-cuda", "torch-cpu"
SERVER_BACKENDS = [ONNXRUNTIME, TORCH_CUDA, TORCH_CPU]
MENU_MANIFEST = Path("manifests/imagenetv2_mf.csv")


def cmd_menu_manifest(args: argparse.Namespace) -> int:
    """Fix ImageNetV2's tuning, test and calibration images once, in a manifest."""
    from edge_inference.menu import check_manifest, imagenetv2_manifest

    root = Path(args.root)
    table = imagenetv2_manifest(root)
    check_manifest(table)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    console.print(
        f"Wrote [bold]{out}[/bold]: {len(table)} images in {table['label'].nunique()} classes, "
        + ", ".join(f"{n} {split}" for split, n in table["split"].value_counts().items())
        + f", {int(table['calibration'].sum())} for calibration"
    )
    return 0


def cmd_menu_export(args: argparse.Namespace) -> int:
    """Export each model at each resolution, with its recipe and an INT8 copy."""
    from edge_inference.menu import check_manifest
    from edge_inference.zoo import build, export_variant

    table = pd.read_csv(args.manifest)
    root = Path(args.root)
    check_manifest(table, root)
    rng = np.random.default_rng(SEED)
    pool = table.loc[table["calibration"], "file"].to_numpy()
    calibration = [root / f for f in rng.choice(pool, size=args.calibration_images, replace=False)]
    for name in args.models:
        model = build(name)
        for resolution in args.resolutions:
            paths = export_variant(
                name, resolution, Path(args.out), calibration=calibration, model=model,
                int8_recipe=args.int8_recipe, requantize=args.requantize,
            )  # fmt: skip
            console.print(f"{name}@{resolution}: " + ", ".join(p.name for p in paths.values()))
    return 0


def cmd_menu_answers(args: argparse.Namespace) -> int:
    """Record what every exported variant answers for every image, per model."""
    from concurrent.futures import ProcessPoolExecutor

    from edge_inference.menu import answer_variant
    from edge_inference.zoo import variant_path

    table = pd.read_csv(args.manifest)
    root, directory = Path(args.root), Path(args.dir)
    jobs = []
    for name in args.models:
        for resolution in args.resolutions:
            sessions = {
                precision: variant_path(directory, name, resolution, precision)
                for precision in ("float32", "int8")
                if variant_path(directory, name, resolution, precision).exists()
            }
            if sessions:
                jobs.append((name, resolution, sessions))

    with ProcessPoolExecutor(args.workers) as pool:
        work = [
            (directory, name, res, sessions, table, root, args.threads)
            for name, res, sessions in jobs
        ]
        found = pd.concat(pool.map(answer_variant, work), ignore_index=True)
    results = Path(args.results)
    for name, frame in found.groupby("model"):
        out = results / f"menu_answers_{name}.csv.gz"
        frame.to_csv(out, index=False)
        console.print(f"Wrote [bold]{out}[/bold]")
    summary = (
        found.groupby(["model", "resolution", "precision", "split"])["correct"]
        .mean()
        .unstack("split")
        .mul(100)
        .round(2)
    )
    console.print(summary.to_string())
    return 0


def parse_variant(key: str, directory: Path) -> tuple[Path, Path]:
    """The graph and recipe for a variant written as model@resolution/precision."""
    from edge_inference.zoo import recipe_path, variant_path

    model, _, rest = key.partition("@")
    resolution, _, precision = rest.partition("/")
    resolution = int(resolution)
    return (
        variant_path(directory, model, resolution, precision or "float32"),
        recipe_path(directory, model, resolution),
    )


def parse_photo(key: str) -> tuple[int, int]:
    """A photo variant written as jpeg<shorter side>q<quality>, e.g. jpeg256q85."""
    shorter, _, quality = key.removeprefix("jpeg").partition("q")
    return int(shorter), int(quality)


def cmd_menu_photo_answers(args: argparse.Namespace) -> int:
    """Record what each server model answers for every sent photo, per model."""
    from concurrent.futures import ProcessPoolExecutor

    from edge_inference.menu import answer_photos
    from edge_inference.zoo import variant_path

    directory, payloads = Path(args.dir), Path(args.payloads)
    jobs = []
    for key in args.models:
        name, _, resolution = key.partition("@")
        resolution = int(resolution or 224)
        sessions = {
            precision: variant_path(directory, name, resolution, precision)
            for precision in args.precisions
            if variant_path(directory, name, resolution, precision).exists()
        }
        for photo in args.photos:
            index = pd.read_csv(payloads / f"{photo}.csv")
            jobs.append((directory, name, resolution, sessions, payloads / photo, index,
                         args.threads))  # fmt: skip
    with ProcessPoolExecutor(args.workers) as pool:
        found = pd.concat(pool.map(answer_photos, jobs), ignore_index=True)
    results = Path(args.results)
    for name, frame in found.groupby("model"):
        out = results / f"menu_photo_answers_{name}.csv.gz"
        frame.to_csv(out, index=False)
        console.print(f"Wrote [bold]{out}[/bold]")
    summary = (
        found.groupby(["model", "photo", "precision", "split"])["correct"]
        .mean()
        .unstack("split")
        .mul(100)
        .round(2)
    )
    console.print(summary.to_string())
    return 0


def open_server_model(key: str, directory: Path, *, threads: int, backend: str):
    """A server model from its ONNX graph on the CPU, or from its weights in PyTorch.

    PyTorch on the CPU is for rehearsing a GPU session on a machine without one.
    """
    if backend in (TORCH_CUDA, TORCH_CPU):
        from edge_inference.serving_torch import open_torch_model

        return open_torch_model(key, backend.removeprefix("torch-"))
    from edge_inference.serving import open_model

    return open_model(*parse_variant(key, directory), threads=threads)


def server_hardware(backend: str) -> dict[str, str]:
    """What ran a server model: the processor always, the GPU when there is one."""
    found = {"backend": backend, "machine": machine_model(), "cpu": cpu_model(),
             "power_source": power_source()}  # fmt: skip
    if backend == TORCH_CUDA:
        from edge_inference.serving_torch import device_name

        found["gpu"] = device_name("cuda")
    return found


def cmd_menu_serve(args: argparse.Namespace) -> int:
    """Answer photos with the menu's server models until interrupted."""
    from edge_inference.serving import MenuServer
    from edge_inference.wire import serve as serve_requests

    directory = Path(args.dir)
    models = {
        key: open_server_model(key, directory, threads=args.threads, backend=args.backend)
        for key in args.models
    }
    server = MenuServer(models, args.models[0])
    console.print(
        f"Serving {', '.join(args.models)} on {args.host}:{args.port} "
        f"with {args.backend} at {args.threads} threads",
        highlight=False,
    )
    try:
        serve_requests(server.handle, args.host, args.port)
    except KeyboardInterrupt:
        console.print("Stopped")
    return 0


def cmd_menu_scale_server(args: argparse.Namespace) -> int:
    """Time a server by another's timings, scaled by how much slower it ran live."""
    from edge_inference.menu_table import scaled_server, server_scale

    results = Path(args.results)
    files = sorted(results.glob(f"menu_server_{args.source}_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no timings for server {args.source} in {results}")
    timing = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    live = pd.concat(
        [pd.read_csv(results / f"menu_live_{link}_{args.label}.csv.gz") for link in args.links],
        ignore_index=True,
    )
    factor, count = server_scale(timing, live)
    console.print(f"{args.name} ran {factor:.3f}x the time of {args.source} on {count} photos")
    for (model, photo), frame in scaled_server(timing, factor, args.name).groupby(
        ["model", "photo"]
    ):
        out = results / f"menu_server_{args.name}_{model.replace('/', '_')}_{photo}.csv.gz"
        frame.to_csv(out, index=False)
        console.print(f"Wrote [bold]{out}[/bold]")
    return 0


def cmd_menu_scoreboard(args: argparse.Namespace) -> int:
    """Score every rule and yardstick on one recorded link, for one device setup."""
    from edge_inference.menu import TEST, TUNE
    from edge_inference.menu_table import menu_measurements
    from edge_inference.recorded import RecordedSetup, run_recorded
    from edge_inference.traces import Trace

    results = Path(args.results)

    def stacked(pattern: str) -> pd.DataFrame:
        files = sorted(results.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no files match {results / pattern}")
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    manifest = pd.read_csv(args.manifest)
    servers = {}
    for server in args.servers:
        timing = stacked(f"menu_server_{server}_*.csv.gz")
        timing = timing[timing["photo"].isin(args.photos)]
        if args.server_models:
            timing = timing[timing["model"].isin(args.server_models)]
        servers[server] = timing
    payloads = {photo: pd.read_csv(Path(args.payloads) / f"{photo}.csv") for photo in args.photos}
    table = menu_measurements(
        manifest[["image", "label"]],
        stacked("menu_answers_*.csv.gz"),
        pd.read_csv(results / f"menu_device_{args.config}.csv.gz"),
        pd.read_csv(results / f"menu_prepare_{args.config}.csv.gz"),
        stacked("menu_photo_answers_*.csv.gz"),
        servers,
        payloads,
        link=args.link,
        variants=args.variants or None,
    )
    split = manifest.set_index("image").loc[table.images, "split"].to_numpy()
    tuning, test = table.subset(split == TUNE), table.subset(split == TEST)
    if args.recording:
        # A link recorded once: tune on its first half, test on its second.
        halves = Trace.from_seconds(pd.read_csv(args.recording), args.link).halves()
        traces = dict(zip((args.tune_label, args.test_label), halves, strict=True))
    else:
        traces = {
            label: Trace.from_probe(pd.read_csv(results / f"link_{args.link}_{label}.csv"))
            for label in (args.tune_label, args.test_label)
        }
    console.print(
        f"{args.config} over {args.link}: {len(table.local)} device variants, "
        f"{len(table.routes)} photo routes, {len(tuning)} tuning and {len(test)} test images"
    )
    for deadline in args.deadlines:
        setup = RecordedSetup(
            deadline_ms=deadline, period_ms=args.period, horizon_ms=args.horizon,
            knowledge=args.knowledge, cancel=args.cancel, seed=args.seed,
        )  # fmt: skip
        board = run_recorded(
            tuning, test, traces[args.tune_label], traces[args.test_label], setup
        ).assign(config=args.config, servers=",".join(args.servers))
        out = results / f"menu_scoreboard_{args.config}_{args.link}_{setup.tag()}.csv"
        board.to_csv(out, index=False)
        shown = board[["rule", "useful", "record", "device_busy_ms", "sent_share", "plans"]]
        console.print(f"[bold]{deadline:g} ms[/bold]")
        console.print(shown.round(4).to_string(index=False, max_colwidth=90))
        console.print(f"Wrote [bold]{out}[/bold]")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Record a link to a server over the project's own protocol."""
    from edge_inference.probe import probe
    from edge_inference.wire import Connection

    host, _, port = args.server.rpartition(":")
    connection = Connection.open(host, int(port))
    console.print(f"Probing {args.server} as [bold]{args.link}[/bold] for {args.seconds:g} s")
    rows = probe(connection, link=args.link, seconds=args.seconds, tick_ms=args.tick_ms)
    connection.close()
    rows["label"] = args.label
    out = Path(args.out or RESULTS_DIR / f"link_{args.link}_{args.label}.csv")
    rows.to_csv(out, index=False)
    summary = rows[["rtt_ms", "uplink_mbps", "photo_ms"]].describe(percentiles=[0.5, 0.95])
    console.print(summary.loc[["count", "50%", "95%"]].round(2).to_string())
    console.print(f"Wrote [bold]{out}[/bold] ({int(rows['error'].notna().sum())} errors)")
    return 0


def cmd_menu_payloads(args: argparse.Namespace) -> int:
    """Encode each image as every photo variant, once, with a hash a server can check."""
    import hashlib

    from edge_inference.preprocess import encode_photo, load_frame

    table = pd.read_csv(args.manifest)
    if args.split:
        table = table[table["split"] == args.split]
    root, out = Path(args.root), Path(args.out)
    for key in args.photos:
        shorter, quality = parse_photo(key)
        folder = out / key
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for row in table.itertuples():
            data = encode_photo(load_frame(root / row.file), shorter, quality)
            (folder / f"{row.image}.jpg").write_bytes(data)
            rows.append(
                {
                    "image": row.image,
                    "label": row.label,
                    "split": row.split,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        index = pd.DataFrame(rows)
        index.to_csv(out / f"{key}.csv", index=False)
        console.print(
            f"{key}: {len(index)} photos, median {index['bytes'].median() / 1000:.1f} kB, "
            f"in [bold]{folder}[/bold]"
        )
    return 0


def cmd_menu_profile_server(args: argparse.Namespace) -> int:
    """Time a server model on every photo of a payload set: decode and prepare, then run."""
    import hashlib
    import time

    from edge_inference.menu import timing_sample
    from edge_inference.preprocess import decode_photo
    from edge_inference.serving import answer

    payloads = Path(args.payloads)
    index = timing_sample(pd.read_csv(payloads.parent / f"{payloads.name}.csv"), args.images)
    directory, results = Path(args.dir), Path(args.results)
    hardware = server_hardware(args.backend)
    for key in args.models:
        served = open_server_model(key, directory, threads=args.threads, backend=args.backend)
        cool_down(args.cool_to)
        warm = (payloads / f"{index['image'].iloc[0]}.jpg").read_bytes()
        for _ in range(args.warmup):
            served.infer(served.recipe(decode_photo(warm)))
        rows = []
        for row in index.itertuples():
            data = (payloads / f"{row.image}.jpg").read_bytes()
            if hashlib.sha256(data).hexdigest() != row.sha256:
                raise ValueError(f"photo {row.image} differs from its manifest")
            start = time.perf_counter()
            x = served.recipe(decode_photo(data))
            prepared = time.perf_counter()
            logits = served.infer(x)
            done = time.perf_counter()
            predicted, confidence = answer(logits)
            rows.append(
                {
                    "image": row.image,
                    "label": row.label,
                    "split": row.split,
                    "bytes": row.bytes,
                    "receive_ms": (prepared - start) * 1000.0,
                    "compute_ms": (done - prepared) * 1000.0,
                    "predicted": predicted,
                    "confidence": confidence,
                    "correct": predicted == row.label,
                    **cpu_state(),
                }
            )
        frame = pd.DataFrame(rows).assign(
            server=args.server, model=key, photo=payloads.name, threads=args.threads, **hardware
        )  # fmt: skip
        safe = key.replace("/", "_")
        out = results / f"menu_server_{args.server}_{safe}_{payloads.name}.csv.gz"
        frame.to_csv(out, index=False)
        console.print(
            f"{key} on {args.server}: accuracy {frame['correct'].mean() * 100:.2f}%, "
            f"receive {frame['receive_ms'].median():.2f} ms, compute "
            f"{frame['compute_ms'].median():.2f} ms (medians); wrote [bold]{out}[/bold]"
        )
    return 0


DEVICE_MODELS = ("resnet18", "mnv3l", "mnv4s", "mnv4m")


def device_variants(directory: Path) -> list[str]:
    """Every exported device variant, as model@resolution/precision, in a stable order."""
    keys = []
    for path in sorted(directory.glob("*@*.json")):
        model, _, resolution = path.stem.partition("@")
        if model not in DEVICE_MODELS:
            continue
        for precision in ("float32", "int8"):
            graph = path.with_suffix(".int8.onnx" if precision == "int8" else ".onnx")
            if graph.exists():
                keys.append(f"{model}@{resolution}/{precision}")
    return keys


def cmd_menu_profile_device(args: argparse.Namespace) -> int:
    """Time every device variant per image, the photo encodings, and optionally energy."""
    import time

    from edge_inference.energy import counters, measure, summarise
    from edge_inference.menu import cpu_settings, profile_variant, timing_sample
    from edge_inference.preprocess import encode_photo, load_frame
    from edge_inference.serving import open_model

    directory, root, results = Path(args.dir), Path(args.root), Path(args.results)
    cpus = [int(c) for c in args.cpus.split(",")] if args.cpus else sorted(os.sched_getaffinity(0))
    if args.cpus:
        os.sched_setaffinity(0, cpus)
    table = pd.read_csv(args.manifest)
    chosen = timing_sample(table, args.images)
    frames = [(row.image, load_frame(root / row.file)) for row in chosen.itertuples()]
    keys = args.variants or device_variants(directory)
    settings = cpu_settings(cpus)
    labels = {"config": args.config, "threads": args.threads, **settings,
              "machine": machine_model(), "cpu": cpu_model(), "power_source": power_source()}  # fmt: skip
    console.print(f"{args.config}: {len(keys)} variants on {len(frames)} images, cores {cpus}")

    timed = []
    for key in keys:
        served = open_model(*parse_variant(key, directory), threads=args.threads)
        cool_down(args.cool_to)
        rows = profile_variant(
            served.session, served.recipe, frames, warmup=args.warmup,
            clock=time.perf_counter, cpus=cpus,
        )  # fmt: skip
        timed.append(rows.assign(variant=key))
        console.print(
            f"  {key}: device {rows['device_ms'].median():.2f} ms, "
            f"preprocess {rows['preprocess_ms'].median():.2f} ms (medians)"
        )
    device = pd.concat(timed, ignore_index=True).assign(**labels)
    device.to_csv(results / f"menu_device_{args.config}.csv.gz", index=False)

    encoded = []
    for photo in args.photos:
        shorter, quality = parse_photo(photo)
        for image, frame in frames:
            start = time.perf_counter()
            data = encode_photo(frame, shorter, quality)
            encoded.append({"image": image, "photo": photo, "bytes": len(data),
                            "prepare_ms": (time.perf_counter() - start) * 1000.0})  # fmt: skip
    pd.DataFrame(encoded).assign(**labels).to_csv(
        results / f"menu_encode_{args.config}.csv.gz", index=False
    )

    if args.energy:
        found = counters()
        if not found:
            console.print("[yellow]RAPL counters are not readable: no energy measured.[/yellow]")
        else:
            rows = []
            frame = frames[0][1]
            for key in keys:
                served = open_model(*parse_variant(key, directory), threads=args.threads)
                run = lambda s=served: s.session.run(None, {"input": s.recipe(frame)})  # noqa: E731
                cool_down(args.cool_to)
                runs = [measure(run, args.energy_runs, found) for _ in range(args.energy_repeats)]
                for domain in found:
                    rows.append({"what": key, "domain": domain, **summarise(runs, domain),
                                 "idle_w": float(np.mean([r[f"{domain}_idle_w"] for r in runs])),
                                 "runs": args.energy_runs, "repeats": args.energy_repeats})  # fmt: skip
                console.print(f"  {key}: {rows[-len(found)]['mean_j'] * 1000:.1f} mJ (package)")
            for photo in args.photos:
                shorter, quality = parse_photo(photo)
                encode = lambda s=shorter, q=quality: encode_photo(frame, s, q)  # noqa: E731
                runs = [
                    measure(encode, args.energy_runs, found) for _ in range(args.energy_repeats)
                ]
                for domain in found:
                    rows.append({"what": photo, "domain": domain, **summarise(runs, domain),
                                 "idle_w": float(np.mean([r[f"{domain}_idle_w"] for r in runs])),
                                 "runs": args.energy_runs, "repeats": args.energy_repeats})  # fmt: skip
            pd.DataFrame(rows).assign(**labels).to_csv(
                results / f"menu_energy_{args.config}.csv", index=False
            )
    console.print(f"Wrote the {args.config} profiles to [bold]{results}[/bold]")
    return 0


def cmd_menu_profile_prepare(args: argparse.Namespace) -> int:
    """Time each device recipe's preparation, and each photo's encoding, on every image."""
    import functools
    import time

    from edge_inference.menu import cpu_settings, profile_preparation
    from edge_inference.preprocess import Preprocess, encode_photo, load_frame

    directory, root, results = Path(args.dir), Path(args.root), Path(args.results)
    cpus = [int(c) for c in args.cpus.split(",")] if args.cpus else sorted(os.sched_getaffinity(0))
    if args.cpus:
        os.sched_setaffinity(0, cpus)
    table = pd.read_csv(args.manifest)
    if args.split:
        table = table[table["split"] == args.split]
    images = [(row.image, root / row.file) for row in table.sort_values("image").itertuples()]
    steps = {
        path.stem: Preprocess.load(path)
        for path in sorted(directory.glob("*@*.json"))
        if path.stem.partition("@")[0] in DEVICE_MODELS
    }
    for photo in args.photos:
        shorter, quality = parse_photo(photo)
        steps[photo] = functools.partial(encode_photo, shorter_side=shorter, quality=quality)
    console.print(f"{args.config}: {len(steps)} steps on {len(images)} images, cores {cpus}")
    cool_down(args.cool_to)
    rows = profile_preparation(steps, images, load=load_frame, clock=time.perf_counter, cpus=cpus)
    labels = {"config": args.config, **cpu_settings(cpus), "machine": machine_model(),
              "cpu": cpu_model(), "power_source": power_source()}  # fmt: skip
    out = results / f"menu_prepare_{args.config}.csv.gz"
    rows.assign(**labels).to_csv(out, index=False)
    medians = rows.groupby("step")["ms"].median().round(2)
    console.print(medians.to_string())
    console.print(f"Wrote [bold]{out}[/bold]")
    return 0


def cmd_menu_live(args: argparse.Namespace) -> int:
    """Carry plans out frame by frame against a menu server, and log every frame."""
    import time

    from edge_inference.device import Device, PhotoRoute, Variant
    from edge_inference.preprocess import load_frame
    from edge_inference.serving import open_model
    from edge_inference.simulate import Plan
    from edge_inference.wire import Connection

    table = pd.read_csv(args.manifest)
    table = table[table["split"] == args.split].sample(
        n=min(args.frames, len(table)), random_state=SEED
    )
    directory, root = Path(args.dir), Path(args.root)
    variants = {}
    if args.variant:
        served = open_model(*parse_variant(args.variant, directory), threads=args.threads)
        variants[args.variant] = Variant(served.session, served.recipe)
    shorter, quality = parse_photo(args.photo)
    routes = {"photo": PhotoRoute(shorter, quality, args.server_model)}
    link = None
    if args.emulate:
        rtt, mbps = (float(v) for v in args.emulate.split(","))
        link = lambda size: rtt + size * 8 / (mbps * 1000.0)  # noqa: E731
    host, _, port = args.server.rpartition(":")
    connection = Connection.open(host, int(port), link=link)
    device = Device(variants, routes, connection)
    plans = {
        "local": Plan(args.variant),
        "offload": Plan(route="photo"),
        "hedge": Plan(args.variant, route="photo", hedge=True),
    }
    rows = []
    for name in args.plans:
        plan = plans[name]
        for row in table.itertuples():
            frame = load_frame(root / row.file)
            measured = device.run(
                plan, frame, row.label,
                deadline_ms=args.deadline, horizon_ms=args.horizon, cancel=args.cancel,
            )  # fmt: skip
            rows.append({"plan": name, "image": row.image, **dataclasses.asdict(measured)})
            time.sleep(args.period_ms / 1000.0)
    connection.close()
    frames = pd.DataFrame(rows).assign(
        variant=args.variant, photo=args.photo, server_model=args.server_model,
        deadline_ms=args.deadline, horizon_ms=args.horizon, cancel=args.cancel,
        emulated=args.emulate or "", link=args.link,
    )  # fmt: skip
    frames["useful"] = frames["correct"] & (frames["total_ms"] <= args.deadline)
    arrived = frames["server_at_ms"] <= args.horizon
    frames["record"] = frames["server_correct"].where(arrived, frames["local_correct"])
    out = Path(args.out or RESULTS_DIR / f"menu_live_{args.link}_{args.label}.csv.gz")
    frames.to_csv(out, index=False)
    summary = frames.groupby("plan").agg(
        useful=("useful", "mean"),
        record=("record", "mean"),
        busy_ms=("device_busy_ms", "mean"),
        total_ms=("total_ms", "median"),
        cancelled=("cancelled", "mean"),
    )
    console.print(summary.round(3).to_string())
    console.print(f"Wrote [bold]{out}[/bold]")
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

    if args.early_exit:
        model = build_early_exit(args.backbone, num_classes=NUM_CLASSES, pretrained=True)
        exit_weights = args.exit_weights
        if len(exit_weights) != len(EXIT_NAMES):
            console.print(
                f"[bold red]--exit-weights needs {len(EXIT_NAMES)} values, "
                f"one per exit ({', '.join(EXIT_NAMES)}).[/bold red]"
            )
            return 2
        variant = "early_exit"
    else:
        model = build_backbone(args.backbone, num_classes=NUM_CLASSES, pretrained=True)
        exit_weights = None
        variant = "plain"

    # Defaults depend on the variant, so a plain run and an early-exit run never
    # overwrite each other's checkpoint or history.
    suffix = "_early_exit" if args.early_exit else ""
    out = args.out or str(CHECKPOINT_DIR / f"{args.backbone}{suffix}_imagenette.pt")
    history_out = args.history or str(RESULTS_DIR / f"training_history{suffix}.csv")

    console.print(
        f"Fine-tuning [bold]{args.backbone}[/bold] ({variant}) on {len(train_dataset)} images, "
        f"validating on {len(val_dataset)}, for {args.epochs} epoch(s) on {device}"
    )

    history = fine_tune(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        exit_weights=exit_weights,
    )

    checkpoint = Path(out)
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
    frame["variant"] = variant
    if exit_weights is not None:
        frame["exit_weights"] = ",".join(str(w) for w in exit_weights)

    history_path = Path(history_out)
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


def cmd_exits_profile(args: argparse.Namespace) -> int:
    """Record latency, confidence and correctness per image at every exit."""
    seed_everything()

    model = build_early_exit(args.backbone, num_classes=NUM_CLASSES)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        console.print(
            f"[bold red]No checkpoint at {checkpoint}.[/bold red] "
            "Train one with `edge train --early-exit` first. Untrained exits would "
            "report confidences that mean nothing."
        )
        return 1
    load_checkpoint(model, checkpoint)

    dataset = load_split("val", root=Path(args.root), image_size=args.image_size)
    macs = operations_per_exit(model, image_size=args.image_size)
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))

    if args.runtime == "pytorch" and args.precision != "float32":
        console.print("[bold red]INT8 is profiled through ONNX Runtime only.[/bold red]")
        return 2

    console.print(
        f"Profiling {count} images at every exit, {args.runtime} {args.precision}, "
        f"{args.threads} thread(s), after {args.warmup} warmup runs"
    )
    cool_down(args.cool_to)
    if args.runtime == "pytorch":
        rows = profile_exits(
            model,
            dataset,
            threads=args.threads,
            warmup=args.warmup,
            macs_per_exit=macs,
            limit=args.limit,
        )
    else:
        suffix = ".onnx" if args.precision == "float32" else f".{args.precision}.onnx"
        paths = stage_paths(sized_base(checkpoint, args.image_size), suffix=suffix)
        missing = [path for path in paths if not path.exists()]
        if missing:
            command = "export" if args.precision == "float32" else "quantize"
            console.print(
                f"[bold red]No stage graph at {missing[0]}.[/bold red] "
                f"Run `edge exits {command}` first."
            )
            return 1
        rows = profile_exit_sessions(
            build_stage_sessions(paths, threads=args.threads),
            dataset,
            threads=args.threads,
            warmup=args.warmup,
            macs_per_exit=macs,
            precision=args.precision,
            limit=args.limit,
        )
    frame = pd.DataFrame(rows)

    summary = frame.groupby("exit_name", sort=False).agg(
        mmacs=("mmacs", "first"),
        accuracy=("correct", "mean"),
        median_ms=("cumulative_ms", "median"),
        p95_ms=("cumulative_ms", lambda s: s.quantile(0.95)),
        mean_confidence=("confidence", "mean"),
    )
    table = Table(title=f"Every exit, {count} images, {args.threads} thread(s)")
    for column in ("exit", "MMAC", "accuracy", "median ms", "p95 ms", "mean confidence"):
        table.add_column(column, justify="right")
    for name, row in summary.iterrows():
        table.add_row(
            str(name),
            f"{row['mmacs']:.1f}",
            f"{row['accuracy'] * 100:.2f}%",
            f"{row['median_ms']:.2f}",
            f"{row['p95_ms']:.2f}",
            f"{row['mean_confidence']:.3f}",
        )
    console.print(table)

    # The checkpoint name is part of the default, so profiling two differently
    # trained models never overwrites one with the other.
    #
    # Gzipped because profiles are measurements, not derived files: a re-run
    # measures a different moment rather than reproducing this one, so they
    # are committed rather than regenerated. Compression cuts them about 5x,
    # and pandas reads .csv.gz directly.
    default_name = (
        f"exit_profile_{sized_base(checkpoint, args.image_size).name}_{args.runtime}"
        f"_{args.precision}_t{args.threads}.csv.gz"
    )
    out = Path(args.out or RESULTS_DIR / default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    console.print(f"Wrote [bold]{out}[/bold] ({len(frame)} rows)")
    return 0


def cmd_exits_export(args: argparse.Namespace) -> int:
    """Export each stage of an early-exit model to ONNX and verify every exit."""
    seed_everything()

    model = build_early_exit(args.backbone, num_classes=NUM_CLASSES)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        console.print(
            f"[bold red]No checkpoint at {checkpoint}.[/bold red] "
            "Train one with `edge train --early-exit` first."
        )
        return 1
    load_checkpoint(model, checkpoint)

    # Stage graphs sit beside the checkpoint they came from, named after it.
    base = sized_base(checkpoint, args.image_size)
    paths = export_exit_stages(model, base, image_size=args.image_size, opset=args.opset)
    for path in paths:
        console.print(f"Exported [bold]{path}[/bold] ({path.stat().st_size / 2**20:.2f} MiB)")

    rows = verify_stage_parity(
        model, paths, image_size=args.image_size, samples=args.samples, atol=args.atol
    )

    table = Table(title="PyTorch against chained ONNX Runtime stages")
    for column in ("exit", "max abs error", "mean abs error", "prediction agreement"):
        table.add_column(column, justify="right")
    for row in rows:
        table.add_row(
            row["exit_name"],
            f"{row['max_abs_error']:.3e}",
            f"{row['mean_abs_error']:.3e}",
            f"{row['prediction_agreement'] * 100:.1f}%",
        )
    console.print(table)

    for row, path in zip(rows, paths, strict=True):
        row["backbone"] = args.backbone
        row["checkpoint"] = checkpoint.name
        row["opset"] = args.opset
        row["onnx_mib"] = path.stat().st_size / 2**20

    results = Path(args.results or RESULTS_DIR / f"onnx_parity_{base.name}.csv")
    results.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results, index=False)
    console.print(f"Wrote [bold]{results}[/bold]")

    # As with the plain export, a flipped prediction at any exit is an error.
    failed = [row["exit_name"] for row in rows if not row["predictions_match"]]
    if failed:
        console.print(
            f"[bold red]Parity failed[/bold red] at {', '.join(failed)}. "
            "Do not measure against these files."
        )
        return 1
    loose = [row["exit_name"] for row in rows if not row["within_tolerance"]]
    if loose:
        console.print(
            f"[bold yellow]Note:[/bold yellow] {', '.join(loose)} exceeded the "
            f"{args.atol:.0e} tolerance, but every prediction still agrees."
        )
    return 0


def cmd_exits_quantize(args: argparse.Namespace) -> int:
    """Quantize each stage of an early-exit model and compare every exit."""
    seed_everything()

    base = sized_base(Path(args.checkpoint), args.image_size)
    sources = stage_paths(base)
    missing = [path for path in sources if not path.exists()]
    if missing:
        console.print(
            f"[bold red]No stage graph at {missing[0]}.[/bold red] Run `edge exits export` first."
        )
        return 1

    # Training images with evaluation preprocessing, as for the plain model.
    calibration_dataset = load_split(
        "train", root=Path(args.root), augment=False, image_size=args.image_size
    )
    calibration_loader = build_dataloader(
        calibration_dataset, batch_size=args.batch_size, shuffle=True
    )
    batches = calibration_batches(calibration_loader, limit=args.calibration_images)

    console.print(
        f"Calibrating on [bold]{args.calibration_images}[/bold] training images, "
        "then quantizing each stage to INT8"
    )
    destinations = quantize_exit_stages(
        sources,
        stage_paths(base, suffix=".int8.onnx"),
        batches,
        per_channel=not args.per_tensor,
    )

    val_dataset = load_split("val", root=Path(args.root), image_size=args.image_size)
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    rows = []
    for precision, paths in (("float32", sources), ("int8", destinations)):
        console.print(f"Evaluating {precision} on {len(val_dataset)} images")
        sessions = [build_session(path, threads=args.threads) for path in paths]
        accuracies = evaluate_stages(sessions, val_loader)
        cumulative_mib = 0.0
        for name, path, accuracy in zip(EXIT_NAMES, paths, accuracies, strict=True):
            size_mib = path.stat().st_size / 2**20
            cumulative_mib += size_mib
            rows.append(
                {
                    "checkpoint": Path(args.checkpoint).name,
                    "precision": precision,
                    "exit_name": name,
                    "accuracy": accuracy,
                    "stage_mib": size_mib,
                    # What a device that stops at this exit must hold in memory.
                    "cumulative_mib": cumulative_mib,
                }
            )

    frame = pd.DataFrame(rows)
    table = Table(title="float32 against INT8 at every exit")
    for column in ("exit", "float32", "int8", "drop (pts)", "float32 MiB", "int8 MiB"):
        table.add_column(column, justify="right")
    by_precision = {p: group.reset_index() for p, group in frame.groupby("precision")}
    for (_, f32), (_, i8) in zip(
        by_precision["float32"].iterrows(), by_precision["int8"].iterrows(), strict=True
    ):
        table.add_row(
            f32["exit_name"],
            f"{f32['accuracy'] * 100:.2f}%",
            f"{i8['accuracy'] * 100:.2f}%",
            f"{(f32['accuracy'] - i8['accuracy']) * 100:+.2f}",
            f"{f32['cumulative_mib']:.2f}",
            f"{i8['cumulative_mib']:.2f}",
        )
    console.print(table)

    out = Path(args.results or RESULTS_DIR / f"quantization_{base.name}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    console.print(f"Wrote [bold]{out}[/bold]")
    return 0


def cmd_split_profile(args: argparse.Namespace) -> int:
    """Payload size, preparation time and 8-bit accuracy at every split point."""
    seed_everything()

    checkpoint = Path(args.checkpoint)
    suffix = ".onnx" if args.precision == "float32" else f".{args.precision}.onnx"
    paths = stage_paths(checkpoint.with_suffix(""), suffix=suffix)
    missing = [path for path in paths if not path.exists()]
    if missing:
        command = "export" if args.precision == "float32" else "quantize"
        console.print(
            f"[bold red]No stage graph at {missing[0]}.[/bold red] Run `edge exits {command}` first."
        )
        return 1

    dataset = load_split("val", root=Path(args.root))
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    console.print(
        f"Profiling payloads for {count} images at every split point, "
        f"{args.precision} stages, {args.threads} thread(s)"
    )
    cool_down(args.cool_to)
    rows = profile_splits(
        build_stage_sessions(paths, threads=args.threads),
        dataset,
        threads=args.threads,
        warmup=args.warmup,
        precision=args.precision,
        files=image_files(dataset),
        limit=args.limit,
        level=args.level,
    )
    frame = pd.DataFrame(rows)

    # Decimal kilobytes, the unit network bandwidth is quoted in.
    kb = 1000
    table = Table(title=f"What crosses the network, median of {count} images")
    for column in (
        "split after",
        "float32 KB",
        "+ zstd",
        "uint8 KB",
        "+ zstd",
        "zstd ms",
        "accuracy",
        "uint8 accuracy",
    ):
        table.add_column(column, justify="right")
    for name, group in frame.groupby("split_name", sort=False):
        table.add_row(
            str(name),
            f"{group['float32_bytes'].median() / kb:.1f}",
            f"{group['float32_zstd_bytes'].median() / kb:.1f}",
            f"{group['uint8_bytes'].median() / kb:.1f}",
            f"{group['uint8_zstd_bytes'].median() / kb:.1f}",
            f"{(group['uint8_quantize_ms'] + group['uint8_zstd_ms']).median():.2f}",
            f"{group['correct'].mean() * 100:.2f}%",
            f"{group['correct_uint8'].mean() * 100:.2f}%",
        )
    console.print(table)
    jpeg = frame.drop_duplicates("image")["jpeg_bytes"]
    console.print(
        f"Full offload sends the JPEG instead: median [bold]{jpeg.median() / kb:.1f} KB[/bold]"
    )

    default_name = f"split_profile_{checkpoint.stem}_{args.precision}_t{args.threads}.csv.gz"
    out = Path(args.out or RESULTS_DIR / default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    console.print(f"Wrote [bold]{out}[/bold] ({len(frame)} rows)")
    return 0


def cmd_server_profile(args: argparse.Namespace) -> int:
    """Receive and compute time on the server for every way a frame can arrive."""
    seed_everything()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        console.print(f"[bold red]No checkpoint at {checkpoint}.[/bold red]")
        return 1

    # Pinned for both runtimes: JPEG preprocessing runs on PyTorch's pool even
    # when ONNX Runtime does the inference.
    threads = set_thread_count(args.threads)
    if args.runtime == "pytorch":
        device = resolve_device(args.device)
        model = build_early_exit(args.backbone, num_classes=NUM_CLASSES)
        load_checkpoint(model, checkpoint)
        server = TorchServer(model, device)
        accelerator = torch.cuda.get_device_name(0) if device.startswith("cuda") else None
        labels = {"runtime": "pytorch", "precision": "float32", "device": device}
    else:
        suffix = ".onnx" if args.precision == "float32" else f".{args.precision}.onnx"
        paths = stage_paths(checkpoint.with_suffix(""), suffix=suffix)
        missing = [path for path in paths if not path.exists()]
        if missing:
            console.print(f"[bold red]No stage graph at {missing[0]}.[/bold red]")
            return 1
        server = OnnxServer(build_stage_sessions(paths, threads=threads))
        accelerator = None
        labels = {"runtime": "onnxruntime", "precision": args.precision, "device": "cpu"}
    labels |= {"accelerator": accelerator, "threads": threads}

    dataset = load_split("val", root=Path(args.root))
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    target = accelerator or f"{threads} CPU thread(s)"
    console.print(f"Profiling the server side of {count} images on {target}, {args.runtime}")
    cool_down(args.cool_to)
    rows = profile_server(
        server,
        dataset,
        image_files(dataset),
        warmup=args.warmup,
        labels=labels,
        limit=args.limit,
        level=args.level,
    )
    frame = pd.DataFrame(rows)

    table = Table(title=f"Server time per frame, median of {count} images, on {target}")
    for column in ("arrives as", "receive ms", "compute ms", "total ms", "p95 ms", "accuracy"):
        table.add_column(column, justify="right")
    keys = ["mode", "split_name", "payload"]
    for (mode, split_name, payload), group in frame.groupby(keys, sort=False, dropna=False):
        arrives = "JPEG" if mode == "offload" else f"{payload} after {split_name}"
        table.add_row(
            arrives,
            f"{group['receive_ms'].median():.2f}",
            f"{group['compute_ms'].median():.2f}",
            f"{group['total_ms'].median():.2f}",
            f"{group['total_ms'].quantile(0.95):.2f}",
            f"{group['correct'].mean() * 100:.2f}%",
        )
    console.print(table)

    where = "cuda" if accelerator else f"t{threads}"
    default_name = (
        f"server_profile_{checkpoint.stem}_{args.runtime}_{labels['precision']}_{where}.csv.gz"
    )
    out = Path(args.out or RESULTS_DIR / default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    console.print(f"Wrote [bold]{out}[/bold] ({len(frame)} rows)")
    return 0


# Servers profiled so far: the name each gets in the offload table, and the
# suffix of its server profile.
SERVERS = {"t4": "pytorch_float32_cuda", "cpu_t6": "onnxruntime_float32_t6"}


def cmd_offload_table(args: argparse.Namespace) -> int:
    """Join every profile into per-image costs and summarise them per network."""
    stem = Path(args.checkpoint).stem
    results = Path(args.results)
    try:
        components = load_components(results, stem, threads=args.threads, servers=SERVERS)
    except FileNotFoundError as error:
        console.print(f"[bold red]Missing profile:[/bold red] {error.filename}")
        return 1

    summary = summarise(components)
    local = components[(components["mode"] == "local") & (components["point"] == "layer4")]
    breakeven = pd.concat(
        [
            breakeven_mbps(
                components,
                rtt_ms=network.rtt_ms,
                local_ms=local[local["device_precision"] == precision]["device_ms"].median(),
            ).assign(network=name, against=f"local {precision}")
            for name, network in NETWORKS.items()
            for precision in ("float32", "int8")
        ],
        ignore_index=True,
    )

    # A readable cut of the summary: the endpoints of each way of working.
    shown = [
        ("local", "layer4", "float32", "none", "none"),
        ("local", "layer4", "int8", "none", "none"),
        ("local", "layer2", "int8", "none", "none"),
        ("split", "layer3", "float32", "uint8_zstd", "t4"),
        ("offload", "input", "none", "jpeg", "t4"),
        ("offload", "input", "none", "jpeg", "cpu_t6"),
    ]
    keys = ["mode", "point", "device_precision", "payload", "server"]
    for name, network in NETWORKS.items():
        table = Table(
            title=f"{name}: {network.rtt_ms:g} ms round trip, {network.uplink_mbps:g} Mbps uplink"
        )
        for column in ("configuration", "accuracy", "median ms", "p95 ms", "within 50 ms"):
            table.add_column(column, justify="right")
        rows = summary[summary["network"] == name].set_index(keys)
        for key in shown:
            row = rows.loc[key]
            mode, point, precision, payload, server = key
            label = {
                "local": f"local {precision} to {point}",
                "split": f"split after {point}, {payload}, {server}",
                "offload": f"offload to {server}",
            }[mode]
            table.add_row(
                label,
                f"{row['accuracy'] * 100:.2f}%",
                f"{row['median_ms']:.2f}",
                f"{row['p95_ms']:.2f}",
                f"{row['within_50ms'] * 100:.1f}%",
            )
        console.print(table)

    slowdown = breakeven_slowdown(summary)
    table = Table(title="How much slower a device must be for sending to pay off, by median")
    for column in ("network", "fastest way to send", "its median ms", "vs float32", "vs INT8"):
        table.add_column(column, justify="right")
    for network, group in slowdown.groupby("network", sort=False):
        factors = group.set_index("against")["slowdown"]
        first = group.iloc[0]
        how = "offload" if first["best_mode"] == "offload" else f"split after {first['best_point']}"
        table.add_row(
            str(network),
            f"{how} to {first['best_server']}",
            f"{first['best_median_ms']:.2f}",
            f"{factors['local float32']:.2f}x",
            f"{factors['local int8']:.2f}x",
        )
    console.print(table)

    outputs = {
        f"offload_components_{stem}.csv.gz": components,
        f"offload_slowdown_{stem}.csv": slowdown,
        f"offload_summary_{stem}.csv": summary,
        f"offload_breakeven_{stem}.csv": breakeven,
    }
    for filename, frame in outputs.items():
        frame.to_csv(results / filename, index=False)
        console.print(f"Wrote [bold]{results / filename}[/bold] ({len(frame)} rows)")
    return 0


def short_option(row: pd.Series) -> str:
    """An option's name cut down to fit a table cell."""
    if row["useful"] == 0:
        return "nothing fits"
    if row["mode"] == "offload":
        return "offload"
    exit_number = EXIT_NAMES.index(row["point"]) + 1
    if row["mode"] == "split":
        return f"split after {exit_number}"
    if exit_number == len(EXIT_NAMES):
        return f"local {row['device_precision']}"
    return f"{row['device_precision']} exit {exit_number}"


def cmd_decide_map(args: argparse.Namespace) -> int:
    """Name the best single option for every device speed, network and deadline."""
    stem = Path(args.checkpoint).stem
    results = Path(args.results)
    try:
        components = load_components(results, stem, threads=args.threads, servers=SERVERS)
    except FileNotFoundError as error:
        console.print(f"[bold red]Missing profile:[/bold red] {error.filename}")
        return 1

    options, best = decision_map(
        components,
        slowdowns=args.slowdowns,
        deadlines=args.deadlines,
        tie_margin=args.tie_margin,
    )

    # One cell per scenario: the winning option, then the percentage of frames
    # it answers correctly in time, then in brackets the percentage reachable
    # by choosing per frame.
    shown = best[best["server"] == args.server]
    table = Table(
        title=f"Best single option by deadline, server {args.server}\n"
        "% of frames correct in time (best possible per frame)",
        padding=(0, 1),
    )
    table.add_column("device")
    table.add_column("network")
    for deadline in args.deadlines:
        table.add_column(f"{deadline:g} ms")
    for slowdown, speed in shown.groupby("slowdown", sort=False):
        for index, (network, rows) in enumerate(speed.groupby("network", sort=False)):
            cells = [
                f"{short_option(row)}\n{row['useful'] * 100:.2f} ({row['best_possible'] * 100:.2f})"
                for _, row in rows.iterrows()
            ]
            last = index == speed["network"].nunique() - 1
            table.add_row(
                f"{slowdown:g}x" if index == 0 else "", str(network), *cells, end_section=last
            )
    console.print(table)

    outputs = {f"decide_options_{stem}.csv": options, f"decide_map_{stem}.csv": best}
    for filename, frame in outputs.items():
        frame.to_csv(results / filename, index=False)
        console.print(f"Wrote [bold]{results / filename}[/bold] ({len(frame)} rows)")
    return 0


def cmd_decide_evaluate(args: argparse.Namespace) -> int:
    """Run every rule over the same frames and networks, next to the yardsticks."""
    stem = Path(args.checkpoint).stem
    results = Path(args.results)
    try:
        components = load_components(results, stem, threads=args.threads, servers=SERVERS)
    except FileNotFoundError as error:
        console.print(f"[bold red]Missing profile:[/bold red] {error.filename}")
        return 1

    setup = Setup(
        server=args.server,
        slowdown=args.slowdown,
        deadline_ms=args.deadline,
        good=args.good,
        bad=args.bad,
        stay_good=args.stay_good,
        stay_bad=args.stay_bad,
        knowledge=args.knowledge,
        passes=args.passes,
        tune_share=args.tune_share,
        seed=args.seed,
        **hedge_settings(args),
    )
    table = Measurements.from_components(components, args.server)
    result = run_setup(with_sizes(table, results, stem, args.threads, args.sizes), setup)
    board, scenario = result.board, result.scenario
    if not result.held_out:
        console.print(
            "[yellow]No tuning images held out: tuned rules are scored optimistically.[/]"
        )

    shares = ", ".join(
        f"{name} {np.mean(scenario.states == index) * 100:.0f}%"
        for index, name in enumerate(setup.names)
    )
    console.print(
        f"{len(scenario.frames)} frames from {result.test_images} test images; "
        f"device {setup.slowdown:g}x slower; deadline {setup.deadline_ms:g} ms; "
        f"server {setup.server}; network {shares}; rules know the network: {setup.knowledge}"
    )
    shown = Table(title="% of frames answered correctly in time, and what it cost")
    for column in ("rule", "useful", "behind", "accuracy", "in time", "mean ms", "p95 ms", "sent"):
        shown.add_column(column, justify="left" if column == "rule" else "right")
    last_rule = board.index[board["kind"] != "yardstick"][-1]
    for index, row in board.iterrows():
        shown.add_row(
            row["rule"],
            f"{row['useful'] * 100:.2f}",
            f"{row['behind_fair'] * 100:.2f}" if row["kind"] != "yardstick" else "",
            f"{row['accuracy'] * 100:.2f}",
            f"{row['on_time'] * 100:.2f}",
            f"{row['mean_ms']:.2f}",
            f"{row['p95_ms']:.2f}",
            f"{row['sent_share'] * 100:.0f}%",
            end_section=index == last_rule,
        )
    console.print(shown)
    described = board[(board["rule"].str.startswith(("SplitEE", OURS, "1.", "2.", "3.")))]
    for _, row in described.iterrows():
        console.print(f"  {row['rule']}: {row['plans']}")
    console.print(
        "  Sending features after exit 1 costs, in layers of device work: "
        + ", ".join(f"{name} {board[f'offload_layers_{name}'].iloc[0]:.1f}" for name in setup.names)
        + " (the paper assumes 1 to 5)"
    )
    decide_us = board.loc[board["rule"] == OURS, "decide_us"].iloc[0]
    console.print(f"  {OURS} takes {decide_us:.2f} microseconds to decide on a frame")

    tag = f"{stem}_{setup.tag()}{sizes_tag(args.sizes)}"
    outputs = {
        f"decide_scoreboard_{tag}.csv": board,
        f"decide_trace_{tag}.csv.gz": result.traces,
    }
    for filename, frame in outputs.items():
        frame.to_csv(results / filename, index=False)
        console.print(f"Wrote [bold]{results / filename}[/bold] ({len(frame)} rows)")
    return 0


def cmd_decide_sweep(args: argparse.Namespace) -> int:
    """Score the rules across device speeds and deadlines, several seeds each."""
    stem = Path(args.checkpoint).stem
    results = Path(args.results)
    try:
        components = load_components(results, stem, threads=args.threads, servers=SERVERS)
    except FileNotFoundError as error:
        console.print(f"[bold red]Missing profile:[/bold red] {error.filename}")
        return 1

    base = Setup(
        server=args.server,
        good=args.good,
        bad=args.bad,
        stay_good=args.stay,
        stay_bad=args.stay,
        knowledge=args.knowledge,
        passes=args.passes,
        seed=args.seed,
        **hedge_settings(args),
    )
    setups = grid(base, slowdowns=args.slowdowns, deadlines=args.deadlines, seeds=args.seeds)
    console.print(f"Scoring {len(setups)} setups on {args.workers} processes")
    table = Measurements.from_components(components, args.server)
    table = with_sizes(table, results, stem, args.threads, args.sizes)
    boards = run_sweep(table, setups, args.workers)

    # Mean over seeds of the useful share, per device speed and deadline.
    shown = {
        "SplitEE as published": "SplitEE",
        "SplitEE, every repair at once": "repaired",
        OURS: "ours",
        "1. best fixed option": "best fixed",
        FAIR_TARGET: "fair target",
    }
    if args.hedge:
        shown |= {HEDGE: "hedge", OURS_HEDGED: "ours, hedged", HEDGED_TARGET: "3h"}
    useful = (
        boards[boards["rule"].isin(shown)]
        .groupby(["slowdown", "deadline_ms", "rule"], sort=False)["useful"]
        .agg(["mean", "std"])
    )
    table = Table(
        title=f"% of frames correct in time, mean of {args.seeds} seeds "
        f"({args.good} and {args.bad}, rules know the network: {args.knowledge})"
    )
    table.add_column("device")
    table.add_column("deadline")
    for label in shown.values():
        table.add_column(label, justify="right")
    table.add_column("ours, spread", justify="right")
    for slowdown in args.slowdowns:
        for index, deadline in enumerate(args.deadlines):
            cells = [
                f"{useful.loc[(slowdown, deadline, rule), 'mean'] * 100:.2f}" for rule in shown
            ]
            spread = useful.loc[(slowdown, deadline, OURS), "std"] * 100
            table.add_row(
                f"{slowdown:g}x" if index == 0 else "",
                f"{deadline:g} ms",
                *cells,
                f"±{spread:.2f}",
                end_section=index == len(args.deadlines) - 1,
            )
    console.print(table)

    tag = f"{stem}_{args.server}_{args.good}-{args.bad}"
    if args.good != args.bad:
        tag += f"_stay{args.stay:g}"
    if args.knowledge != "exact":
        tag += f"_{args.knowledge.replace(':', '')}"
    tag += base.extras() + sizes_tag(args.sizes)
    out = results / f"decide_sweep_{tag}.csv"
    boards.to_csv(out, index=False)
    console.print(f"Wrote [bold]{out}[/bold] ({len(boards)} rows)")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Answer a device's requests with the float32 network, until interrupted."""
    base = Path(args.checkpoint).with_suffix("")
    paths = stage_paths(base)
    missing = [path for path in paths if not path.exists()]
    if missing:
        console.print(f"[bold red]Missing stage graph:[/bold red] {missing[0]}")
        return 1
    responder = Responder(build_stage_sessions(paths, threads=args.threads))

    def ready(port: int) -> None:
        console.print(
            f"Serving [bold]{base.name}[/bold] on {args.host}:{port} at {args.threads} threads. "
            "Ctrl-C stops."
        )

    try:
        serve(responder, args.host, args.port, on_ready=ready)
    except KeyboardInterrupt:
        console.print("Stopped.")
    return 0


LIVE_RULES = ("lookup", "splitee", "offload", "local-int8", "local-float32")


def cmd_run(args: argparse.Namespace) -> int:
    """Carry a rule out frame by frame against a server, next to the same run looked up."""
    stem = Path(args.checkpoint).stem
    base = Path(args.checkpoint).with_suffix("")
    results = Path(args.results)
    try:
        components = load_components(results, stem, threads=args.threads, servers=SERVERS)
    except FileNotFoundError as error:
        console.print(f"[bold red]Missing profile:[/bold red] {error.filename}")
        return 1

    setup = Setup(
        server=args.server_profile,
        slowdown=args.slowdown,
        deadline_ms=args.deadline,
        good=args.good,
        bad=args.bad,
        stay_good=args.stay,
        stay_bad=args.stay,
        knowledge=args.knowledge,
        passes=1,
        seed=args.seed,
    )
    table = Measurements.from_components(components, setup.server)
    held = split_images(table.labels, setup.tune_share, setup.seed)
    tuning, test = table.subset(held), table.subset(~held)
    whole = setup.scenario(test, setup.seed)
    count = min(args.frames, len(whole.frames))
    scenario = Scenario(
        test,
        whole.frames[:count],
        whole.states[:count],
        whole.networks,
        whole.state_names,
        slowdown=setup.slowdown,
        deadline_ms=setup.deadline_ms,
    )

    def make_rule():
        """A fresh copy of the chosen rule, since a learning rule keeps state."""
        if args.rule == "lookup":
            return Lookup(OURS, tuning, setup.slowdown)
        if args.rule == "splitee":
            repaired = variants(setup.scenario(tuning, setup.seed + 1), scenario, every=False)[-1]
            return Bandit(repaired.name, repaired.settings)
        plans = {"offload": OFFLOAD, "local-int8": Plan("int8"), "local-float32": Plan("float32")}
        return Always(args.rule, plans[args.rule])

    try:
        connection = connect(args.host, args.port)
    except OSError as error:
        console.print(
            f"[bold red]No server at {args.host}:{args.port}[/bold red] ({error}). "
            "Start one with `edge serve` in another terminal."
        )
        return 1

    dataset = load_split("val", root=Path(args.root))
    files = image_files(dataset)
    sessions = {
        "float32": build_stage_sessions(stage_paths(base), threads=args.threads),
        "int8": build_stage_sessions(stage_paths(base, suffix=".int8.onnx"), threads=args.threads),
    }

    def images(image: int) -> tuple[np.ndarray, int]:
        tensor, label = dataset[image]
        return tensor.unsqueeze(0).numpy(), int(label)

    def photos(image: int) -> bytes:
        return files[image].read_bytes()

    device = Device(sessions, connection, slowdown=setup.slowdown)

    def show(frame: dict[str, object]) -> None:
        verdict = "[green]right[/green]" if frame["correct"] else "[red]wrong[/red]"
        timing = "in time" if frame["in_time"] else "[yellow]late[/yellow]"
        console.print(
            f"{frame['frame']:>5}  {frame['network']:<8} {frame['plan']:<42} "
            f"{frame['total_ms']:7.2f} ms  {verdict}, {timing}"
        )

    live = LiveRun(
        device,
        scenario,
        images,
        photos,
        shaped=not args.real_link,
        on_frame=show if args.watch else None,
    )

    # Warm every path up before anything is timed: both precisions and a send.
    first = int(test.images[scenario.frames[0]])
    warm_slowdown, device.slowdown = device.slowdown, 1.0
    for plan in (Plan("float32"), Plan("int8"), OFFLOAD):
        for _ in range(3):
            device.run(plan, images(first)[0], photos(first))
    device.slowdown = warm_slowdown

    before = {"power_source": power_source(), **cpu_state()}
    looked_up = simulate(make_rule(), scenario, setup.knowledge)
    carried_out = simulate(make_rule(), scenario, setup.knowledge, execute=live)
    after = cpu_state()
    connection.close()

    log = pd.DataFrame(live.log)
    carried_out = carried_out.merge(log, on="frame")
    sends = carried_out[carried_out["sent_bytes"] > 0]
    link = "the link as it is" if args.real_link else "an emulated link"
    console.print(
        f"{count} frames; rule {args.rule}; device {setup.slowdown:g}x slower (emulated by "
        f"waiting); deadline {setup.deadline_ms:g} ms; {link}; power {before['power_source']}; "
        f"clock {before['cpu_mhz_max']:.0f} to {after['cpu_mhz_max']:.0f} MHz"
    )
    both = compare(looked_up, carried_out, setup.deadline_ms)
    shown = Table(title="The same frames, looked up and carried out")
    for column in ("run", "useful", "accuracy", "in time", "mean ms", "p95 ms", "sent"):
        shown.add_column(column, justify="left" if column == "run" else "right")
    for _, row in both.iterrows():
        shown.add_row(
            row["run"],
            f"{row['useful'] * 100:.2f}",
            f"{row['accuracy'] * 100:.2f}",
            f"{row['on_time'] * 100:.2f}",
            f"{row['mean_ms']:.2f}",
            f"{row['p95_ms']:.2f}",
            f"{row['sent_share'] * 100:.0f}%",
        )
    console.print(shown)
    agreed = both.iloc[0]
    console.print(
        f"  Same plan on {agreed['same_plan'] * 100:.1f}% of frames, same right-or-wrong on "
        f"{agreed['same_answer'] * 100:.1f}%. Carried out took {agreed['median_gap_ms']:+.2f} ms "
        f"against looked up at the median; 95% of frames were within "
        f"{agreed['p95_abs_gap_ms']:.2f} ms."
    )
    if len(sends):
        named = (sends["network_guessed"] == sends["network"]).mean()
        console.print(
            f"  {len(sends)} frames were sent. From the time each took alone, the network was "
            f"named correctly for {named * 100:.1f}%. Server time {sends['server_ms'].median():.2f} "
            "ms at the median."
        )

    out = results / f"live_{stem}_{setup.tag()}_{args.rule}.csv.gz"
    pd.concat(
        [looked_up.assign(run="simulated"), carried_out.assign(run="live")], ignore_index=True
    ).assign(**dataclasses.asdict(setup), rule=args.rule, real_link=args.real_link).to_csv(
        out, index=False
    )
    console.print(f"Wrote [bold]{out}[/bold]")
    return 0


def cmd_exits_sweep(args: argparse.Namespace) -> int:
    """Exit vectors and a confidence-threshold sweep from a recorded profile."""
    profile = Path(args.profile)
    if not profile.exists():
        console.print(f"[bold red]No profile at {profile}.[/bold red] Run `edge exits profile`.")
        return 1
    frame = pd.read_csv(profile)

    vectors = exit_vectors(frame)
    table = Table(title=f"Exit vectors, {frame['image'].nunique()} images")
    for column in ("exit", "c (MMAC)", "a (accuracy)", "median ms", "p95 ms"):
        table.add_column(column, justify="right")
    for _, row in vectors.iterrows():
        table.add_row(
            row["exit_name"],
            f"{row['mmacs']:.1f}",
            f"{row['accuracy'] * 100:.2f}%",
            f"{row['median_ms']:.2f}",
            f"{row['p95_ms']:.2f}",
        )
    console.print(table)

    sweep = sweep_thresholds(frame, default_thresholds())
    table = Table(title="Stop at the first exit at least this confident")
    columns = ("threshold", "accuracy", "mean ms", "p95 ms", "exit 1", "exit 2", "exit 3")
    for column in columns:
        table.add_column(column, justify="right")
    for _, row in sweep.iterrows():
        table.add_row(
            f"{row['threshold']:.2f}",
            f"{row['accuracy'] * 100:.2f}%",
            f"{row['mean_ms']:.2f}",
            f"{row['p95_ms']:.2f}",
            *(f"{row[f'share_exit{i}'] * 100:.0f}%" for i in (1, 2, 3)),
        )
    console.print(table)

    # Outputs are named after the profile they came from, so sweeps of
    # different models and configurations sit side by side.
    # Strip `.csv` as well as `.gz`, since Path.stem removes only the last suffix.
    stem = profile.name.removesuffix(".gz").removesuffix(".csv").removeprefix("exit_profile_")
    vectors_out = profile.with_name(f"exit_vectors_{stem}.csv")
    sweep_out = profile.with_name(f"exit_sweep_{stem}.csv")
    vectors.to_csv(vectors_out, index=False)
    sweep.to_csv(sweep_out, index=False)
    console.print(f"Wrote [bold]{vectors_out}[/bold] and [bold]{sweep_out}[/bold]")
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


def add_size_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sizes",
        type=lambda s: [int(v) for v in s.split(",")],
        default=[],
        help="also offer the network at these input sizes, from their exit profiles",
    )


def sizes_tag(sizes) -> str:
    """A file-name part for runs that add input sizes, so they never replace runs without."""
    return f"_sizes{'-'.join(map(str, sizes))}" if sizes else ""


def with_sizes(table: Measurements, results: Path, stem: str, threads: int, sizes) -> Measurements:
    """`table` with a variant per precision and extra input size, named like `int8@96`."""
    extra = {}
    for size in sizes:
        for precision in ("float32", "int8"):
            name = f"exit_profile_{stem}@{size}_onnxruntime_{precision}_t{threads}.csv.gz"
            profile = pd.read_csv(results / name)
            extra[f"{precision}@{size}"] = local_from_profile(profile, table.images)
    return table.with_local(extra)


def add_hedge_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for hedged plans, the record horizon and prices, shared by evaluate and sweep."""
    parser.add_argument(
        "--hedge",
        action="store_true",
        help="add hedged plans: send the photo and compute locally at once",
    )
    parser.add_argument(
        "--cancel",
        nargs="?",
        const=AT_EXIT,
        default=NEVER,
        choices=[AT_EXIT, AT_ONCE],
        help="once the server has answered, stop a hedged local run at the next exit "
        "(the default) or at once (now)",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=HORIZON_MS,
        help="ms until which a late answer still corrects the record",
    )
    parser.add_argument(
        "--price-busy",
        type=float,
        default=0.0,
        help="useful share the lookup rules give up per ms of device time",
    )
    parser.add_argument(
        "--price-kb",
        type=float,
        default=0.0,
        help="useful share the lookup rules give up per kB sent",
    )


def hedge_settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        "hedge": args.hedge,
        "cancel": args.cancel,
        "horizon_ms": args.horizon,
        "price_busy": args.price_busy,
        "price_kb": args.price_kb,
    }


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
    origin = data_sub.add_parser(
        "origin", help="label test images by ImageNet origin and compare accuracy"
    )
    origin.add_argument("profiles", nargs="*", help="exit profiles to break down by origin")
    origin.add_argument("--root", default=str(DATA_DIR), help="dataset directory")
    origin.add_argument("--manifest", default="manifests/imagenette_val.csv")
    origin.add_argument("--results", default=str(RESULTS_DIR))
    origin.set_defaults(func=cmd_data_origin)

    menu = subparsers.add_parser("menu", help="pretrained models on ImageNetV2, for the menu")
    menu_sub = menu.add_subparsers(dest="menu_command", required=True)
    menu_manifest = menu_sub.add_parser(
        "manifest", help="fix the tuning, test and calibration images once"
    )
    menu_manifest.add_argument("--root", default=str(IMAGENETV2_ROOT))
    menu_manifest.add_argument("--out", default=str(MENU_MANIFEST))
    menu_manifest.set_defaults(func=cmd_menu_manifest)

    menu_export = menu_sub.add_parser("export", help="export models at resolutions, with INT8")
    menu_export.add_argument("--models", type=lambda s: s.split(","), required=True)
    menu_export.add_argument(
        "--resolutions", type=lambda s: [int(v) for v in s.split(",")], default=[224]
    )
    menu_export.add_argument("--calibration-images", type=int, default=256)
    menu_export.add_argument("--int8-recipe", default="standard", choices=sorted(INT8_RECIPES))
    menu_export.add_argument(
        "--requantize", action="store_true", help="make the INT8 copy again even if present"
    )
    menu_export.add_argument("--manifest", default=str(MENU_MANIFEST))
    menu_export.add_argument("--root", default=str(IMAGENETV2_ROOT))
    menu_export.add_argument("--out", default=str(MENU_DIR))
    menu_export.set_defaults(func=cmd_menu_export)

    menu_answers = menu_sub.add_parser(
        "answers", help="what every exported variant answers for every image"
    )
    menu_answers.add_argument("--models", type=lambda s: s.split(","), required=True)
    menu_answers.add_argument(
        "--resolutions", type=lambda s: [int(v) for v in s.split(",")], default=[224]
    )
    menu_answers.add_argument("--threads", type=int, default=2)
    menu_answers.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    menu_answers.add_argument("--manifest", default=str(MENU_MANIFEST))
    menu_answers.add_argument("--root", default=str(IMAGENETV2_ROOT))
    menu_answers.add_argument("--dir", default=str(MENU_DIR))
    menu_answers.add_argument("--results", default=str(RESULTS_DIR))
    menu_answers.set_defaults(func=cmd_menu_answers)

    menu_photo_answers = menu_sub.add_parser(
        "photo-answers", help="what every server model answers for every sent photo"
    )
    menu_photo_answers.add_argument(
        "--models", type=lambda s: s.split(","), required=True, help="model or model@resolution"
    )
    menu_photo_answers.add_argument(
        "--photos", type=lambda s: s.split(","), default=["jpeg256q85", "jpeg160q85"]
    )
    menu_photo_answers.add_argument(
        "--precisions", type=lambda s: s.split(","), default=["float32", "int8"]
    )
    menu_photo_answers.add_argument("--payloads", default=str(DATA_DIR / "payloads"))
    menu_photo_answers.add_argument("--threads", type=int, default=2)
    menu_photo_answers.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2)
    )
    menu_photo_answers.add_argument("--dir", default=str(MENU_DIR))
    menu_photo_answers.add_argument("--results", default=str(RESULTS_DIR))
    menu_photo_answers.set_defaults(func=cmd_menu_photo_answers)

    menu_serve = menu_sub.add_parser("serve", help="answer photos with the menu's server models")
    menu_serve.add_argument(
        "--models", type=lambda s: s.split(","), required=True,
        help="variants as model@resolution/precision; the first is the default",
    )  # fmt: skip
    menu_serve.add_argument("--threads", type=int, default=6)
    menu_serve.add_argument("--backend", default=ONNXRUNTIME, choices=SERVER_BACKENDS)
    menu_serve.add_argument("--host", default="0.0.0.0")
    menu_serve.add_argument("--port", type=int, default=8765)
    menu_serve.add_argument("--dir", default=str(MENU_DIR))
    menu_serve.set_defaults(func=cmd_menu_serve)

    menu_payloads = menu_sub.add_parser("payloads", help="encode every photo variant once")
    menu_payloads.add_argument("--photos", type=lambda s: s.split(","), default=["jpeg256q85"])
    menu_payloads.add_argument("--split", default="")
    menu_payloads.add_argument("--manifest", default=str(MENU_MANIFEST))
    menu_payloads.add_argument("--root", default=str(IMAGENETV2_ROOT))
    menu_payloads.add_argument("--out", default=str(DATA_DIR / "payloads"))
    menu_payloads.set_defaults(func=cmd_menu_payloads)

    menu_profile_server = menu_sub.add_parser(
        "profile-server", help="time server models on a payload set"
    )
    menu_profile_server.add_argument("--payloads", required=True, help="a folder from payloads")
    menu_profile_server.add_argument("--models", type=lambda s: s.split(","), required=True)
    menu_profile_server.add_argument("--server", required=True, help="a name for this server")
    menu_profile_server.add_argument("--threads", type=int, default=6)
    menu_profile_server.add_argument("--backend", default=ONNXRUNTIME, choices=SERVER_BACKENDS)
    menu_profile_server.add_argument(
        "--images", type=int, default=1000, help="test photos to time; 0 for all"
    )
    menu_profile_server.add_argument("--warmup", type=int, default=20)
    menu_profile_server.add_argument("--cool-to", type=float, default=None)
    menu_profile_server.add_argument("--dir", default=str(MENU_DIR))
    menu_profile_server.add_argument("--results", default=str(RESULTS_DIR))
    menu_profile_server.set_defaults(func=cmd_menu_profile_server)

    menu_profile_device = menu_sub.add_parser(
        "profile-device", help="time every device variant per image, with encodings and energy"
    )
    menu_profile_device.add_argument("--config", required=True, help="a name for this device setup")
    menu_profile_device.add_argument("--variants", type=lambda s: s.split(","), default=None)
    menu_profile_device.add_argument("--images", type=int, default=1000)
    menu_profile_device.add_argument("--threads", type=int, default=2)
    menu_profile_device.add_argument("--cpus", default="", help="cores to pin to, e.g. 2,4")
    menu_profile_device.add_argument("--warmup", type=int, default=20)
    menu_profile_device.add_argument("--cool-to", type=float, default=None)
    menu_profile_device.add_argument(
        "--photos", type=lambda s: s.split(","), default=["jpeg256q85", "jpeg160q85"]
    )
    menu_profile_device.add_argument("--energy", action="store_true")
    menu_profile_device.add_argument("--energy-runs", type=int, default=200)
    menu_profile_device.add_argument("--energy-repeats", type=int, default=5)
    menu_profile_device.add_argument("--manifest", default=str(MENU_MANIFEST))
    menu_profile_device.add_argument("--root", default=str(IMAGENETV2_ROOT))
    menu_profile_device.add_argument("--dir", default=str(MENU_DIR))
    menu_profile_device.add_argument("--results", default=str(RESULTS_DIR))
    menu_profile_device.set_defaults(func=cmd_menu_profile_device)

    menu_profile_prepare = menu_sub.add_parser(
        "profile-prepare", help="time each device recipe's preparation on every image"
    )
    menu_profile_prepare.add_argument("--config", required=True, help="a name for this setup")
    menu_profile_prepare.add_argument("--cpus", default="", help="cores to pin to, e.g. 2,4")
    menu_profile_prepare.add_argument("--split", default="", help="tune or test; empty for all")
    menu_profile_prepare.add_argument(
        "--photos", type=lambda s: s.split(","), default=["jpeg256q85", "jpeg160q85"]
    )
    menu_profile_prepare.add_argument("--cool-to", type=float, default=None)
    menu_profile_prepare.add_argument("--manifest", default=str(MENU_MANIFEST))
    menu_profile_prepare.add_argument("--root", default=str(IMAGENETV2_ROOT))
    menu_profile_prepare.add_argument("--dir", default=str(MENU_DIR))
    menu_profile_prepare.add_argument("--results", default=str(RESULTS_DIR))
    menu_profile_prepare.set_defaults(func=cmd_menu_profile_prepare)

    menu_live = menu_sub.add_parser("live", help="carry plans out against a menu server")
    menu_live.add_argument("--server", required=True, help="host:port")
    menu_live.add_argument("--variant", default="", help="the device's model@resolution/precision")
    menu_live.add_argument("--photo", default="jpeg256q85")
    menu_live.add_argument("--server-model", required=True)
    menu_live.add_argument(
        "--plans", type=lambda s: s.split(","), default=["local", "offload", "hedge"]
    )
    menu_live.add_argument("--cancel", default=NEVER, choices=[NEVER, AT_ONCE])
    menu_live.add_argument("--deadline", type=float, default=100.0)
    menu_live.add_argument("--horizon", type=float, default=HORIZON_MS)
    menu_live.add_argument("--frames", type=int, default=200)
    menu_live.add_argument("--split", default="test")
    menu_live.add_argument("--period-ms", type=float, default=100.0)
    menu_live.add_argument("--threads", type=int, default=2)
    menu_live.add_argument(
        "--emulate", default="", help="rtt_ms,uplink_mbps to emulate the link; empty for a real one"
    )
    menu_live.add_argument("--link", default="loopback", help="a name for the link")
    menu_live.add_argument("--label", default="run")
    menu_live.add_argument("--out", default="")
    menu_live.add_argument("--manifest", default=str(MENU_MANIFEST))
    menu_live.add_argument("--root", default=str(IMAGENETV2_ROOT))
    menu_live.add_argument("--dir", default=str(MENU_DIR))
    menu_live.set_defaults(func=cmd_menu_live)

    menu_scale = menu_sub.add_parser(
        "scale-server",
        help="time a server by another's timings, scaled by how much slower it ran live",
    )
    menu_scale.add_argument("--source", required=True, help="the server whose timings to scale")
    menu_scale.add_argument("--name", required=True, help="the server the live runs reached")
    menu_scale.add_argument(
        "--links", type=lambda s: s.split(","), required=True, help="links it was run live over"
    )
    menu_scale.add_argument("--label", default="live", help="the live runs' label")
    menu_scale.add_argument("--results", default=str(RESULTS_DIR))
    menu_scale.set_defaults(func=cmd_menu_scale_server)

    menu_scoreboard = menu_sub.add_parser(
        "scoreboard", help="score every rule on a recorded link for one device setup"
    )
    menu_scoreboard.add_argument("--config", required=True, help="the device setup profiled")
    menu_scoreboard.add_argument("--link", required=True, help="the recorded link's name")
    menu_scoreboard.add_argument(
        "--servers", type=lambda s: s.split(","), required=True, help="servers over this link"
    )
    menu_scoreboard.add_argument(
        "--deadlines", type=lambda s: [float(v) for v in s.split(",")], default=[50.0, 100.0]
    )
    menu_scoreboard.add_argument(
        "--photos", type=lambda s: s.split(","), default=["jpeg256q85", "jpeg160q85"]
    )
    menu_scoreboard.add_argument(
        "--variants", type=lambda s: s.split(","), default=[], help="device variants; all if empty"
    )
    menu_scoreboard.add_argument(
        "--server-models", type=lambda s: s.split(","), default=[], help="all if empty"
    )
    menu_scoreboard.add_argument(
        "--recording", default="", help="a link recorded once by scripts/record_network.py"
    )
    menu_scoreboard.add_argument("--tune-label", default="tune")
    menu_scoreboard.add_argument("--test-label", default="test")
    menu_scoreboard.add_argument("--period", type=float, default=100.0, help="ms between frames")
    menu_scoreboard.add_argument("--horizon", type=float, default=HORIZON_MS)
    menu_scoreboard.add_argument("--knowledge", default="exact")
    menu_scoreboard.add_argument("--cancel", default=NEVER, choices=[NEVER, AT_EXIT, AT_ONCE])
    menu_scoreboard.add_argument("--seed", type=int, default=SEED)
    menu_scoreboard.add_argument("--manifest", default=str(MENU_MANIFEST))
    menu_scoreboard.add_argument("--payloads", default=str(DATA_DIR / "payloads"))
    menu_scoreboard.add_argument("--results", default=str(RESULTS_DIR))
    menu_scoreboard.set_defaults(func=cmd_menu_scoreboard)

    probe_parser = subparsers.add_parser(
        "probe", help="record a link to a server over the project's protocol"
    )
    probe_parser.add_argument("--server", required=True, help="host:port")
    probe_parser.add_argument("--link", required=True, help="a name for the link")
    probe_parser.add_argument("--label", default="session")
    probe_parser.add_argument("--seconds", type=float, default=600.0)
    probe_parser.add_argument("--tick-ms", type=float, default=100.0)
    probe_parser.add_argument("--out", default="")
    probe_parser.set_defaults(func=cmd_probe)

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
    train.add_argument(
        "--early-exit",
        action="store_true",
        help=f"attach exit heads after {', '.join(EXIT_NAMES[:-1])} and train all exits jointly",
    )
    train.add_argument(
        "--exit-weights",
        type=parse_weight_list,
        default=parse_weight_list("1,1,1"),
        help="relative loss weight per exit, shallowest first (default: 1,1,1)",
    )
    train.add_argument(
        "--out",
        default=None,
        help="checkpoint path (default depends on --early-exit, so runs never overwrite)",
    )
    train.add_argument("--history", default=None, help="per-epoch metrics CSV")
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

    exits = subparsers.add_parser("exits", help="early-exit profiling and analysis")
    exits_sub = exits.add_subparsers(dest="exits_command", required=True)
    exits_profile = exits_sub.add_parser(
        "profile", help="latency, confidence and correctness per image at every exit"
    )
    exits_profile.add_argument("--backbone", default="resnet18", choices=SUPPORTED_BACKBONES)
    exits_profile.add_argument("--root", default=str(DATA_DIR), help="dataset directory")
    exits_profile.add_argument(
        "--checkpoint", default=str(CHECKPOINT_DIR / "resnet18_early_exit_imagenette.pt")
    )
    exits_profile.add_argument(
        "--runtime",
        default="pytorch",
        choices=("pytorch", "onnxruntime"),
        help="onnxruntime runs the stage graphs from `edge exits export`, as deployed",
    )
    exits_profile.add_argument(
        "--precision",
        default="float32",
        choices=("float32", "int8"),
        help="int8 needs `--runtime onnxruntime` and `edge exits quantize`",
    )
    exits_profile.add_argument("--threads", type=int, default=2, help="pinned thread count")
    exits_profile.add_argument(
        "--image-size", type=int, default=IMAGE_SIZE, help="input size; stage graphs must match"
    )
    exits_profile.add_argument("--warmup", type=int, default=10, help="discarded runs first")
    exits_profile.add_argument(
        "--cool-to",
        type=float,
        default=None,
        help="wait until the CPU is at or below this many degrees C before timing",
    )
    exits_profile.add_argument(
        "--limit", type=int, default=None, help="profile only the first N images"
    )
    exits_profile.add_argument(
        "--out", default=None, help="default names the runtime, precision and thread count"
    )
    exits_profile.set_defaults(func=cmd_exits_profile)

    exits_export = exits_sub.add_parser(
        "export", help="export each stage to ONNX and verify every exit against PyTorch"
    )
    exits_export.add_argument("--backbone", default="resnet18", choices=SUPPORTED_BACKBONES)
    exits_export.add_argument(
        "--checkpoint", default=str(CHECKPOINT_DIR / "resnet18_early_exit_imagenette.pt")
    )
    exits_export.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    exits_export.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version")
    exits_export.add_argument("--samples", type=int, default=8, help="inputs compared for parity")
    exits_export.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="absolute tolerance on raw outputs; predictions must match regardless",
    )
    exits_export.add_argument(
        "--results", default=None, help="parity CSV (default names the checkpoint)"
    )
    exits_export.set_defaults(func=cmd_exits_export)

    exits_quantize = exits_sub.add_parser(
        "quantize", help="quantize each stage to INT8 and compare every exit against float32"
    )
    exits_quantize.add_argument("--root", default=str(DATA_DIR), help="dataset directory")
    exits_quantize.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT_DIR / "resnet18_early_exit_imagenette.pt"),
        help="the stage graphs exported beside it are what gets quantized",
    )
    exits_quantize.add_argument(
        "--calibration-images",
        type=int,
        default=256,
        help="training images used to measure activation ranges",
    )
    exits_quantize.add_argument(
        "--per-tensor",
        action="store_true",
        help="one scale per layer instead of per convolution filter, usually less accurate",
    )
    exits_quantize.add_argument("--batch-size", type=int, default=32)
    exits_quantize.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    exits_quantize.add_argument("--threads", type=int, default=6, help="threads for evaluation")
    exits_quantize.add_argument(
        "--results", default=None, help="accuracy CSV (default names the checkpoint)"
    )
    exits_quantize.set_defaults(func=cmd_exits_quantize)

    exits_sweep = exits_sub.add_parser(
        "sweep", help="exit vectors and a confidence-threshold sweep from a profile"
    )
    exits_sweep.add_argument("profile", help="CSV written by `edge exits profile`")
    exits_sweep.set_defaults(func=cmd_exits_sweep)

    split = subparsers.add_parser("split", help="split computing payloads")
    split_sub = split.add_subparsers(dest="split_command", required=True)
    split_profile = split_sub.add_parser(
        "profile", help="payload size, preparation time and 8-bit accuracy at every split point"
    )
    split_profile.add_argument("--root", default=str(DATA_DIR), help="dataset directory")
    split_profile.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT_DIR / "resnet18_early_exit_imagenette.pt"),
        help="the stage graphs exported beside it are what the device runs",
    )
    split_profile.add_argument(
        "--precision",
        default="float32",
        choices=("float32", "int8"),
        help="precision of the device's stage graphs",
    )
    split_profile.add_argument("--threads", type=int, default=2, help="pinned thread count")
    split_profile.add_argument("--warmup", type=int, default=10, help="discarded runs first")
    split_profile.add_argument(
        "--cool-to",
        type=float,
        default=None,
        help="wait until the CPU is at or below this many degrees C before timing",
    )
    split_profile.add_argument("--level", type=int, default=3, help="zstd compression level")
    split_profile.add_argument(
        "--limit", type=int, default=None, help="profile a seeded random sample of N images"
    )
    split_profile.add_argument(
        "--out", default=None, help="default names the checkpoint, precision and thread count"
    )
    split_profile.set_defaults(func=cmd_split_profile)

    server = subparsers.add_parser("server", help="server-side profiling")
    server_sub = server.add_subparsers(dest="server_command", required=True)
    server_profile = server_sub.add_parser(
        "profile", help="receive and compute time for every way a frame can arrive"
    )
    server_profile.add_argument("--backbone", default="resnet18", choices=SUPPORTED_BACKBONES)
    server_profile.add_argument("--root", default=str(DATA_DIR), help="dataset directory")
    server_profile.add_argument(
        "--checkpoint", default=str(CHECKPOINT_DIR / "resnet18_early_exit_imagenette.pt")
    )
    server_profile.add_argument(
        "--runtime",
        default="pytorch",
        choices=("pytorch", "onnxruntime"),
        help="pytorch for a GPU server, onnxruntime for a CPU server",
    )
    server_profile.add_argument(
        "--device", default="cuda", help="pytorch only: cuda for a GPU server, or cpu"
    )
    server_profile.add_argument(
        "--precision",
        default="float32",
        choices=("float32", "int8"),
        help="onnxruntime only: precision of the stage graphs",
    )
    server_profile.add_argument(
        "--threads", type=int, default=6, help="CPU threads, for a CPU server or JPEG decoding"
    )
    server_profile.add_argument("--warmup", type=int, default=10, help="discarded runs first")
    server_profile.add_argument(
        "--cool-to",
        type=float,
        default=None,
        help="wait until the CPU is at or below this many degrees C before timing",
    )
    server_profile.add_argument("--level", type=int, default=3, help="zstd compression level")
    server_profile.add_argument(
        "--limit", type=int, default=None, help="profile a seeded random sample of N images"
    )
    server_profile.add_argument(
        "--out", default=None, help="default names the checkpoint, runtime and hardware"
    )
    server_profile.set_defaults(func=cmd_server_profile)

    offload = subparsers.add_parser("offload", help="per-image costs of every mode, with networks")
    offload_sub = offload.add_subparsers(dest="offload_command", required=True)
    offload_table = offload_sub.add_parser(
        "table", help="join exit, split and server profiles and summarise per network"
    )
    offload_table.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT_DIR / "resnet18_early_exit_imagenette.pt"),
        help="names the profiles to join; the file itself is not read",
    )
    offload_table.add_argument(
        "--threads", type=int, default=2, help="thread count of the device profiles"
    )
    offload_table.add_argument("--results", default=str(RESULTS_DIR), help="profile directory")
    offload_table.set_defaults(func=cmd_offload_table)

    decide = subparsers.add_parser("decide", help="which option is best, and when")
    decide_sub = decide.add_subparsers(dest="decide_command", required=True)
    decide_map = decide_sub.add_parser(
        "map", help="best single option per device speed, network and deadline"
    )
    decide_map.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT_DIR / "resnet18_early_exit_imagenette.pt"),
        help="names the profiles to join; the file itself is not read",
    )
    decide_map.add_argument(
        "--threads", type=int, default=2, help="thread count of the device profiles"
    )
    decide_map.add_argument("--results", default=str(RESULTS_DIR), help="profile directory")
    decide_map.add_argument(
        "--server",
        default="t4",
        choices=sorted(SERVERS),
        help="server shown in the printed table; the CSVs hold every server",
    )
    decide_map.add_argument(
        "--slowdowns",
        type=parse_number_list,
        default=list(SLOWDOWNS),
        help="how many times slower than the profiled device, comma separated",
    )
    decide_map.add_argument(
        "--deadlines",
        type=parse_number_list,
        default=list(DEADLINES_MS),
        help="deadlines in ms, comma separated",
    )
    decide_map.add_argument(
        "--tie-margin",
        type=float,
        default=TIE_MARGIN,
        help="options within this share of the top score tie, and the fastest wins",
    )
    decide_map.set_defaults(func=cmd_decide_map)

    decide_evaluate = decide_sub.add_parser(
        "evaluate", help="score rules on a stream of frames under a changing network"
    )
    decide_evaluate.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT_DIR / "resnet18_early_exit_imagenette.pt"),
        help="names the profiles to join; the file itself is not read",
    )
    decide_evaluate.add_argument(
        "--threads", type=int, default=2, help="thread count of the device profiles"
    )
    decide_evaluate.add_argument("--results", default=str(RESULTS_DIR), help="profile directory")
    decide_evaluate.add_argument("--server", default="t4", choices=sorted(SERVERS))
    decide_evaluate.add_argument(
        "--slowdown", type=float, default=8.0, help="times slower than the profiled device"
    )
    decide_evaluate.add_argument("--deadline", type=float, default=30.0, help="deadline in ms")
    decide_evaluate.add_argument("--good", default="wifi", choices=sorted(NETWORKS))
    decide_evaluate.add_argument(
        "--bad",
        default="4g",
        choices=sorted(NETWORKS),
        help="the same as --good gives a steady network",
    )
    decide_evaluate.add_argument(
        "--stay-good",
        type=float,
        default=0.95,
        help="chance the good network lasts another frame (0.95 is 20 frames on average)",
    )
    decide_evaluate.add_argument("--stay-bad", type=float, default=0.95)
    decide_evaluate.add_argument(
        "--knowledge",
        default="exact",
        help="what rules know of the network: exact, probe:N (every N frames) or sends",
    )
    decide_evaluate.add_argument(
        "--passes",
        type=int,
        default=5,
        help="times each test image is shown; more passes steady the share of each network",
    )
    decide_evaluate.add_argument(
        "--tune-share",
        type=float,
        default=0.5,
        help="share of each class held out for choosing settings; 0 scores every image",
    )
    decide_evaluate.add_argument("--seed", type=int, default=SEED)
    add_hedge_arguments(decide_evaluate)
    add_size_argument(decide_evaluate)
    decide_evaluate.set_defaults(func=cmd_decide_evaluate)

    decide_sweep = decide_sub.add_parser(
        "sweep", help="score the rules across device speeds and deadlines, several seeds each"
    )
    decide_sweep.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT_DIR / "resnet18_early_exit_imagenette.pt"),
        help="names the profiles to join; the file itself is not read",
    )
    decide_sweep.add_argument(
        "--threads", type=int, default=2, help="thread count of the device profiles"
    )
    decide_sweep.add_argument("--results", default=str(RESULTS_DIR), help="profile directory")
    decide_sweep.add_argument("--server", default="t4", choices=sorted(SERVERS))
    decide_sweep.add_argument("--slowdowns", type=parse_number_list, default=[2.0, 4.0, 8.0, 16.0])
    decide_sweep.add_argument(
        "--deadlines", type=parse_number_list, default=[20.0, 30.0, 50.0, 100.0]
    )
    decide_sweep.add_argument("--good", default="wifi", choices=sorted(NETWORKS))
    decide_sweep.add_argument("--bad", default="4g", choices=sorted(NETWORKS))
    decide_sweep.add_argument(
        "--stay", type=float, default=0.95, help="chance a network state lasts another frame"
    )
    decide_sweep.add_argument(
        "--knowledge", default="exact", help="exact, probe:N (every N frames) or sends"
    )
    decide_sweep.add_argument("--passes", type=int, default=5)
    decide_sweep.add_argument(
        "--seeds", type=int, default=3, help="runs per setup, each with its own seed"
    )
    decide_sweep.add_argument("--seed", type=int, default=SEED)
    decide_sweep.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="setups scored at once (default: half the logical cores)",
    )
    add_hedge_arguments(decide_sweep)
    add_size_argument(decide_sweep)
    decide_sweep.set_defaults(func=cmd_decide_sweep)

    serve_parser = subparsers.add_parser(
        "serve", help="answer a device's requests with the float32 network"
    )
    serve_parser.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT_DIR / "resnet18_early_exit_imagenette.pt"),
        help="names the stage graphs to serve; the file itself is not read",
    )
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="0.0.0.0 to accept other machines"
    )
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--threads", type=int, default=6)
    serve_parser.set_defaults(func=cmd_serve)

    run = subparsers.add_parser("run", help="carry a rule out frame by frame against `edge serve`")
    run.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT_DIR / "resnet18_early_exit_imagenette.pt"),
        help="names the stage graphs and profiles; the file itself is not read",
    )
    run.add_argument("--rule", default="lookup", choices=LIVE_RULES)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8765)
    run.add_argument("--threads", type=int, default=2, help="the device's thread count")
    run.add_argument("--frames", type=int, default=300)
    run.add_argument("--slowdown", type=float, default=8.0)
    run.add_argument("--deadline", type=float, default=30.0, help="deadline in ms")
    run.add_argument("--good", default="wifi", choices=sorted(NETWORKS))
    run.add_argument("--bad", default="4g", choices=sorted(NETWORKS))
    run.add_argument("--stay", type=float, default=0.95)
    run.add_argument("--knowledge", default="exact", help="exact, probe:N or sends")
    run.add_argument(
        "--server-profile",
        default="cpu_t6",
        choices=sorted(SERVERS),
        help="the profiled server the looked-up run uses; pick the one most like the live server",
    )
    run.add_argument(
        "--real-link",
        action="store_true",
        help="add no network delay: the link to the server is the real one",
    )
    run.add_argument("--watch", action="store_true", help="print each frame as it is decided")
    run.add_argument("--root", default=str(DATA_DIR))
    run.add_argument("--results", default=str(RESULTS_DIR), help="profile directory")
    run.add_argument("--seed", type=int, default=SEED)
    run.set_defaults(func=cmd_run)

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
