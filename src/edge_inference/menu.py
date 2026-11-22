"""The model menu: which images, which split, and what each variant answers.

ImageNetV2's matched-frequency set holds 10 images for each of ImageNet's 1000
classes, in folders named by class index. The manifest fixes, once, which
images are for tuning and which for testing, and which tuning images calibrate
INT8. Every later table joins on the manifest's `image` index, never on the
order files happen to be listed in.
"""

import contextlib
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from edge_inference.config import SEED

TUNE, TEST = "tune", "test"
EXTENSIONS = (".jpeg", ".jpg", ".png")


def imagenetv2_manifest(
    root: Path, *, tune_per_class: int = 5, calibrate_per_class: int = 1, seed: int = SEED
) -> pd.DataFrame:
    """One row per image: index, label, file, sha256, size, split and calibration flag.

    The label is the folder's name read as a number. Sorting the folders as
    text would put "10" before "2" and mislabel nearly every image. Images are
    indexed in (label, file name) order. Within each class, `tune_per_class`
    images are drawn for tuning, and `calibrate_per_class` of those calibrate
    INT8; calibration images are never test images.
    """
    files = sorted(
        (int(path.parent.name), path)
        for path in root.glob("*/*")
        if path.suffix.lower() in EXTENSIONS and path.parent.name.isdigit()
    )
    rng = np.random.default_rng(seed)
    rows = []
    for image, (label, path) in enumerate(files):
        with Image.open(path) as opened:
            width, height = opened.size
        rows.append(
            {
                "image": image,
                "label": label,
                "file": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "width": width,
                "height": height,
            }
        )
    table = pd.DataFrame(rows)
    table["split"] = TEST
    table["calibration"] = False
    for _, members in table.groupby("label"):
        drawn = rng.permutation(members.index.to_numpy())
        table.loc[drawn[:tune_per_class], "split"] = TUNE
        table.loc[drawn[:calibrate_per_class], "calibration"] = True
    return table


def check_manifest(table: pd.DataFrame, root: Path | None = None) -> None:
    """Raise if the manifest is inconsistent, or if `root` holds different files."""
    if not table["image"].is_unique:
        raise ValueError("an image index appears twice")
    if (table["calibration"] & (table["split"] != TUNE)).any():
        raise ValueError("a calibration image is not a tuning image")
    if root is not None:
        for row in table.itertuples():
            data = (root / row.file).read_bytes()
            if hashlib.sha256(data).hexdigest() != row.sha256:
                raise ValueError(f"{row.file} differs from the manifest")


def answers(
    sessions: dict[str, Path],
    recipe_path: Path,
    table: pd.DataFrame,
    root: Path,
    *,
    threads: int,
    batch: int = 32,
    load=None,
) -> pd.DataFrame:
    """What each graph in `sessions` answers for every image in `table`, and how sure it is.

    `sessions` maps a precision to a graph; all share one recipe, so each image
    is prepared once. Accuracy does not depend on the thread count or the batch
    size, so this runs as fast as the machine allows. Timing is measured
    separately, one image at a time.

    `load` turns a file under `root` into a frame: by default the stored image,
    or for a server, the photo a device sent, decoded.
    """
    import onnxruntime as ort

    from edge_inference.ort_session import INPUT_NAME, OUTPUT_NAME
    from edge_inference.preprocess import Preprocess, load_frame

    load = load or load_frame
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    opened = {
        precision: ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        for precision, path in sessions.items()
    }
    recipe = Preprocess.load(recipe_path)
    found = {precision: ([], []) for precision in opened}
    files = table["file"].tolist()
    for start in range(0, len(files), batch):
        x = np.concatenate([recipe(load(root / f)) for f in files[start : start + batch]])
        for precision, session in opened.items():
            logits = session.run([OUTPUT_NAME], {INPUT_NAME: x})[0].astype(np.float64)
            exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
            found[precision][0].append(probabilities.argmax(axis=1))
            found[precision][1].append(probabilities.max(axis=1))
    frames = []
    for precision, (predicted, confidence) in found.items():
        predicted = np.concatenate(predicted)
        frames.append(
            pd.DataFrame(
                {
                    "image": table["image"].to_numpy(),
                    "label": table["label"].to_numpy(),
                    "split": table["split"].to_numpy(),
                    "precision": precision,
                    "predicted": predicted,
                    "confidence": np.concatenate(confidence),
                    "correct": predicted == table["label"].to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def answer_variant(job: tuple) -> pd.DataFrame:
    """`answers` for one model at one resolution, as a job a worker process can run."""
    directory, name, resolution, sessions, table, root, threads = job
    recipe = Path(directory) / f"{name}@{resolution}.json"
    found = answers(sessions, recipe, table, Path(root), threads=threads)
    return found.assign(model=name, resolution=resolution)


def read_photo(path: Path):
    """A sent photo as the server sees it: the stored bytes, decoded."""
    from edge_inference.preprocess import decode_photo

    return decode_photo(Path(path).read_bytes())


def answer_photos(job: tuple) -> pd.DataFrame:
    """`answers` for one server model on one set of sent photos, as a worker's job.

    The photos stand in for frames: each is decoded as the server would, then
    prepared by the model's own recipe.
    """
    directory, name, resolution, sessions, photos, index, threads = job
    table = index.assign(file=index["image"].astype(str) + ".jpg")
    recipe = Path(directory) / f"{name}@{resolution}.json"
    found = answers(sessions, recipe, table, Path(photos), threads=threads, load=read_photo)
    return found.assign(model=name, resolution=resolution, photo=Path(photos).name)


def timing_sample(table: pd.DataFrame, count: int, seed: int = SEED) -> pd.DataFrame:
    """The test images that timing runs use: the same ones on the device and on servers.

    A graph's run time does not depend on what the image shows, so it is timed
    on a sample and borrowed for the rest. `count` of 0 keeps every test image.
    """
    test = table[table["split"] == TEST].sort_values("image")
    if count:
        test = test.sample(n=min(count, len(test)), random_state=seed)
    return test.sort_values("image")


def cpu_settings(cpus: list[int]) -> dict[str, object]:
    """How the device's cores are set: which, their governor, limits and average clock.

    Read for every pinned core, since a slowed device pins some cores and the
    others keep the system's settings.
    """
    base = Path("/sys/devices/system/cpu")

    def read(cpu: int, name: str) -> str:
        try:
            return (base / f"cpu{cpu}" / "cpufreq" / name).read_text().strip()
        except OSError:
            return ""

    boost = ""
    with contextlib.suppress(OSError):
        boost = (base / "cpufreq" / "boost").read_text().strip()
    return {
        "cpus": ",".join(map(str, cpus)),
        "boost": boost,
        "governor": ",".join(read(c, "scaling_governor") for c in cpus),
        "min_mhz": ",".join(str(int(read(c, "scaling_min_freq") or 0) // 1000) for c in cpus),
        "max_mhz": ",".join(str(int(read(c, "scaling_max_freq") or 0) // 1000) for c in cpus),
    }


def average_mhz(cpus: list[int]) -> float:
    """The pinned cores' current clock, averaged; NaN where it cannot be read."""
    values = []
    for cpu in cpus:
        try:
            path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
            values.append(int(path.read_text()) / 1000)
        except OSError, ValueError:
            continue
    return float(np.mean(values)) if values else float("nan")


def profile_variant(
    session,
    recipe,
    frames: list,
    *,
    warmup: int,
    clock,
    cpus: list[int],
) -> pd.DataFrame:
    """Time one graph on each frame: preparing the input, then running it.

    `frames` are (image index, decoded frame) pairs; decoding is the camera's
    work and is not timed.
    """
    from edge_inference.ort_session import INPUT_NAME, OUTPUT_NAME

    first = recipe(frames[0][1])
    for _ in range(warmup):
        session.run([OUTPUT_NAME], {INPUT_NAME: first})
    rows = []
    for image, frame in frames:
        start = clock()
        x = recipe(frame)
        prepared = clock()
        session.run([OUTPUT_NAME], {INPUT_NAME: x})
        done = clock()
        rows.append(
            {
                "image": image,
                "preprocess_ms": (prepared - start) * 1000.0,
                "device_ms": (done - prepared) * 1000.0,
                "cpu_mhz": average_mhz(cpus),
            }
        )
    return pd.DataFrame(rows)


def profile_preparation(
    steps: dict[str, object],
    images: list[tuple[int, Path]],
    *,
    load,
    clock,
    cpus: list[int],
) -> pd.DataFrame:
    """Time every step on every image: each recipe's resize, crop and scaling, and each photo's encoding.

    Both depend on the stored image's size, unlike a graph's run, so they are
    timed on every image rather than on a sample. Each image is decoded once,
    untimed, then handed to each step in turn; the order of the steps rotates
    from one image to the next, so no step always finds the frame freshly in
    cache.
    """
    keys = list(steps)
    rows = []
    for position, (image, path) in enumerate(images):
        frame = load(path)
        shift = position % len(keys)
        for key in keys[shift:] + keys[:shift]:
            start = clock()
            steps[key](frame)
            done = clock()
            rows.append(
                {
                    "image": image,
                    "step": key,
                    "ms": (done - start) * 1000.0,
                    "cpu_mhz": average_mhz(cpus),
                }
            )
    return pd.DataFrame(rows)
