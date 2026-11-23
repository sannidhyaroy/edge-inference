"""The model menu's measurements, as one per-image table the simulator can replay.

What an answer is depends only on the image and the model, so every image has
its own: the device models' answers on the stored images, the server models'
answers on the photos the device would send. Preparing an input and encoding a
photo depend on the stored image's size, so they too are timed on every image.

A graph's run time depends on the machine, not on what the image shows, so it
is timed on a sample of test images. Every other image borrows the times of one
timed image, its donor. The donor is the same for every device variant and
every server model, so an image that met a slow moment on one option meets it
on all of them, as it would have if everything had been timed.

Routes are named `photo>server/model@resolution/precision`, for example
`jpeg256q85>edge/convnext_t@224/float32`.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from edge_inference.config import SEED
from edge_inference.simulate import PHOTO, Local, Measurements, Route

FINAL_ONLY = ("final",)


def route_key(photo: str, server: str, model: str) -> str:
    return f"{photo}>{server}/{model}"


def donors(images: np.ndarray, timed: np.ndarray, seed: int = SEED) -> np.ndarray:
    """For each image, the position in `timed` of the image whose times it uses.

    A timed image uses its own. The draw depends only on the two sets of
    images, so every source timed on the same sample gets the same donors.
    """
    timed = np.sort(np.asarray(timed))
    images = np.asarray(images)
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(timed), size=len(images))
    own = np.searchsorted(timed, images)
    is_timed = (own < len(timed)) & (timed[np.minimum(own, len(timed) - 1)] == images)
    return np.where(is_timed, own, drawn)


def _per_image(frame: pd.DataFrame, images: np.ndarray, column: str, what: str) -> np.ndarray:
    """`column` of `frame` in the order of `images`, refusing gaps and repeats."""
    if frame["image"].duplicated().any():
        raise ValueError(f"{what}: an image appears twice")
    indexed = frame.set_index("image")[column]
    missing = np.setdiff1d(images, indexed.index.to_numpy())
    if len(missing):
        raise ValueError(f"{what}: {len(missing)} images missing, e.g. {missing[:3].tolist()}")
    return indexed.loc[images].to_numpy()


def _sampled(frame: pd.DataFrame, column: str, images: np.ndarray, seed: int) -> np.ndarray:
    """A sampled time for every image: its own if timed, else its donor's."""
    frame = frame.sort_values("image")
    if frame["image"].duplicated().any():
        raise ValueError("a timed image appears twice")
    timed = frame["image"].to_numpy()
    return frame[column].to_numpy()[donors(images, timed, seed)]


def _variant(key: str) -> tuple[str, int, str]:
    model, _, rest = key.partition("@")
    resolution, _, precision = rest.partition("/")
    return model, int(resolution), precision or "float32"


def menu_measurements(
    images: pd.DataFrame,
    answers: pd.DataFrame,
    device: pd.DataFrame,
    prepare: pd.DataFrame,
    photo_answers: pd.DataFrame,
    servers: Mapping[str, pd.DataFrame],
    payloads: Mapping[str, pd.DataFrame],
    *,
    link: str,
    variants: Sequence[str] | None = None,
    seed: int = SEED,
) -> Measurements:
    """Assemble the table for one device setup and the servers reached over `link`.

    * `images`: the manifest's `image` and `label` for every image to include.
    * `answers`: device models' answers (`model`, `resolution`, `precision`).
    * `device`: graph times for one device setup, per `variant`, on a sample.
    * `prepare`: per image and `step`, the preparation (`model@resolution`)
      or photo encoding (`jpeg...`) time on that device.
    * `photo_answers`: server models' answers per `photo`.
    * `servers`: per server name, its timing rows: `model` as
      `model@resolution/precision`, `photo`, `image`, `receive_ms`, `compute_ms`.
    * `payloads`: per photo, every image's size in `bytes`.
    """
    images = images.sort_values("image")
    order = images["image"].to_numpy()
    labels = images["label"].to_numpy()
    steps = dict(tuple(prepare.groupby("step")))

    local = {}
    for key in variants or sorted(device["variant"].unique()):
        model, resolution, precision = _variant(key)
        timed = device[device["variant"] == key]
        if timed.empty:
            raise ValueError(f"{key}: no device times")
        prepared = _per_image(steps[f"{model}@{resolution}"], order, "ms", f"{key} preparation")
        ms = prepared + _sampled(timed, "device_ms", order, seed)
        said = answers[
            (answers["model"] == model)
            & (answers["resolution"] == resolution)
            & (answers["precision"] == precision)
        ]
        local[key] = Local(
            ms[:, None],
            _per_image(said, order, "confidence", f"{key} answers")[:, None],
            _per_image(said, order, "correct", f"{key} answers").astype(bool)[:, None],
            FINAL_ONLY,
        )

    routes = {}
    for server, timing in servers.items():
        for (model_key, photo), timed in timing.groupby(["model", "photo"]):
            model, resolution, precision = _variant(model_key)
            said = photo_answers[
                (photo_answers["model"] == model)
                & (photo_answers["resolution"] == resolution)
                & (photo_answers["precision"] == precision)
                & (photo_answers["photo"] == photo)
            ]
            what = route_key(photo, server, model_key)
            timed = timed.assign(server_ms=timed["receive_ms"] + timed["compute_ms"])
            routes[what] = Route(
                bytes=_per_image(payloads[photo], order, "bytes", f"{what} bytes").astype(float),
                prepare_ms=_per_image(steps[photo], order, "ms", f"{what} encoding"),
                server_ms=_sampled(timed, "server_ms", order, seed),
                correct=_per_image(said, order, "correct", f"{what} answers").astype(bool),
                confidence=_per_image(said, order, "confidence", f"{what} answers"),
                payload=PHOTO,
                server=server,
                link=link,
            )
    return Measurements(order, labels, local, routes)


def server_scale(timing: pd.DataFrame, live: pd.DataFrame) -> tuple[float, int]:
    """How much slower a server ran live than `timing` says, and on how many photos.

    `live` holds the frames of live runs that reached the server (`server_ms`
    set), naming the model as `server_model`. The factor is the median, over
    photos timed both ways, of the live time over the timed one: one number,
    since the live runs used one model.
    """
    timed = timing.assign(timed_ms=timing["receive_ms"] + timing["compute_ms"])
    sent = live[live["server_ms"].notna()].rename(columns={"server_model": "model"})
    both = sent.merge(timed, on=["model", "photo", "image"])
    if both.empty:
        raise ValueError("no photo was both timed and sent live")
    return float(np.median(both["server_ms"] / both["timed_ms"])), len(both)


def scaled_server(timing: pd.DataFrame, factor: float, name: str) -> pd.DataFrame:
    """`timing` as server `name`, every time multiplied by `factor`, provenance kept."""
    return timing.assign(
        receive_ms=timing["receive_ms"] * factor,
        compute_ms=timing["compute_ms"] * factor,
        server=name,
        scaled_from=timing["server"],
        scale=factor,
    )
