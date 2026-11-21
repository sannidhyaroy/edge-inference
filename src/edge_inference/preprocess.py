"""Turn a camera frame into a network's input, and into a photo to send, without torch.

Each pretrained model was trained on images prepared its own way: the shorter
side resized to some length with some filter, a centred square cropped, and the
colours normalised with some mean and spread. A device has to repeat that
exactly, or accuracy drops without any error. `Preprocess` holds one model's
recipe and applies it with Pillow and NumPy alone, matching torchvision's
transforms on the same image (a test holds it to that).

At a resolution other than the model's own, the crop is that resolution and
the resize keeps the model's ratio between the two.
"""

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

FILTERS = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC}


def resize_shorter(image: Image.Image, length: int, interpolation: str) -> Image.Image:
    """Resize so the shorter side is `length`, keeping the aspect ratio.

    The longer side is truncated, not rounded, as torchvision does.
    """
    width, height = image.size
    if width <= height:
        size = (length, int(length * height / width))
    else:
        size = (int(length * width / height), length)
    return image.resize(size, FILTERS[interpolation])


def centre_crop(image: Image.Image, size: int) -> Image.Image:
    """The centred `size` by `size` square, placed as torchvision places it."""
    width, height = image.size
    top = round((height - size) / 2.0)
    left = round((width - size) / 2.0)
    return image.crop((left, top, left + size, top + size))


@dataclass(frozen=True)
class Preprocess:
    """One model's input recipe at one resolution."""

    resolution: int
    resize: int
    interpolation: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def at(self, resolution: int) -> Preprocess:
        """The same recipe for another input size, with the resize ratio kept."""
        resize = round(resolution * self.resize / self.resolution)
        return Preprocess(resolution, resize, self.interpolation, self.mean, self.std)

    def __call__(self, image: Image.Image) -> np.ndarray:
        """A batch of one, channels first, as float32."""
        image = centre_crop(resize_shorter(image, self.resize, self.interpolation), self.resolution)
        pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return ((pixels - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Preprocess:
        values = json.loads(Path(path).read_text())
        return cls(
            int(values["resolution"]),
            int(values["resize"]),
            values["interpolation"],
            tuple(values["mean"]),
            tuple(values["std"]),
        )


def load_frame(path: Path | str) -> Image.Image:
    """A stored image as the camera frame: decoded, in RGB."""
    with Image.open(path) as image:
        return image.convert("RGB")


def encode_photo(frame: Image.Image, shorter_side: int, quality: int) -> bytes:
    """The photo a device sends: the frame resized and encoded as JPEG."""
    resized = resize_shorter(frame, shorter_side, "bilinear")
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def decode_photo(data: bytes) -> Image.Image:
    """A received photo, decoded, in RGB."""
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")
