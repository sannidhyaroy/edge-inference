"""Browser demo comparing the float32 and INT8 models side by side.

Run with:

    uv run --group app python app/main.py

Shows the quantization trade-off on a real image rather than in a table: the
same prediction, from a model a quarter the size, in noticeably less time.

**Latency here is measured around the model call only**, never around the web
request. The browser round trip, image upload, and decoding are excluded,
because those measure the demo rather than the model. The figures should track
`results/quantization.csv` closely, allowing for a shared laptop being busier
than a benchmark loop.
"""

import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

# Allow running this file directly, without the package being on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_inference.config import CHECKPOINT_DIR, DATA_DIR
from edge_inference.data import build_transforms, load_split
from edge_inference.export import INPUT_NAME, build_session

FLOAT_MODEL = CHECKPOINT_DIR / "resnet18_imagenette.onnx"
INT8_MODEL = CHECKPOINT_DIR / "resnet18_imagenette.int8.onnx"

# One thread per physical core was the best operating point on the reference
# machine, so the demo reports numbers from the configuration the results use.
THREADS = 6

# Timing a single call would report whatever the operating system happened to
# be doing at that instant. A few repeats and the median is far steadier, and
# still fast enough to feel immediate.
TIMED_RUNS = 5


def load_class_names() -> list[str]:
    """Human readable Imagenette class names, in label order.

    torchvision returns each class as a tuple of WordNet synonyms, so "tench"
    arrives as `('tench', 'Tinca tinca')`. Only the first is wanted for a label.
    """
    classes = load_split("val", root=DATA_DIR).classes
    return [names[0] if isinstance(names, tuple) else names for names in classes]


def softmax(scores: np.ndarray) -> np.ndarray:
    """Convert raw scores into probabilities that sum to one.

    Subtracting the maximum first changes nothing mathematically, since it
    cancels in the ratio, but it stops the exponential from overflowing on
    large scores.
    """
    exponentiated = np.exp(scores - scores.max())
    return exponentiated / exponentiated.sum()


class Predictor:
    """Holds one loaded model and runs a preprocessed image through it."""

    def __init__(self, path: Path, label: str) -> None:
        self.label = label
        self.path = path
        self.available = path.exists()
        self.size_mib = path.stat().st_size / (1024 * 1024) if self.available else 0.0
        self.session = build_session(path, threads=THREADS) if self.available else None

    def predict(self, tensor: torch.Tensor) -> tuple[np.ndarray, float]:
        """Return class probabilities and the median latency in milliseconds."""
        batch = tensor.unsqueeze(0).numpy()

        # One untimed call first. The first inference pays one-off costs that
        # have nothing to do with steady-state latency, exactly as in the
        # benchmark harness.
        self.session.run(None, {INPUT_NAME: batch})

        timings = []
        for _ in range(TIMED_RUNS):
            start = time.perf_counter()
            outputs = self.session.run(None, {INPUT_NAME: batch})[0]
            timings.append((time.perf_counter() - start) * 1000.0)

        return softmax(outputs[0]), float(np.median(timings))


def build_interface() -> gr.Blocks:
    class_names = load_class_names()
    transform = build_transforms(train=False)

    models = [Predictor(FLOAT_MODEL, "float32"), Predictor(INT8_MODEL, "INT8")]
    missing = [model.path.name for model in models if not model.available]
    advice = "Run `edge export` and then `edge quantize` to produce them."

    def classify(image: Image.Image | None):
        if image is None or missing:
            return {}, "", {}, ""

        tensor = transform(image.convert("RGB"))
        results: list = []
        for model in models:
            probabilities, median_ms = model.predict(tensor)
            labels = {
                name: float(probability)
                for name, probability in zip(class_names, probabilities, strict=True)
            }
            results.extend([labels, f"**{median_ms:.2f} ms**  ·  {model.size_mib:.1f} MiB"])
        return tuple(results)

    with gr.Blocks(title="Edge Inference") as demo:
        gr.Markdown(
            "# Edge Inference\n"
            "The same fine-tuned ResNet-18 at two precisions. Latency is measured around "
            f"the model call only, at {THREADS} threads, as the median of {TIMED_RUNS} runs "
            "after one discarded warmup. The browser round trip is excluded."
        )
        if missing:
            gr.Markdown(f"> Missing `{'`, `'.join(missing)}`. {advice}")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Image", sources=["upload", "webcam"])
                run = gr.Button("Classify", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("### float32")
                float_labels = gr.Label(num_top_classes=3, label="Prediction")
                float_stats = gr.Markdown()
            with gr.Column(scale=1):
                gr.Markdown("### INT8")
                int8_labels = gr.Label(num_top_classes=3, label="Prediction")
                int8_stats = gr.Markdown()

        outputs = [float_labels, float_stats, int8_labels, int8_stats]
        run.click(classify, inputs=image_input, outputs=outputs)
        image_input.change(classify, inputs=image_input, outputs=outputs)

    return demo


def main() -> int:
    build_interface().launch()
    return 0


# Guarded because gradio's reload mode re-imports this module.
if __name__ == "__main__":
    raise SystemExit(main())
