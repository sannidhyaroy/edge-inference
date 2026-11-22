"""Sanity checks for quantizing early-exit stages."""

import torch

from edge_inference.export import (
    FEATURES_NAME,
    OUTPUT_NAME,
    build_session,
    export_exit_stages,
    run_stage,
    stage_paths,
)
from edge_inference.models import EXIT_NAMES, build_early_exit
from edge_inference.quantization import quantize_exit_stages

IMAGE_SIZE = 32
NUM_CLASSES = 4


def test_quantized_stages_chain_and_shrink(tmp_path):
    """Each later stage must accept the features of the quantized stage before it.

    Calibration feeds every stage its own kind of input, so a stage calibrated
    on the wrong tensor would fail here rather than in a long profiling run.
    """
    torch.manual_seed(0)
    model = build_early_exit(num_classes=NUM_CLASSES)
    base = tmp_path / "tiny"
    sources = export_exit_stages(model, base, image_size=IMAGE_SIZE)
    batches = [torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE).numpy() for _ in range(2)]

    destinations = quantize_exit_stages(sources, stage_paths(base, suffix=".int8.onnx"), batches)

    for source, destination in zip(sources, destinations, strict=True):
        assert destination.stat().st_size < source.stat().st_size / 2

    x = batches[0][:1]
    for name, path in zip(EXIT_NAMES, destinations, strict=True):
        outputs = run_stage(build_session(path), x)
        assert outputs[OUTPUT_NAME].shape == (1, NUM_CLASSES), name
        x = outputs.get(FEATURES_NAME)


def test_the_no_vnni_recipe_keeps_the_first_convolution_and_the_classifier_in_float(tmp_path):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    from edge_inference.quantization import INT8_RECIPES, ArrayCalibrationReader, quantize_onnx

    rng = np.random.default_rng(0)

    def tensor(name, *shape):
        values = rng.normal(size=shape).astype(np.float32)
        return helper.make_tensor(name, TensorProto.FLOAT, shape, values.flatten())

    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "w1"], ["a"], name="first", pads=[1, 1, 1, 1]),
            helper.make_node("Relu", ["a"], ["b"]),
            helper.make_node("Conv", ["b", "w2"], ["c"], name="middle", pads=[1, 1, 1, 1]),
            helper.make_node("GlobalAveragePool", ["c"], ["d"]),
            helper.make_node("Flatten", ["d"], ["e"]),
            helper.make_node("MatMul", ["e", "w3"], ["logits"], name="classifier"),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["n", 3, 8, 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n", 5])],
        [tensor("w1", 4, 3, 3, 3), tensor("w2", 4, 4, 3, 3), tensor("w3", 4, 5)],
    )
    source = tmp_path / "tiny.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), source)
    batches = [rng.normal(size=(4, 3, 8, 8)).astype(np.float32) for _ in range(4)]

    def quantized_weights(recipe):
        out = tmp_path / f"{recipe}.onnx"
        quantize_onnx(source, out, ArrayCalibrationReader(batches), **INT8_RECIPES[recipe])
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        assert session.run(None, {"input": batches[0]})[0].shape == (4, 5)
        model = onnx.load(str(out))
        made = {o for n in model.graph.node if n.op_type == "DequantizeLinear" for o in n.output}
        by_name = {n.name: n for n in model.graph.node}
        return {name: by_name[name].input[1] in made for name in ("first", "middle", "classifier")}

    assert quantized_weights("standard") == {"first": True, "middle": True, "classifier": True}
    assert quantized_weights("no-vnni") == {"first": False, "middle": True, "classifier": False}
    assert quantized_weights("no-vnni-minmax") == {
        "first": False,
        "middle": True,
        "classifier": False,
    }
