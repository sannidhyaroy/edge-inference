import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from edge_inference.preprocess import Preprocess, decode_photo, encode_photo
from edge_inference.serving import MenuServer
from edge_inference.serving_torch import TorchServed, open_torch_model


def tiny_served() -> TorchServed:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 5)).eval()
    recipe = Preprocess(8, 9, "bilinear", (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    return TorchServed(model, recipe, torch.device("cpu"))


def test_infer_gives_the_logits_of_the_one_image_as_numpy():
    served = tiny_served()
    x = np.random.default_rng(0).standard_normal((1, 3, 8, 8)).astype(np.float32)
    expected = served.model(torch.from_numpy(x))[0].detach().numpy()
    found = served.infer(x)
    assert found.shape == (5,) and np.allclose(found, expected)


def test_the_menu_server_answers_with_a_torch_model():
    served = tiny_served()
    photo = encode_photo(Image.fromarray(np.full((12, 16, 3), 200, dtype=np.uint8)), 10, 90)
    reply = MenuServer({"tiny": served}, "tiny").handle({"kind": "photo"}, photo)
    expected = int(served.infer(served.recipe(decode_photo(photo))).argmax())
    assert reply["predicted"] == expected and reply["model"] == "tiny"


def test_only_float32_is_offered():
    with pytest.raises(ValueError, match="float32 only"):
        open_torch_model("convnext_t@224/int8", "cpu")
