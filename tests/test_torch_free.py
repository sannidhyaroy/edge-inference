"""The device side and the deciding code must import without torch.

A small board can run ONNX graphs with NumPy and ONNX Runtime alone; torch is
only needed to train and export. These modules are what a device imports.
"""

import subprocess
import sys

import pytest

DEVICE_SIDE = (
    "constants",
    "ort_session",
    "offload",
    "decide",
    "simulate",
    "policies",
    "yardsticks",
    "bandit",
    "scoreboard",
    "manifest",
    "preprocess",
    "menu",
    "wire",
    "serving",
    "probe",
    "signals",
)


@pytest.mark.parametrize("module", DEVICE_SIDE)
def test_imports_without_torch(module):
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['torchvision'] = None\n"
        f"import edge_inference.{module}\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
