"""Steps run on a Colab GPU VM by `scripts/cloud_session.sh`, one per `colab exec`.

The environment variable STEP picks the step:

    where    where the VM is: a line `REGION <region>`, then city, country, network
    setup    unpack the code, install it with CUDA, fetch tailcat and ImageNetV2,
             encode the test photos and check every one against the device's hashes
    profile  time the cloud models on the GPU, on both photo sizes
    start    start the menu server and `tailcat serve`; a line `ADDRESS tc...`
    status   whether both are still running, with the end of their logs

A step that finishes prints a line `STEP OK`; the caller looks for it, since a
failure inside the notebook kernel does not always reach `colab exec`'s exit
status. Standard library only: this runs in the kernel's own Python.
"""

import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CONTENT = Path("/content")
ROOT = CONTENT / "edge"
TAILCAT = CONTENT / "tailcat"
# The device's photo index for each size, uploaded as /content/expected-<photo>.csv.
EXPECTED = CONTENT
TAILCAT_VERSION = "0.7.0"
TAILCAT_RELEASE = f"https://github.com/tailscale/tailcat/releases/download/v{TAILCAT_VERSION}"
TAILCAT_TARBALL = f"tailcat_{TAILCAT_VERSION}_linux_amd64.tar.gz"
IMAGENETV2 = (
    "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/"
    "imagenetv2-matched-frequency.tar.gz"
)
PHOTOS = ("jpeg256q85", "jpeg160q85")
MODELS = "convnextv2_b@224/float32,convnext_t@224/float32"
SERVER = "cloud-t4"
# A rehearsal on a VM without a GPU sets BACKEND=torch-cpu and IMAGES to a few.
BACKEND = os.environ.get("BACKEND", "torch-cuda")
IMAGES = int(os.environ.get("IMAGES", "1000"))
PORT = 8765
# The installer is told where to put uv: its default differs between VMs.
UV_HOME = CONTENT / "uv-bin"
UV = UV_HOME / "uv"
# `uv run` would otherwise sync the project back to its default extras and
# remove the CUDA build of torch that setup installed.
ENV = {**os.environ, "UV_NO_SYNC": "1", "PATH": f"{UV.parent}:{os.environ['PATH']}"}


def run(command: str, cwd: Path = ROOT) -> None:
    """Run a shell command, its output passed on line by line; stop on failure."""
    print(f"$ {command}", flush=True)
    process = subprocess.Popen(
        command, shell=True, cwd=cwd, env=ENV, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )  # fmt: skip
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise SystemExit(f"failed ({process.returncode}): {command}")


def where() -> None:
    with urllib.request.urlopen("https://ipinfo.io/json", timeout=15) as reply:
        found = json.load(reply)
    print(f"REGION {found.get('region', '')}")
    print(found.get("city"), found.get("country"), found.get("org"))


def check_photos() -> None:
    """Every test photo encoded here must be byte for byte the one the device sends."""
    for photo in PHOTOS:
        with (EXPECTED / f"expected-{photo}.csv").open() as file:
            expected = {row["image"]: row["sha256"] for row in csv.DictReader(file)}
        with (ROOT / "data" / "payloads" / f"{photo}.csv").open() as file:
            made = {row["image"]: row["sha256"] for row in csv.DictReader(file)}
        differ = [image for image, digest in made.items() if expected.get(image) != digest]
        if differ or not made:
            raise SystemExit(
                f"{photo}: {len(differ)} of {len(made)} photos differ, e.g. {differ[:3]}"
            )
        print(f"{photo}: all {len(made)} test photos match the device's hashes")


def setup() -> None:
    ROOT.mkdir(exist_ok=True)
    run(f"tar -xzf {CONTENT / 'code.tgz'} -C {ROOT}")
    if not UV.exists():
        run(
            f"curl -LsSf https://astral.sh/uv/install.sh "
            f"| env UV_INSTALL_DIR={UV_HOME} UV_NO_MODIFY_PATH=1 sh > /dev/null"
        )
    run(f"{UV} sync --extra cuda --no-dev --locked -q")
    if BACKEND == "torch-cuda":
        run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
    run("mkdir -p results")

    TAILCAT.mkdir(exist_ok=True)
    run(f"curl -fsSLO {TAILCAT_RELEASE}/{TAILCAT_TARBALL}", cwd=TAILCAT)
    run(f"curl -fsSLO {TAILCAT_RELEASE}/checksums.txt", cwd=TAILCAT)
    run(f"grep ' {TAILCAT_TARBALL}$' checksums.txt | sha256sum -c -", cwd=TAILCAT)
    run(f"tar -xzf {TAILCAT_TARBALL} tailcat", cwd=TAILCAT)

    data = ROOT / "data" / "imagenetv2"
    data.mkdir(parents=True, exist_ok=True)
    run(f"curl -fsSL {IMAGENETV2} | tar -xz -C {data}")
    run(f"{UV} run edge menu payloads --photos {','.join(PHOTOS)} --split test")
    check_photos()


def profile() -> None:
    for photo in PHOTOS:
        run(
            f"{UV} run edge menu profile-server --backend {BACKEND} --server {SERVER} "
            f"--models {MODELS} --payloads data/payloads/{photo} --threads 2 --images {IMAGES}"
        )


def start() -> None:
    logs = {name: (CONTENT / f"{name}.log").open("w") for name in ("serve", "tailcat")}
    subprocess.Popen(
        f"{UV} run edge menu serve --backend {BACKEND} --models {MODELS} "
        f"--host 127.0.0.1 --port {PORT}",
        shell=True, cwd=ROOT, env=ENV, stdout=logs["serve"], stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )  # fmt: skip
    wait_for(CONTENT / "serve.log", "Serving", seconds=600)
    subprocess.Popen(
        f"./tailcat serve {PORT}",
        shell=True, cwd=TAILCAT, env=ENV, stdout=logs["tailcat"], stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )  # fmt: skip
    line = wait_for(CONTENT / "tailcat.log", "new address:", seconds=120)
    print(f"ADDRESS {line.rsplit(':', 1)[1].strip()}")


def wait_for(log: Path, text: str, *, seconds: float) -> str:
    """The first line of `log` containing `text`, waiting up to `seconds` for it."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for line in log.read_text().splitlines():
            if text in line:
                print(line, flush=True)
                return line
        time.sleep(2)
    print(log.read_text()[-3000:])
    raise SystemExit(f"{log.name}: no '{text}' after {seconds:g} s")


def status() -> None:
    found = subprocess.run(
        "pgrep -af 'edge menu serve|tailcat serve' | grep -v pgrep",
        shell=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    print(found)
    for name in ("serve", "tailcat"):
        print(f"--- {name}.log, last lines")
        print("\n".join((CONTENT / f"{name}.log").read_text().splitlines()[-5:]))
    if "edge menu serve" not in found or "tailcat serve" not in found:
        raise SystemExit("the server or tailcat has stopped")


STEPS = {"where": where, "setup": setup, "profile": profile, "start": start, "status": status}
STEPS[os.environ["STEP"]]()
print("STEP OK", flush=True)
sys.stdout.flush()
