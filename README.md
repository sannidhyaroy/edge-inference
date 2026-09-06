# Edge Inference

Early exit and INT8 quantization for inference on edge devices. Measured on a
CPU-only path with constrained cores, the deployment setting edge hardware
imposes.

## What this measures

The model under test is ResNet-18, a convolutional network pretrained on
ImageNet and fine-tuned here on Imagenette, a 10-class ImageNet subset, at its
160px variant.

An early-exit network has extra classifiers attached partway through the
backbone, so an easy input can be answered from an intermediate layer without
running the rest of the network. That turns a fixed-cost model into one with a
tunable accuracy/latency trade-off.

Following Angelucci et al. (2026), a network with early exits is characterised
by two vectors:

- **`c`**: operations (in MOPs) required to reach each exit
- **`a`**: accuracy achieved at each exit, evaluated over the whole test set

Those vectors are normally obtained by offline profiling, and downstream
schedulers then treat the network as a black box described by them. This repo
produces `c` and `a` from real measurements on real hardware, and studies how
INT8 quantization changes them, including whether quantization shifts *which*
exit a given image takes by perturbing the confidence scores the exit criterion
depends on.

Full citations and licenses for the reference papers are in
[`papers/README.md`](papers/README.md).

## Measurement notes

Latency is measured on CPU with batch size 1, warmup runs discarded, and
reported as median and p95 over N timed runs. Thread count is pinned and
recorded alongside every result, both so numbers stay comparable across
machines and so a constrained-core device can be approximated on a laptop.

Two caveats that affect how the numbers should be read:

- **The reference machine is a laptop CPU, not a dedicated edge board.** It
  stands in for one. Constrained thread counts are the approximation.
- **INT8 speedup is hardware-dependent.** Model size reduction (~4x) and
  accuracy change are properties of the quantized model and carry across
  machines. Throughput gains are not: a CPU with AVX2 but no VNNI computes INT8
  correctly but without the single-instruction dot product, so it sees far less
  speedup than the literature's typical figures. Results are reported as
  measured, per machine, with the CPU recorded in every row.

## Requirements

Python 3.14 and [uv](https://docs.astral.sh/uv/). No GPU required: on Linux,
torch resolves from PyTorch's CPU index, keeping the install around 350 MB
rather than pulling in a CUDA stack this project never executes.

## Setup

```bash
uv sync --extra cpu
```

This creates the virtual environment and installs the exact dependency versions
recorded in `uv.lock`.

The extra is required, not optional. `torch` and `torchvision` sit in two
mutually exclusive extras because the correct build depends on the machine and
no environment marker can express "has a usable GPU":

| extra | build | use on |
| --- | --- | --- |
| `cpu` | CPU-only, ~350 MB | measurement machines, where all reported latency comes from |
| `cuda` | CUDA, ~2.5 GB | a training box with an NVIDIA GPU |

One lockfile holds both resolutions, so nothing diverges between machines.
Training may run wherever is fastest, but **every latency number in the results
comes from a CPU machine**, and each result row records the CPU and thread
count it was measured under.

## Usage

Download and verify Imagenette. Roughly 94 MB, cached under `data/`, and safe
to re-run:

```bash
uv run edge data prepare
```

Measure forward pass latency across a sweep of thread counts:

```bash
uv run edge bench --threads 1,2,4,6,12 --runs 50 --warmup 10
```

Results are written to `results/latency_backbone.csv`, one row per thread
count, each recording the machine, CPU, core counts, and effective thread
count alongside the timings. `uv run edge bench --help` lists the remaining
options.

The model is left randomly initialised here on purpose. Latency depends on
tensor shapes rather than weight values, so this measures the architecture
without downloading pretrained weights or requiring a trained checkpoint.

Fine-tune the backbone on Imagenette, writing weights to `checkpoints/` and
per-epoch metrics to `results/training_history.csv`:

```bash
uv run edge train --epochs 3 --batch-size 32
```

Report parameters, operations, and checkpoint size for a trained model:

```bash
uv run edge profile
```

## Training on a GPU machine

Training is the one part of this project that does not have to happen on the
measurement machine, because weights are identical wherever they are computed.
Only latency is hardware-specific.

The difference is large enough to matter. Measured at 35 images per second on
the reference laptop CPU, one epoch over Imagenette's 9469 training images
takes about 4.5 minutes, so three epochs with validation runs roughly 18
minutes. The same three epochs on a free Colab T4 took 85 seconds, about 30
seconds per epoch, reaching 96.20% validation accuracy.

### Getting a Colab runtime

[Google Colab](https://colab.research.google.com) offers a free T4. The
official CLI can create a runtime and open a shell on it, which is preferable
to working in notebook cells: the repo is used as it actually is, and nothing
depends on hidden cell state.

Once, on the local machine:

```bash
uv tool install git+https://github.com/googlecolab/google-colab-cli.git@v0.7.0
```

Install from the tag rather than from PyPI. `colab ssh` landed in v0.7.0, and
the released package may still be older.

An SSH key is required, and it must not be group or world readable:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/colab
chmod 600 ~/.ssh/colab
```

Then create, connect to, and eventually stop a runtime:

```bash
colab new -s edge --gpu T4
colab ssh -s edge -i ~/.ssh/colab
colab stop -s edge
```

The session name is arbitrary. With only one session running, `-s` can be
omitted. **Stop the runtime when finished**, since it consumes quota while it
is alive.

### On the runtime

```bash
# The GPU driver libraries are not on the loader path in an SSH shell.
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH

curl -LsSf https://astral.sh/uv/0.12.1/install.sh | sh

git clone https://github.com/sannidhyaroy/edge-inference.git
cd edge-inference

uv sync --extra cuda
uv run edge data prepare
uv run edge train --device cuda --epochs 8 --batch-size 64 --num-workers 2
```

That first line is not optional and the failure it prevents is confusing.
Without it, `nvidia-smi` reports a missing `libnvidia-ml.so`, torch reports no
CUDA device, and the install looks wrong even though every CUDA package is
present. The GPU is passed through correctly, and `/dev/nvidia0` exists, but
the driver libraries live in `/usr/lib64-nvidia`, which `ldconfig` does not
index. A Colab notebook kernel gets that directory injected into its
environment; a shell opened over SSH does not.

Point it at `/usr/lib64-nvidia` specifically. There is also a
`/usr/local/cuda-*/compat/` directory holding an older driver shim, and using
that one produces a version mismatch rather than a clean failure.

Pin the uv version in that URL to whatever `uv --version` reports on the
machine the lockfile was generated on. Installing it takes about a second.

Three things about that sequence:

- **`--extra cuda` is required.** A bare `uv sync` installs neither torch
  build, and `--extra cpu` would install a CPU-only torch and leave the GPU
  idle while training silently runs at laptop speed.
- **The interpreter needs no separate step.** Colab ships an older Python, but
  `.python-version` is committed and uv downloads a matching interpreter on
  its own during `uv sync`.
- **A larger batch pays off on a GPU** and does not on a CPU, so 64 rather than
  32. Data loading workers are a different matter: a free Colab runtime has
  only two vCPUs, so asking for more than two makes them contend rather than
  help. The scarce resource on that machine is the CPU, not the accelerator.

`--device cuda` fails loudly if torch reports no CUDA device rather than
falling back to CPU, because a silent fallback would still finish, just far
slower, with nothing explaining why.

Training writes `checkpoints/resnet18_imagenette.pt` (about 43 MB) and
`results/training_history.csv`. Copy both back with `colab download` before
stopping the runtime, since Colab runtimes are ephemeral and everything
downstream is measured locally.

Verified on a Tesla T4: torch 2.14 resolves to a CUDA 13 build whose
architecture list includes `sm_75`, which is what a T4 needs.

## Results so far

Forward pass latency for ResNet-18 at 160px, batch size 1, on an HP ProBook
445 G10 (Ryzen 5 7530U, 6 cores, 12 threads), 50 timed runs after 10 discarded
warmup runs:

| threads | median ms | p95 ms | std ms |
| ---: | ---: | ---: | ---: |
| 1 | 55.39 | 56.55 | 0.70 |
| 2 | 30.26 | 31.38 | 0.53 |
| 4 | 38.90 | 40.93 | 0.94 |
| 6 | 24.07 | 29.24 | 1.95 |
| 12 | 25.34 | 40.63 | 6.44 |

Six threads, one per physical core, is the best operating point. Twelve threads
oversubscribes those cores via SMT and buys nothing useful: the median barely
moves while the spread grows roughly threefold, pushing p95 from 29.2 to
40.6 ms. For inference under a deadline, that loss of predictability matters
more than the median does.

The four thread result is slower than two and does not fit that pattern. It
reproduces across separate sweeps and in an isolated process, so it is a
genuine property of this CPU with this model rather than a measurement
artefact. The mechanism is unexplained and is recorded here as measured.

These runs were taken on a machine under normal desktop load. A sweep taken
while a browser spiked produced a 368 ms outlier at two threads, visible
immediately as a standard deviation of 64 ms against a 33 ms median. Reporting
the spread alongside the median is what makes such contamination obvious rather
than merely disappointing.

## Status

Setup, data loading, the latency harness, fine-tuning, and cost profiling are
in place. ONNX export, quantization, and the early-exit models follow.
