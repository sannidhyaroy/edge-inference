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

## Status

Environment and project setup. Data loading, the latency harness, baseline
training, quantization, and the early-exit models follow.
