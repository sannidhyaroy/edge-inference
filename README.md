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
count, each recording the CPU, host, and effective thread count alongside the
timings. `uv run edge bench --help` lists the remaining options.

The model is left randomly initialised here on purpose. Latency depends on
tensor shapes rather than weight values, so this measures the architecture
without downloading pretrained weights or requiring a trained checkpoint.

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

Setup, data loading, and the latency harness are in place. Baseline training,
quantization, and the early-exit models follow.
