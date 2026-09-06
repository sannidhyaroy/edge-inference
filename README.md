# Edge Inference

Early exit and INT8 quantization for inference on edge devices. Measured on a
CPU-only path with constrained cores, the deployment setting edge hardware
imposes.

---
## **Navigation**
- [Overview](#overview)
- [How measurements are taken](#how-measurements-are-taken)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
  - [Why the extra is required](#why-the-extra-is-required)
- [Usage](#usage)
  - [Prepare the dataset](#prepare-the-dataset)
  - [Measure latency](#measure-latency)
  - [Fine-tune the backbone](#fine-tune-the-backbone)
  - [Profile the model](#profile-the-model)
- [Training on a GPU machine](#training-on-a-gpu-machine)
  - [Getting a Colab runtime](#getting-a-colab-runtime)
  - [On the runtime](#on-the-runtime)
  - [Bringing results back](#bringing-results-back)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)
- [Reference Papers](#reference-papers)

---
## Overview

The model under test is ResNet-18, a convolutional network pretrained on
ImageNet and fine-tuned here on Imagenette, a 10-class ImageNet subset, at its
160px variant.

An early-exit network has extra classifiers attached partway through the
backbone, so an easy input can be answered from an intermediate layer without
running the rest of the network. That turns a fixed-cost model into one with a
tunable accuracy and latency trade-off.

Following Angelucci et al. (2026), a network with early exits is characterised
by two vectors:

| Vector | Meaning |
|---|---|
| `c` | operations, in MOPs, required to reach each exit |
| `a` | accuracy at each exit, evaluated over the whole test set |

Those vectors are normally obtained by offline profiling, and a scheduler then
treats the network as a black box described by them. This repository produces
`c` and `a` from real measurements on real hardware, and studies how INT8
quantization changes them, including whether quantization shifts *which* exit a
given image takes by perturbing the confidence scores the exit criterion
depends on.

---
## How measurements are taken

Latency is measured on CPU with batch size 1, warmup runs discarded, and
reported as median and p95 over N timed runs. Thread count is pinned and
recorded alongside every result, both so numbers stay comparable across
machines and so a constrained-core device can be approximated on a laptop.

Each of those choices exists for a reason:

- **Batch size 1**, because edge inference handles one input at a time. Larger
  batches measure throughput, which is a different and much easier problem, and
  batching is exactly what a datacenter can do and an edge device usually
  cannot.
- **Warmup runs discarded**, because the first few inferences pay one-off costs:
  the allocator grows its pools, the backend picks kernels for shapes it has
  just seen, and caches are cold. Including them inflates the result by an
  amount unrelated to steady-state latency.
- **Median and p95, not mean.** The median is the typical case and shrugs off a
  scheduler hiccup. The p95 is the tail, which is what matters when inference
  has a deadline. A mean alone would hide both, and would make a contaminated
  run look merely disappointing rather than obviously invalid.
- **Thread count pinned and recorded**, so a twelve-thread laptop can imitate a
  two-core device, and so two results from different thread budgets can never be
  compared by accident.

> [!IMPORTANT]
> Two caveats affect how these numbers should be read.
>
> **The reference machine is a laptop CPU, not a dedicated edge board.** It
> stands in for one, and constrained thread counts are the approximation. A real
> edge SoC has different cache sizes, memory bandwidth, and thermal behaviour,
> so what this shows is scaling behaviour rather than a prediction for any
> particular board.
>
> **INT8 speedup is hardware-dependent.** Model size reduction of roughly 4x and
> any accuracy change are properties of the quantized model and carry across
> machines. Throughput gains do not. A CPU with AVX2 but no VNNI computes INT8
> correctly, but without the single-instruction dot product, so it sees far less
> speedup than published figures suggest. Results are reported as measured, per
> machine, with the CPU recorded in every row.

---
## Project Structure

```
edge-inference/
├── src/edge_inference/
│   ├── bench.py            latency harness, hardware detection, thread pinning
│   ├── cli.py              argparse entry point, exposed as the `edge` command
│   ├── config.py           paths, seed, dataset constants
│   ├── data.py             Imagenette loading and preprocessing
│   ├── models.py           backbone construction
│   ├── profiling.py        parameters, operations, checkpoint size
│   └── training.py         fine-tuning and evaluation
├── tests/                  sanity checks for the measurement harness
├── papers/                 citations and licenses for the reference papers
├── results/                committed CSVs and plots
├── data/                   Imagenette, gitignored
└── checkpoints/            trained weights, gitignored
```

`results/` is deliberately tracked. The CSVs in it are the deliverable, while
datasets and weights stay out because they are large and reproducible from the
code plus `uv.lock`.

---
## Requirements

- **Python 3.14.** Some source here uses 3.14-only syntax, so an older
  interpreter fails at import rather than at runtime. You do not need to install
  it yourself: `.python-version` is committed and uv fetches a matching
  interpreter during `uv sync`.
- **[uv](https://docs.astral.sh/uv/)**, the project and dependency manager.
- **No GPU.** Everything measured here runs on CPU by design. A GPU is useful
  only for training, which is optional and covered
  [below](#training-on-a-gpu-machine).

---
## Setup

- Clone the repository:

  ```bash
  git clone https://github.com/sannidhyaroy/edge-inference.git
  cd edge-inference
  ```

- Create the environment and install dependencies:

  ```bash
  uv sync --extra cpu
  ```

  This installs the exact versions recorded in `uv.lock`, including a matching
  Python interpreter if you do not already have one.

- Install the git hooks:

  ```bash
  uv run prek install
  ```

  This wires [prek](https://github.com/j178/prek) into `.git/hooks/`, so `ruff`,
  `pytest`, and a Conventional Commits check run before each commit.

### Why the extra is required

`torch` and `torchvision` do not appear in `dependencies`. They live in two
mutually exclusive extras, and **a bare `uv sync` installs neither**:

| Extra | Build | Use on |
|---|---|---|
| `cpu` | CPU-only, around 350 MB | measurement machines, where all reported latency comes from |
| `cuda` | CUDA, around 2.5 GB | a training box with an NVIDIA GPU |

The split exists because no environment marker can express "this machine has a
usable GPU". Selecting by platform was the obvious alternative and it is wrong:
on Linux it would install a CPU build on a GPU box and leave the accelerator
idle while training silently ran at laptop speed.

One lockfile holds both resolutions, so nothing diverges between machines.

> [!NOTE]
> Training may run wherever is fastest, but **every latency number in the
> results comes from a CPU machine**, and each result row records the machine,
> CPU, and thread count it was measured under.

---
## Usage

### Prepare the dataset

```bash
uv run edge data prepare
```

Downloads Imagenette, roughly 94 MB, into `data/` and reports what landed:

```
┏━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ split ┃ images ┃ classes ┃
┡━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ train │   9469 │      10 │
│ val   │   3925 │      10 │
└───────┴────────┴─────────┘
```

Safe to re-run. torchvision raises rather than skipping when asked to download
over an already extracted archive, and that is caught and treated as success.

> [!NOTE]
> Imagenette ships only `train` and `val`, so `val` doubles as the test set
> throughout. That is why models are trained for a fixed epoch budget and the
> final epoch is reported, rather than keeping whichever epoch scored highest.
> Selecting on the same split the accuracy is reported from would bias it upward.

### Measure latency

```bash
uv run edge bench --threads 1,2,4,6,12 --runs 50 --warmup 10
```

Sweeps thread counts and writes one row per count to
`results/latency_backbone.csv`, each carrying the machine, CPU, physical and
logical core counts, and the effective thread count alongside the timings.

The model is left randomly initialised on purpose. Latency depends on tensor
shapes rather than weight values, so this measures the architecture without
downloading pretrained weights or requiring a trained checkpoint.

> [!IMPORTANT]
> The command warns if the requested and effective thread counts differ, or if
> the lowest thread count was not slower than the highest. Either means thread
> pinning is not taking effect, which would make every constrained-core number
> in the results fiction. Do not ignore those warnings.

### Fine-tune the backbone

```bash
uv run edge train --epochs 8 --batch-size 32
```

Writes weights to `checkpoints/` and per-epoch metrics to
`results/training_history.csv`.

> [!CAUTION]
> Eight epochs takes roughly **48 minutes** on a six-core laptop CPU, measured
> at 35 images per second. If you have access to a GPU, see
> [Training on a GPU machine](#training-on-a-gpu-machine), where the same run
> takes about 4 minutes.

Each run starts from the ImageNet weights, never from a previous checkpoint.
That keeps a result reproducible from the repository alone, and avoids
restarting a finished cosine schedule on already-converged weights, which
loses accuracy before regaining it.

### Profile the model

```bash
uv run edge profile
```

Reports the hardware-independent costs and writes `results/model_cost.csv`:

```
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ measure        ┃      value ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ parameters     │ 11,181,642 │
│ MFLOPs         │     1850.6 │
│ MMACs          │      925.3 │
│ checkpoint MiB │      42.72 │
└────────────────┴────────────┘
```

Parameters, operations, and file size are properties of the model, identical on
any machine. Only the mapping from operations to milliseconds is
hardware-specific, which is why the `c` vector transfers between devices and a
latency table does not.

> [!NOTE]
> A multiply and an accumulate are one MAC but two FLOPs, and papers disagree
> about which they quote. Both are reported so a factor of two cannot silently
> change a comparison.

---
## Training on a GPU machine

Training is the one stage that need not happen on the measurement machine,
because weights are identical wherever they are computed. Only latency is
hardware-specific.

The difference is large enough to matter:

| Machine | Per epoch | Eight epochs |
|---|---|---|
| Ryzen 5 7530U, 6 cores | about 6 minutes | about 48 minutes |
| Free Colab T4 | about 30 seconds | about 4 minutes |

Per-epoch figures are measured: 35 images per second on the laptop CPU, and 85
seconds for a three epoch run on the T4. The eight epoch totals follow from
those rates and include validation.

### Getting a Colab runtime

[Google Colab](https://colab.research.google.com) offers a free T4. Its official
CLI can create a runtime and open a shell on it, which beats working in notebook
cells: the repository is used as it actually is, and nothing depends on hidden
cell state.

- Install the CLI, once per machine:

  ```bash
  uv tool install git+https://github.com/googlecolab/google-colab-cli.git@v0.7.0
  ```

  > [!IMPORTANT]
  > Install from the git tag, not from PyPI. `colab ssh` was added in **v0.7.0**,
  > and `uv tool install google-colab-cli` still resolves to v0.6.0, which has no
  > `ssh` subcommand at all. A later tag is fine, an earlier one is not.

- Create an SSH key if you do not have one. It must not be group or world
  readable, or SSH refuses to use it:

  ```bash
  ssh-keygen -t ed25519 -f ~/.ssh/colab
  chmod 600 ~/.ssh/colab
  ```

- Create a runtime and connect to it:

  ```bash
  colab new -s edge --gpu T4
  colab ssh -s edge -i ~/.ssh/colab
  ```

  The session name is arbitrary. With only one session running, `-s` can be
  omitted entirely.

- When finished, stop it:

  ```bash
  colab stop -s edge
  ```

  > [!CAUTION]
  > A runtime consumes quota while alive, and its filesystem is **ephemeral**.
  > Anything not copied off is gone when it stops. Copy your results back before
  > running this.

### On the runtime

- Put the GPU driver libraries on the loader path:

  ```bash
  export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH
  ```

  This line is not optional, and the failure it prevents is thoroughly
  misleading. See
  [torch reports no CUDA device](#1-torch-reports-no-cuda-device-on-colab) for
  what goes wrong without it.

- Install uv, which is not present by default:

  ```bash
  curl -LsSf https://astral.sh/uv/0.12.1/install.sh | sh
  ```

  Pin the version to whatever `uv --version` reports on the machine that
  generated `uv.lock`, so both resolve identically. It installs in about a second.

- Clone and set up:

  ```bash
  git clone https://github.com/sannidhyaroy/edge-inference.git
  cd edge-inference
  uv sync --extra cuda
  ```

  > [!IMPORTANT]
  > `--extra cuda`, not `--extra cpu` and not a bare `uv sync`. A bare sync
  > installs neither torch build, and `--extra cpu` installs a CPU-only torch
  > that leaves the T4 idle while training runs at laptop speed with no error to
  > tell you.

  You do not need to install Python separately. Colab ships an older interpreter,
  but `.python-version` is committed and uv fetches 3.14 during the sync.

- Download the dataset and train:

  ```bash
  uv run edge data prepare
  uv run edge train --device cuda --epochs 8 --batch-size 64 --num-workers 2
  ```

  Expected output ends with something like:

  ```
    epoch 8/8: train loss 0.1500, val loss 0.0982, val accuracy 96.94%
  Final validation accuracy: 96.94% after 8 epoch(s), which is what was saved
  ```

> [!NOTE]
> **Batch size goes up on a GPU, worker count does not.** 64 rather than 32 uses
> the accelerator far better. But a free runtime has only **two vCPUs**, and at
> roughly 316 images per second the run is bound by JPEG decoding on those cores
> rather than by the T4. Asking for four workers makes them contend instead of
> help, and torch will warn you about it.

`--device cuda` fails loudly if torch reports no CUDA device rather than falling
back to CPU. A silent fallback would still finish, just far slower, with nothing
explaining why.

### Bringing results back

Run these on your own machine, not in the SSH session:

```bash
colab download -s edge /content/edge-inference/checkpoints/resnet18_imagenette.pt checkpoints/resnet18_imagenette.pt
colab download -s edge /content/edge-inference/results/training_history.csv results/training_history.csv
```

Everything downstream, profiling and every latency measurement, runs locally on
the CPU.

---
## Results

### Latency

ResNet-18 at 160px, batch size 1, on an HP ProBook 445 G10 (Ryzen 5 7530U, 6
cores, 12 threads). 50 timed runs after 10 discarded warmup runs:

| threads | median ms | p95 ms | std ms |
| ---: | ---: | ---: | ---: |
| 1 | 55.39 | 56.55 | 0.70 |
| 2 | 30.26 | 31.38 | 0.53 |
| 4 | 38.90 | 40.93 | 0.94 |
| 6 | 24.07 | 29.24 | 1.95 |
| 12 | 25.34 | 40.63 | 6.44 |

**Six threads, one per physical core, is the best operating point.** Twelve
oversubscribes those cores through SMT and buys nothing useful: the median
barely moves while the spread roughly triples, pushing p95 from 29.2 to 40.6 ms.
When two threads share a core they contend for the same vector units, which is
essentially all a convolution does. For inference under a deadline, that loss of
predictability matters more than the median does.

**The four thread result is slower than two** and does not fit that pattern. It
reproduces across separate sweeps and in an isolated process, so it is a genuine
property of this CPU with this model rather than a measurement artefact. The
mechanism is unexplained and is recorded here as measured.

> [!NOTE]
> These runs were taken on a machine under normal desktop load. A sweep taken
> while a browser spiked produced a 368 ms outlier at two threads, visible
> immediately as a standard deviation of 64 ms against a 33 ms median. Reporting
> the spread alongside the median is what makes such contamination obvious
> rather than merely disappointing.

### Baseline accuracy

ResNet-18 fine-tuned for 8 epochs on a T4, reaching **96.94%** validation
accuracy. The curve was flat by epoch 7, so this is a converged baseline rather
than an undertrained one, which matters because a quantization accuracy drop
measured against an undertrained model would be confounded by training noise.

---
## Troubleshooting

### 1. torch reports no CUDA device on Colab

```
RuntimeError: cuda requested but torch reports no CUDA device.
```

and `nvidia-smi` fails separately with:

```
NVIDIA-SMI couldn't find libnvidia-ml.so library in your system.
```

**When does this happen?** In a shell opened with `colab ssh`, on a runtime that
genuinely has a GPU attached.

**Symptoms:** every CUDA package is installed, `/dev/nvidia0` exists, and
`torch.cuda.is_available()` still returns `False`. Re-running `uv sync --extra
cuda` changes nothing, which makes it look like a packaging problem.

**Why does this happen?** The GPU is passed through correctly, but the *driver*
libraries live in `/usr/lib64-nvidia`, which `ldconfig` does not index. A Colab
notebook kernel gets that directory injected into its environment; a shell
opened over SSH does not, and `LD_LIBRARY_PATH` is empty. The CUDA toolkit comes
from pip, but the driver comes from the host, and torch needs both.

**Solution:**

```bash
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH
nvidia-smi   # should now print the GPU table
```

> [!CAUTION]
> Point at `/usr/lib64-nvidia` specifically. There is also a
> `/usr/local/cuda-*/compat/` directory holding a shim for an older driver, and
> using that one produces a driver and library version mismatch rather than a
> clean failure.

### 2. Dataloader workers die with a BrokenPipeError

```
File ".../multiprocessing/popen_forkserver.py", line 58, in _launch
BrokenPipeError: [Errno 32] Broken pipe
```

**When does this happen?** Running a script that builds a `DataLoader` with
`num_workers` above zero, where the calling code is at module top level.

**Why does this happen?** Worker processes re-import the entry module. Windows
and macOS have always used `spawn`, and **as of Python 3.14 Linux defaults to
`forkserver`** rather than `fork`, because forking a process that already has
threads is unsafe. So every platform now re-imports, and unguarded top-level
code recurses.

**Solution:** put the entry point behind a guard.

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

The `edge` CLI already does this, so it only affects scripts you write yourself.

### 3. Thread pinning does not take effect

The bench command prints one of:

```
Warning: requested and effective thread counts differ.
Warning: the lowest thread count was not slower than the highest.
```

**Why does this happen?** `torch.set_num_threads` is called at runtime, and the
thread pool size can already be fixed by the time the first operation runs.
`OMP_NUM_THREADS` also takes precedence over it.

**Solution:** set the variable before the process starts, and run each thread
count in its own process:

```bash
OMP_NUM_THREADS=2 uv run edge bench --threads 2
```

> [!IMPORTANT]
> Do not ignore this warning and keep the numbers. Every constrained-core result
> depends on the cap actually applying, which is why the effective count is read
> back from torch and recorded in every row rather than echoed from the request.

---
## Tech Stack

| Purpose | Library |
|---|---|
| Model definitions and training | `torch`, `torchvision` |
| Results handling | `pandas` |
| Terminal output | `rich` |
| Linting and formatting | `ruff` |
| Tests | `pytest` |
| Git hooks | `prek` |
| Package management | `uv` |

---
## Reference Papers

Citations, licenses, and how each paper relates to this work are in
[`papers/README.md`](papers/README.md). The PDFs themselves are not committed,
since not all of them are ours to redistribute.
