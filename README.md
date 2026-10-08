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
    - [Export to ONNX](#export-to-onnx)
    - [Quantize to INT8](#quantize-to-int8)
    - [Train with early exits](#train-with-early-exits)
    - [Export the exit stages](#export-the-exit-stages)
    - [Quantize the exit stages](#quantize-the-exit-stages)
    - [Profile every exit](#profile-every-exit)
    - [Sweep confidence thresholds](#sweep-confidence-thresholds)
    - [Profile split payloads](#profile-split-payloads)
    - [Profile a server](#profile-a-server)
    - [Compare local, split and offload](#compare-local-split-and-offload)
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

| Vector | Meaning                                                  |
| ------ | -------------------------------------------------------- |
| `c`    | operations, in MOPs, required to reach each exit         |
| `a`    | accuracy at each exit, evaluated over the whole test set |

Those vectors are normally obtained by offline profiling, and a scheduler then
treats the network as a black box described by them. This repository produces
`c` and `a` from real measurements on real hardware, and studies how INT8
quantization changes them, including whether quantization shifts _which_ exit a
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
│   ├── exit_analysis.py    exit vectors and confidence-threshold sweeps
│   ├── exit_profiler.py    per-image latency and confidence at every exit
│   ├── export.py           ONNX export, per-stage export, parity checking
│   ├── models.py           backbone construction, early-exit wrapper
│   ├── offload.py          per-image cost of every mode, networks applied
│   ├── profiling.py        parameters, operations, cost per exit, size
│   ├── quantization.py     static INT8 quantization and calibration
│   ├── server.py           server-side time for every way a frame arrives
│   ├── split.py            what crosses the network at each split point
│   └── training.py         fine-tuning and evaluation, single or multi-exit
├── app/main.py             browser demo comparing float32 and INT8
├── tests/                  sanity checks
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

| Extra  | Build                   | Use on                                                      |
| ------ | ----------------------- | ----------------------------------------------------------- |
| `cpu`  | CPU-only, around 350 MB | measurement machines, where all reported latency comes from |
| `cuda` | CUDA, around 2.5 GB     | a training box with an NVIDIA GPU                           |

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
> Eight epochs takes roughly **33 minutes** on a six-core laptop CPU, measured
> at 52 images per second. If you have access to a GPU, see
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

### Export to ONNX

```bash
uv run edge export
```

Writes `checkpoints/resnet18_imagenette.onnx` and checks it against PyTorch:

```
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ check                ┃     value ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ samples              │         8 │
│ max absolute error   │ 3.483e-06 │
│ mean absolute error  │ 7.765e-07 │
│ tolerance            │     1e-04 │
│ prediction agreement │    100.0% │
└──────────────────────┴───────────┘
```

ONNX is an interchange format: a frozen description of a network's operations
and weights, independent of the framework that trained it. ONNX Runtime is the
engine used here, because it is the deployment path on both x86 and ARM and
because its static quantization tooling is what the next stage needs. Measuring
PyTorch latency and then quantizing with a different stack would compare two
different things.

Export is not lossless, which is why it is checked rather than trusted. The
exporter records the operations one run performs, so anything conditional can
be baked in silently, and operator implementations differ between engines.

Two comparisons are made, and the second decides the outcome. Raw outputs are
compared numerically, which catches an operator exported with different
semantics. Exact equality is not expected, since the engines use different
kernels and accumulation orders. Then the predicted classes are compared, which
is what actually determines accuracy: a large numerical difference that never
flips a prediction is a curiosity, a small one that does is a bug.

> [!IMPORTANT]
> A prediction disagreement exits non-zero. If the exported graph and PyTorch
> disagree about any class, every latency and accuracy number measured through
> that file afterwards would describe a model nobody evaluated. Do not proceed
> past a failed parity check.

### Quantize to INT8

```bash
uv run edge quantize
```

Calibrates, quantizes, then evaluates and times both models through the same
code paths, writing `results/quantization.csv`.

Quantization stores weights and activations as 8-bit integers instead of
32-bit floats. Three things follow, and only two are guaranteed:

- **The file shrinks about fourfold.** A property of the format, true anywhere.
- **Accuracy changes slightly**, because 256 integer levels cannot represent
  every float exactly. Also a property of the model, true anywhere.
- **Speed may or may not improve.** Entirely dependent on whether the CPU has
  instructions for 8-bit dot products.

_Post-training_ means quantizing a model that has finished training, with no
retraining. _Static_ means the value ranges are measured in advance from real
data rather than recomputed on every inference, which is what a deployed vision
model wants: the cost is paid once, offline.

**Calibration** is that measurement. A few hundred images are pushed through
the float model while the tooling records the range of values at each layer,
and those ranges decide how the float span maps onto 256 integer levels.

> [!NOTE]
> Calibration uses training images with **evaluation** preprocessing, not the
> augmented pipeline. It measures the range of values inference will actually
> see, and random crops would measure a distribution that never occurs at
> inference time. Calibrating on unrepresentative data produces an accuracy
> drop that looks like a quantization problem but is really a data problem.

### Train with early exits

```bash
uv run edge train --early-exit --epochs 8
```

Attaches classifier heads after `layer2` and `layer3` and trains all three
exits at once, with the loss a weighted average of each exit's cross-entropy.
Checkpoint and history go to `*_early_exit*` paths, so this never overwrites
the plain model it is compared against. Like the plain model, it is far faster
on a GPU: see [Training on a GPU machine](#training-on-a-gpu-machine).

The exits share the whole network body, so with equal weights the weak early
heads pull it toward themselves and the final exit ends up worse than a plain
model's. `--exit-weights` sets each exit's share of the loss, shallowest first.
`--exit-weights 0.3,0.3,1` favours the final exit, at the early exits' expense.

### Export the exit stages

```bash
uv run edge exits export
```

Writes the early-exit model as three ONNX graphs beside its checkpoint,
`<checkpoint>.stage1.onnx` to `stage3.onnx`, one per stage. Each runs its stage
and exit head and returns the exit's `logits` and the `features` the next stage
takes.

One graph per stage is what makes stopping possible. ONNX Runtime always runs
a graph to the end, so a single exported model would compute every exit for
every image. With separate stages, the application runs stage one, reads the
exit's confidence, and only then decides whether to run stage two. The same
boundary is where split computing would cut the network, and `features` is the
tensor that would cross it.

Parity is checked as for the plain model, with the stages chained the way they
run when deployed: each is fed the previous ONNX stage's output, not
PyTorch's, so an error at one stage shows up at every exit after it. Parity
results go to `results/onnx_parity_<checkpoint>.csv`.

### Quantize the exit stages

```bash
uv run edge exits quantize
```

Quantizes each stage graph to INT8 as `<checkpoint>.stage1.int8.onnx` and so
on, then evaluates float32 and INT8 at every exit over the validation set,
writing `results/quantization_<checkpoint>.csv`.

Only the first stage can be calibrated on images. Every later stage takes a
feature map, so it is calibrated on what the stage before it produces from the
same calibration images. Those features come from the float stage, which is
also what quantizing the whole network at once would use, so splitting into
stages changes where graphs end and not the ranges recorded.

### Profile every exit

```bash
uv run edge exits profile --runtime onnxruntime --precision int8 --threads 2
```

Runs each validation image through the early-exit model one stage at a time,
recording at every exit the cumulative latency, the top probability, entropy,
the prediction and whether it was correct. One row per image per exit, written
to `results/exit_profile_<checkpoint>_<runtime>_<precision>_t2.csv.gz`.
Compressed because profiles are measurements rather than derived files:
re-running one measures a different moment instead of reproducing it, so they
are committed.

`--runtime onnxruntime` runs the stage graphs from the two commands above,
which is how the model would be deployed. `--runtime pytorch`, the default,
runs the checkpoint directly and is about 1.7x slower at 2 threads. Both share
one timing loop, so the difference between their profiles is the runtime
rather than the harness. INT8 is profiled through ONNX Runtime only.

Every exit is always reached, so the table records what each exit would answer.
Any confidence threshold, or any other stopping rule, can then be applied
afterwards from the same measurements instead of being fixed before measuring.

`--limit N` profiles a seeded random sample of N images. It does not take the
first N: image folders are sorted by class, so the first few hundred validation
images are all one class.

### Sweep confidence thresholds

```bash
uv run edge exits sweep results/exit_profile_<...>.csv.gz
```

Applies a stopping rule to a recorded profile: each image stops at the first
exit whose top probability reaches the threshold, and the final exit always
answers. Thresholds run from 0.10 to 0.99, and each gives an accuracy, a mean,
median and p95 latency, and the share of images stopping at each exit. Writes
`exit_sweep_<...>.csv`, plus `exit_vectors_<...>.csv` with the `c` and `a`
vectors.

Nothing is re-measured, since the profile already holds every exit's answer for
every image. That is why the profile keeps all exits rather than stopping early.

### Profile split payloads

```bash
uv run edge split profile --cool-to 60
```

Every exit except the last is also a split point, where the device can stop and
send the feature map for a server to finish. For every image and split point
this records the feature map's size as float32 and as an 8-bit payload, each
before and after zstd compression, the device time to prepare it, whether the
8-bit version changes the final answer, and the JPEG size that full offload
would send instead. Written to `results/split_profile_<checkpoint>_<precision>_t2.csv.gz`.

`--cool-to 60` waits until the CPU is at or below 60 C before timing, and works
on all three profiling commands. A laptop drops from its boost clock within
seconds of sustained load, so without it a profile run straight after another
starts hotter and runs slower. Every row also records the CPU's actual clock and
temperature, so a run that was disturbed shows it.

### Profile a server

```bash
uv run edge server profile --runtime pytorch --device cuda --threads 2   # on a GPU machine
uv run edge server profile --runtime onnxruntime --threads 6 --cool-to 60  # CPU as server
```

Times what a server pays for every way a frame can arrive: decoding a JPEG for
full offload, or receiving a float32 or compressed 8-bit feature map at each
split point, then running the rest of the network. Receiving and computing are
timed separately, and on a GPU the timer waits for the work to finish rather
than for it to be queued. The server builds the payloads from the images
itself, outside the timer, so it needs only the checkpoint and the dataset.

On a free Colab runtime use `--threads 2`, matching its two vCPUs, which do
the JPEG decoding.

### Compare local, split and offload

```bash
uv run edge offload table
```

Joins the exit, split and server profiles of one checkpoint into a table with a
row per image for each of 16 ways of processing it, with every cost except the
network: `results/offload_components_<checkpoint>.csv.gz`. The network is left
out on purpose. Its cost is a round trip plus the payload over the uplink, so
any channel model can be applied to these rows afterwards, exactly.

For a first look, the command also evaluates four illustrative uplinks and
writes summaries: accuracy, median and p95 time and the share of frames
answered within 20, 50 and 100 ms per configuration, the bandwidth each way of
sending needs to match local inference, and how many times slower than the
profiled device a device must be before sending pays.

> [!IMPORTANT]
> The four networks (Wi-Fi, 5G, 4G and poor 4G) are assumptions, set in
> `offload.py`, not measurements. They are there to show the shape of the
> trade-off, and should be replaced by a measured or modelled channel before
> any conclusion depends on them.

---

## Training on a GPU machine

Training is the one stage that need not happen on the measurement machine,
because weights are identical wherever they are computed. Only latency is
hardware-specific.

The difference is large enough to matter:

| Machine                | Per epoch        | Eight epochs     |
| ---------------------- | ---------------- | ---------------- |
| Ryzen 5 7530U, 6 cores | about 4 minutes  | about 33 minutes |
| Free Colab T4          | about 30 seconds | about 4 minutes  |

Per-epoch figures are measured: 52 images per second on the laptop CPU, and 85
seconds for a three epoch run on the T4. The eight epoch totals follow from
those rates and include validation.

### Getting a Colab runtime

[Google Colab](https://colab.research.google.com) offers a free T4. Its official
CLI can create a runtime and open a shell on it, which beats working in notebook
cells: the repository is used as it actually is, and nothing depends on hidden
cell state.

- Install the CLI, once per machine:

    ```bash
    uv tool install 'google-colab-cli>=0.7.0'
    ```

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

- Check the GPU is visible:

    ```bash
    nvidia-smi
    ```

    It should print a table naming the GPU. If it instead complains about
    `libnvidia-ml.so`, see
    [torch reports no CUDA device](#1-torch-reports-no-cuda-device-on-colab)
    before going further.

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

    You do not need to install Python separately. Colab ships an older interpreter,
    but `.python-version` is committed and uv fetches 3.14 during the sync.

> [!IMPORTANT]
> `--extra cuda`, not `--extra cpu` and not a bare `uv sync`. A bare sync
> installs neither torch build, and `--extra cpu` installs a CPU-only torch that
> leaves the T4 idle while training runs at laptop speed, with no error to tell
> you.

- Download the dataset and train:

    ```bash
    uv run edge data prepare
    uv run edge train --device cuda --epochs 8 --batch-size 64 --num-workers 2
    ```

    Expected output ends with something like:

    ```
      epoch 8/8: train loss 0.1453, val loss 0.1000, val accuracy 96.89%
    Final validation accuracy: 96.89% after 8 epoch(s), which is what was saved
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
cores, 12 threads), on AC power with the CPU free to boost to 4547 MHz. 50 timed
runs after 10 discarded warmup runs:

| threads | median ms | p95 ms | std ms |
| ------: | --------: | -----: | -----: |
|       1 |     25.10 |  25.99 |   0.39 |
|       2 |     14.54 |  15.62 |   0.51 |
|       4 |     18.61 |  19.92 |   0.70 |
|       6 |     13.73 |  15.19 |   0.76 |
|      12 |     16.97 |  21.30 |   2.04 |

**Beyond two threads, extra cores buy little.** Six threads beat two by only 6%.
A likely reason is that with one or two cores busy the CPU boosts them higher
than it can boost all six, which offsets most of the extra parallelism.

**Twelve threads is worse than six, most visibly in the tail.** It
oversubscribes the six physical cores through SMT: the spread grows about 2.7x
and p95 goes from 15.19 to 21.30 ms. When two threads share a core they contend
for the same vector units, which is essentially all a convolution does. For
inference under a deadline, that loss of predictability matters more than the
median does.

**The four thread result is slower than two** and does not fit either pattern.
It has reproduced in every sweep, in an isolated process, at base clock, at full
boost and on battery, so it is a genuine property of this CPU with this model
rather than a measurement artefact. The mechanism is unexplained and is recorded
here as measured.

> [!IMPORTANT]
> Latencies first recorded in September were re-measured in October. The
> frequency manager in use in September held the CPU at its 2 GHz base clock,
> and the same model ran about 2.2x slower at one and two threads. Accuracy,
> size and operation counts were unaffected and reproduced exactly. Every
> latency row now records its power source and CPU frequency policy, so a cap
> like that shows up in the data instead of hiding in it.

> [!NOTE]
> These runs were taken on a machine under normal desktop load. A sweep taken
> while a browser spiked produced a 368 ms outlier at two threads, visible
> immediately as a standard deviation of 64 ms against a 33 ms median. Reporting
> the spread alongside the median is what makes such contamination obvious
> rather than merely disappointing.

### Baseline accuracy

ResNet-18 fine-tuned for 8 epochs on a T4, reaching **96.89%** validation
accuracy. Per-epoch metrics are in `results/training_history.csv`.

The curve had flattened by epoch 7, gaining 0.05 points over the last epoch
against 3 points over the first three. That matters because a converged
baseline is what quantization gets compared against later. An accuracy drop
measured against an undertrained model would be confounded by training that had
simply not finished.

> [!NOTE]
> A fixed seed here means a reproducible starting point, not identical output.
> An earlier run of the same command with the same seed reached 96.94%. GPU
> arithmetic is not bit-deterministic by default: cuDNN selects convolution
> algorithms by runtime heuristic, and some backward passes accumulate with
> atomics whose order varies. Changing `--num-workers` also reseeds the data
> loading processes, so the random crops and flips differ. Runs land within
> roughly a tenth of a point of each other, and results quote the run that
> produced the checkpoint on disk.

### INT8 quantization

Both models evaluated on all 3925 validation images and timed at 6 threads
through ONNX Runtime, using the same code paths so the comparison describes the
models rather than two harnesses:

| precision | accuracy | size MiB | median ms | p95 ms |
| --------- | -------: | -------: | --------: | -----: |
| float32   |   96.89% |    42.73 |      4.34 |   4.46 |
| int8      |   96.41% |    10.83 |      2.47 |   2.53 |

**0.48 points of accuracy, 3.95x smaller, 1.76x faster.**

The size reduction is essentially the theoretical 4x from 32-bit to 8-bit, and
along with the accuracy drop it is hardware-independent: both would hold
identically on a phone or a single-board computer.

The speedup is not, and it came out **higher than expected**. This CPU has AVX2
but no VNNI, the instruction that performs an 8-bit dot product in one step, so
the prediction was 1.0 to 1.3x. Moving a quarter as much data through cache
turned out to matter more than the missing instruction. Which is the argument
for measuring rather than reasoning from a datasheet.

### The runtime mattered more than the optimization

Worth putting beside the numbers above. The same float32 model, same 6 threads,
same 160px input:

| runtime            | median ms |
| ------------------ | --------: |
| PyTorch eager      |     13.73 |
| ONNX Runtime       |      4.34 |
| ONNX Runtime, INT8 |      2.47 |

**Changing the runtime alone was a 3.2x speedup**, nearly twice what
quantization then added on top. PyTorch dispatches operations one at a time
through a Python-facing interpreter; ONNX Runtime compiles the graph ahead of
time, fuses operations, and plans memory once.

For an edge deployment this says something the accuracy tables do not: choosing
the execution runtime can outweigh the model optimization technique applied to
it. Both are worth doing, but they are not the same size of lever.

### Early exit

Two early-exit models, trained with equal loss weights on all three exits and
with weights 0.3, 0.3 and 1. Accuracy at each exit, and median time to reach it
through ONNX Runtime at 2 threads. Times are the equal-weights model's; the
architecture is identical, and the weighted model is within 0.1 ms at every
exit:

| exit     | `c` MMAC | `a`, 1,1,1 | `a`, 0.3,0.3,1 | float32 ms | INT8 ms |
| -------- | -------: | ---------: | -------------: | ---------: | ------: |
| `layer2` |    505.9 |     62.11% |         49.81% |       4.58 |    2.71 |
| `layer3` |    715.6 |     80.48% |         70.65% |       6.60 |    3.81 |
| `layer4` |    925.3 |     95.44% |         96.89% |       9.11 |    5.10 |

**The first exit already costs 54.7% of the network**, so early exit can save
at most 45% of the operations. ResNet's four stages cost roughly the same, and
the first exit comes after the stem and two of them.

**Stopping on confidence cuts the mean, not the tail.** With a 0.55 threshold
the equal-weights model keeps 94.14% accuracy while its mean latency falls
20%, but p95 falls only 2.5%. Images that still run to the final exit set the
tail, and a deadline cares about the tail.

**INT8 was the larger lever.** The weighted model in INT8, with no early exit,
reaches 96.46% at 5.12 ms mean. Every float32 early-exit configuration above
72% accuracy is slower on average. INT8 also shifts which exit images take,
sending about one in ten to a deeper exit at the same threshold.

The full breakdown, including INT8 accuracy at every exit, is in
[`results/RESULTS.md`](results/RESULTS.md).

### Local, split or offload

With the laptop at 2 threads as the device, **running locally beat every way of
sending on every network tried**, even over Wi-Fi: offloading the JPEG to a
Colab T4 took 10.85 ms median against 9.11 ms locally in float32 and 5.10 ms in
INT8, because the round trip and the server's own 5 ms already exceed the
laptop's time.

**Splitting never won.** Even the smallest feature map payload, 10.8 KB with
INT8 stages, 8-bit values and zstd, is larger than the 7.9 KB median JPEG.

The transferable result is how much slower than this laptop a device must be
before sending pays, by median:

| network                  | against local float32 | against local INT8 |
| ------------------------ | --------------------: | -----------------: |
| Wi-Fi, 5 ms, 100 Mbps    |                 1.19x |              2.13x |
| 5G, 20 ms, 50 Mbps       |                 2.91x |              5.20x |
| 4G, 50 ms, 10 Mbps       |                 6.79x |             12.13x |
| poor 4G, 100 ms, 2 Mbps  |                15.13x |             27.04x |

The networks are illustrative assumptions, and the T4 was served through
eager PyTorch, which a production server would beat. Both caveats, and the
server measurements behind this, are in
[`results/RESULTS.md`](results/RESULTS.md).

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
genuinely has a GPU attached. Runtimes in September 2026 did this every time;
by October they put the driver on the path themselves, so it is unlikely now
but worth recognising if it returns.

**Symptoms:** every CUDA package is installed, `/dev/nvidia0` exists, and
`torch.cuda.is_available()` still returns `False`. Re-running `uv sync --extra
cuda` changes nothing, which makes it look like a packaging problem.

**Why does this happen?** The GPU is passed through correctly, but the _driver_
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

| Purpose                        | Library                             |
| ------------------------------ | ----------------------------------- |
| Model definitions and training | `torch`, `torchvision`              |
| Export and deployment runtime  | `onnx`, `onnxruntime`, `onnxscript` |
| Results handling               | `pandas`                            |
| Terminal output                | `rich`                              |
| Linting and formatting         | `ruff`                              |
| Tests                          | `pytest`                            |
| Git hooks                      | `prek`                              |
| Package management             | `uv`                                |

---

## Reference Papers

Citations, licenses, and how each paper relates to this work are in
[`papers/README.md`](papers/README.md). The PDFs themselves are not committed,
since not all of them are ours to redistribute.
