# Results

Measurements to date. Every number here was produced by a command in this
repository and can be reproduced from it. Raw rows are in the CSVs alongside
this file.

**Setup.** ResNet-18 pretrained on ImageNet, fine-tuned on Imagenette, a
10-class ImageNet subset, at 160px. Trained on a Tesla T4. Every latency figure
measured on an HP ProBook 445 G10, Ryzen 5 7530U, 6 cores and 12 threads,
batch size 1, after discarding warmup runs, on AC power with the CPU free to
boost to 4547 MHz.

> [!NOTE]
> Latencies first recorded in September were re-measured in October and are
> replaced here. The frequency manager in use in September held the CPU at its
> 2 GHz base clock, and the same model at the same settings ran about 2.2x
> slower at one and two threads. Accuracy, size and operation counts were
> unaffected and reproduced exactly. Every latency row now records its power
> source and CPU frequency policy, so a cap like that is visible in the data.

---

## 1. Baseline model

| Measure | Value |
| --- | ---: |
| Validation accuracy | 96.89% |
| Parameters | 11,181,642 |
| Operations per image | 925.3 MMAC (1850.6 MFLOP) |
| Size on disk | 42.72 MiB |

Trained for a fixed 8 epochs, with the final epoch reported rather than the
best. Imagenette ships only `train` and `val`, so `val` doubles as the test
set, and selecting the peak epoch from it would inflate the reported figure.

Operations and parameters are hardware-independent. They are the quantity the
Angelucci et al. scheduler consumes, and they transfer unchanged to any device.
Latency never does.

---

## 2. Thread scaling

Forward pass latency, PyTorch, 50 timed runs after 10 discarded:

| threads | median ms | p95 ms | std ms |
| ------: | --------: | -----: | -----: |
|       1 |     25.10 |  25.99 |   0.39 |
|       2 |     14.54 |  15.62 |   0.51 |
|       4 |     18.61 |  19.92 |   0.70 |
|       6 |     13.73 |  15.19 |   0.76 |
|      12 |     16.97 |  21.30 |   2.04 |

**Beyond two threads, extra cores buy little.** Two threads already reach
14.54 ms and six only improve that by 6%, to 13.73 ms. A likely reason is that
with one or two cores busy the CPU boosts them higher than it can boost all
six, which offsets most of the extra parallelism.

**Twelve threads is worse than six, and the tail shows it most.** It
oversubscribes the six physical cores through SMT: the median rises to
16.97 ms, the spread grows about 2.7x and p95 goes from 15.19 to 21.30 ms. Two
threads sharing a core contend for the same vector units, which is essentially
all a convolution uses. For inference under a deadline the tail matters more
than the median, so this conclusion only exists because p95 is reported.

**The four thread result is slower than two** and does not fit either pattern.
It has reproduced in every sweep, in an isolated process, at base clock, at
full boost and on battery, so it is a genuine property of this CPU with this
model rather than a measurement artefact. The mechanism is unexplained and
recorded as measured.

---

## 3. INT8 quantization

Post-training static quantization through ONNX Runtime, calibrated on 256
training images. Both models evaluated on all 3925 validation images and timed
at 6 threads through identical code paths.

| precision | accuracy | size MiB | median ms | p95 ms |
| --------- | -------: | -------: | --------: | -----: |
| float32   |   96.89% |    42.73 |      4.34 |   4.46 |
| int8      |   96.41% |    10.83 |      2.47 |   2.53 |

**0.48 points of accuracy, 3.95x smaller, 1.76x faster.**

The size reduction is essentially the theoretical 4x from 32-bit to 8-bit.
Together with the accuracy drop it is hardware-independent, and both would hold
on a phone or a single-board computer.

The speedup is not, and it exceeded expectations. This CPU has AVX2 but no
VNNI, the instruction performing an 8-bit dot product in one step, so 1.0 to
1.3x was predicted. Moving a quarter as much data through cache mattered more
than the missing instruction did.

---

## 4. The runtime was a larger lever than the optimization

Same float32 model, same 6 threads, same 160px input:

| runtime            | median ms | against PyTorch |
| ------------------ | --------: | --------------: |
| PyTorch eager      |     13.73 |           1.00x |
| ONNX Runtime       |      4.34 |           3.16x |
| ONNX Runtime, INT8 |      2.47 |           5.56x |

**Changing the execution runtime alone was a 3.16x speedup**, nearly twice what
quantization added on top of it. PyTorch dispatches operations one at a
time through a Python-facing interpreter, while ONNX Runtime compiles the graph
ahead of time, fuses operations, and plans memory once.

This is not an argument against quantization, which still contributes a further
1.76x and the entire size reduction. It is an argument that runtime choice
deserves the same scrutiny as model optimization, and gets less of it.

---

## What these numbers are not

**The reference machine is a laptop, not an edge board.** It stands in for one,
and constrained thread counts are the approximation. A real edge SoC has
different cache sizes, memory bandwidth, and thermal behaviour, so this shows
scaling behaviour rather than a prediction for any particular device.

**Reproducibility is weaker than a fixed seed suggests.** Two runs of the
identical training command gave 96.94% and 96.89%. GPU arithmetic is not
bit-deterministic by default, and the dataloader worker count changes the
augmentation stream. Runs land within about a tenth of a point.

**Latency is never portable.** Every row records the machine, CPU, and thread
count it was measured under, so figures from different hardware cannot be
compared by accident.

---

## Next

Early exit. Intermediate classifiers after `layer2` and `layer3`, trained
jointly, giving per-exit accuracy and cumulative operations. Those two vectors
are what the Angelucci et al. controller treats the network as, and producing
them from real measurements is the point of this project.

After that, the two techniques combined: whether INT8 quantization shifts
*which* exit an image takes, by perturbing the confidence scores the exit
criterion depends on.
