# Results

Measurements to date. Every number here was produced by a command in this
repository and can be reproduced from it. Raw rows are in the CSVs alongside
this file.

**Setup.** ResNet-18 pretrained on ImageNet, fine-tuned on Imagenette, a
10-class ImageNet subset, at 160px. Trained on a Tesla T4. Every latency figure
measured on an HP ProBook 445 G10, Ryzen 5 7530U, 6 cores and 12 threads,
batch size 1, after discarding warmup runs.

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
| ---: | ---: | ---: | ---: |
| 1 | 55.39 | 56.55 | 0.70 |
| 2 | 30.26 | 31.38 | 0.53 |
| 4 | 38.90 | 40.93 | 0.94 |
| 6 | 24.07 | 29.24 | 1.95 |
| 12 | 25.34 | 40.63 | 6.44 |

**Six threads, one per physical core, is the best operating point.** Twelve
oversubscribes those cores through SMT and gains nothing useful: the median
barely moves while the spread roughly triples and p95 rises from 29.2 to
40.6 ms. Two threads sharing a core contend for the same vector units, which is
essentially all a convolution uses.

For inference under a deadline the tail matters more than the median, so this
conclusion only exists because p95 is reported.

**The four thread result is slower than two** and does not fit that pattern. It
reproduces across separate sweeps and in an isolated process, so it is a
genuine property of this CPU with this model rather than a measurement
artefact. The mechanism is unexplained and recorded as measured.

---

## 3. INT8 quantization

Post-training static quantization through ONNX Runtime, calibrated on 256
training images. Both models evaluated on all 3925 validation images and timed
at 6 threads through identical code paths.

| precision | accuracy | size MiB | median ms | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| float32 | 96.89% | 42.73 | 6.46 | 7.10 |
| int8 | 96.41% | 10.83 | 3.88 | 4.43 |

**0.48 points of accuracy, 3.95x smaller, 1.67x faster.**

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

| runtime | median ms | against PyTorch |
| --- | ---: | ---: |
| PyTorch eager | 24.07 | 1.00x |
| ONNX Runtime | 6.46 | 3.72x |
| ONNX Runtime, INT8 | 3.88 | 6.20x |

**Changing the execution runtime alone was a 3.72x speedup**, more than twice
what quantization added on top of it. PyTorch dispatches operations one at a
time through a Python-facing interpreter, while ONNX Runtime compiles the graph
ahead of time, fuses operations, and plans memory once.

This is not an argument against quantization, which still contributes a further
1.67x and the entire size reduction. It is an argument that runtime choice
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
