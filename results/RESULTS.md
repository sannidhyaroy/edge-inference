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

## 5. Early exit: the `c` and `a` vectors

Exit heads after `layer2` and `layer3`, trained jointly with the final exit on
a T4 for 8 epochs. Two runs, differing only in how the loss weights the three
exits:

| exit     |   `c` MMAC | share of network | `a`, weights 1,1,1 | `a`, weights 0.3,0.3,1 |
| -------- | ---------: | ---------------: | -----------------: | ---------------------: |
| `layer2` |      505.9 |            54.7% |             62.11% |                 49.81% |
| `layer3` |      715.6 |            77.3% |             80.48% |                 70.65% |
| `layer4` |      925.3 |             100% |             95.44% |                 96.89% |

**ResNet front-loads its compute.** The first exit already costs 54.7% of the
network, so no stopping rule can save more than 45% of the operations, however
easy the inputs.

**Equal weights cost the final exit 1.45 points.** The exits share the network
body, and weak early heads pull it toward themselves. Weighting the final exit
more restores it exactly to the plain model's 96.89%, at a cost of 12.30 and
9.83 points at the early exits. Neither run dominates the other, so both are
carried through every measurement below.

**The early heads are underconfident.** Mean top probability at the first exit
is 0.32 at 62% accuracy with equal weights, and 0.16 at 50% with the weighted
run. A confidence threshold calibrated by intuition would almost never let them
answer, which is why the sweep in section 8 starts at 0.10.

---

## 6. INT8 at every exit

Each stage quantized separately and evaluated chained, as deployed, on all 3925
validation images. INT8 MiB is the weights a device stopping at that exit must
hold:

| exit     | 1,1,1 float32 | 1,1,1 INT8 |  drop | 0.3 float32 | 0.3 INT8 |  drop | INT8 MiB |
| -------- | ------------: | ---------: | ----: | ----------: | -------: | ----: | -------: |
| `layer2` |        62.11% |     56.79% |  5.32 |      49.81% |   47.95% |  1.86 |     0.72 |
| `layer3` |        80.48% |     78.11% |  2.37 |      70.65% |   68.69% |  1.96 |     2.77 |
| `layer4` |        95.44% |     93.81% |  1.63 |      96.89% |   96.46% |  0.43 |    10.85 |

**Quantizing stage by stage costs nothing.** The weighted run's final exit
loses 0.43 points, against 0.48 for the plain model quantized whole.

**The equal-weights model is far more sensitive to INT8**, losing 5.32 points
at its first exit. Both models went through the same stages and the same
calibration, so the difference lies in the trained weights, not the method.
Why is not yet established.

---

## 7. Latency at every exit

Median time to reach each exit, equal-weights model, 2 threads, all 3925
images, on AC power:

| exit     | PyTorch ms | ONNX Runtime ms | ONNX Runtime INT8 ms |
| -------- | ---------: | --------------: | -------------------: |
| `layer2` |       7.16 |            5.03 |                 3.08 |
| `layer3` |      10.02 |            7.25 |                 4.29 |
| `layer4` |      15.46 |            9.94 |                 5.68 |

The ONNX Runtime figures are the deployed path: each stage is its own graph,
and the exit decision happens between graphs.

**Time tracks operations roughly, not exactly.** The first exit takes 50.5% of
the full float32 time for 54.7% of the operations.

**ONNX Runtime's lead over PyTorch shrinks at fewer threads**: 1.56x here at 2
threads, against 3.16x at 6 threads in section 4. INT8 adds a further 1.75x,
consistent with the 1.76x measured on the plain model.

---

## 8. Confidence thresholds

Each image stops at the first exit whose top probability reaches the threshold,
and the final exit always answers. Applied to the ONNX Runtime profiles of the
equal-weights model:

| threshold          | float32 accuracy | mean ms | p95 ms | INT8 accuracy | mean ms | p95 ms |
| ------------------ | ---------------: | ------: | -----: | ------------: | ------: | -----: |
| 0.30               |           83.16% |    6.87 |   9.99 |        80.66% |    4.08 |   5.85 |
| 0.45               |           91.95% |    7.76 |  10.52 |        89.45% |    4.59 |   6.01 |
| 0.55               |           94.14% |    8.17 |  10.62 |        92.00% |    4.84 |   6.06 |
| 0.75               |           95.36% |    8.97 |  10.76 |        93.68% |    5.28 |   6.13 |
| none, final exit   |           95.44% |   10.12 |  10.94 |        93.81% |    5.74 |   6.19 |

**Stopping on confidence cuts the mean, not the tail.** At 0.55 the float32
mean falls 19%, from 10.12 to 8.17 ms, while p95 falls 3%. The images that
still run to the final exit set the tail, and under a deadline the tail is what
decides whether a task completes. Capping latency needs the exit chosen from
the time remaining, as a state-driven controller does.

**Quantization changes which exit an image takes.** At 0.55, INT8 moves 13.1%
of images to a different exit: 10.6% go deeper, mostly from the second exit to
the third, and 2.6% stop earlier. The second exit's mean confidence drops from
0.62 to 0.57, so fewer images clear the threshold there, and the mean saving
from early exit shrinks from 19% to 16%.

**The weighted run's exits barely fire.** At 0.55 only 2.3% of its images stop
at the first exit and 1.6% at the second, because its heads are so
underconfident. Confidence thresholds give it almost nothing.

**INT8 was the larger lever.** The weighted run in INT8, with no early exit at
all, reaches 96.46% at 6.02 ms mean and 6.57 ms p95. Every float32 early-exit
configuration of either model above 72% accuracy is slower on average. The
only configurations more accurate are the weighted run's own float32 ones, by
at most 0.43 points, at 1.7x the latency. Early exit still matters where the
exit is chosen for a deadline or an offloading decision rather than by
confidence, which is the setting the next measurements address.

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

Offloading. Every result above is local inference. Next comes the time to run
the same stages on a server, the size of what would cross the network at each
split point, and a simulated wireless link, so local, split and offloaded
inference can be compared per image on the same footing.
