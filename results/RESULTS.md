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
images, on AC power. The ONNX Runtime runs started from a CPU cooled to 60 C and
record its clock and temperature on every row; under this sustained load the
busy cores settle near 4.05 GHz:

| exit     | PyTorch ms | ONNX Runtime ms | ONNX Runtime INT8 ms |
| -------- | ---------: | --------------: | -------------------: |
| `layer2` |       7.16 |            4.58 |                 2.71 |
| `layer3` |      10.02 |            6.60 |                 3.81 |
| `layer4` |      15.46 |            9.11 |                 5.10 |

The ONNX Runtime figures are the deployed path: each stage is its own graph,
and the exit decision happens between graphs.

> [!NOTE]
> These replace ONNX Runtime figures first recorded with one thread pool per
> stage. A pool's threads keep spinning briefly after a run, so each stage
> competed with the previous stage's idle threads and every early-exit timing
> came out inflated. The stages now share one pool, which brings them within 3%
> of the same network exported as a single graph. With that fix and each run
> starting cool, the final exit went from 9.94 to 9.11 ms. Accuracy and
> confidence were never affected.

**Time tracks operations roughly, not exactly.** The first exit takes 50.3% of
the full float32 time for 54.7% of the operations.

**ONNX Runtime's lead over PyTorch shrinks at fewer threads**: 1.70x here at 2
threads, against 3.16x at 6 threads in section 4. The PyTorch profile predates
the cool-down and clock recording, so treat that ratio as approximate. INT8 adds
a further 1.79x, close to the 1.76x measured on the plain model.

---

## 8. Confidence thresholds

Each image stops at the first exit whose top probability reaches the threshold,
and the final exit always answers. Applied to the ONNX Runtime profiles of the
equal-weights model:

| threshold          | float32 accuracy | mean ms | p95 ms | INT8 accuracy | mean ms | p95 ms |
| ------------------ | ---------------: | ------: | -----: | ------------: | ------: | -----: |
| 0.30               |           83.16% |    6.15 |   9.16 |        80.66% |    3.64 |   5.12 |
| 0.45               |           91.95% |    6.95 |   9.29 |        89.45% |    4.10 |   5.29 |
| 0.55               |           94.14% |    7.33 |   9.36 |        92.00% |    4.33 |   5.35 |
| 0.75               |           95.36% |    8.06 |   9.48 |        93.68% |    4.73 |   5.43 |
| none, final exit   |           95.44% |    9.11 |   9.60 |        93.81% |    5.15 |   5.49 |

**Stopping on confidence cuts the mean, not the tail.** At 0.55 the float32
mean falls 20%, from 9.11 to 7.33 ms, while p95 falls 2.5%. The images that
still run to the final exit set the tail, and under a deadline the tail is what
decides whether a task completes. Capping latency needs the exit chosen from
the time remaining, as a state-driven controller does.

**Quantization changes which exit an image takes.** At 0.55, INT8 moves 13.1%
of images to a different exit: 10.6% go deeper, mostly from the second exit to
the third, and 2.6% stop earlier. The second exit's mean confidence drops from
0.62 to 0.57, so fewer images clear the threshold there, and the mean saving
from early exit shrinks from 20% to 16%.

**The weighted run's exits barely fire.** At 0.55 only 2.3% of its images stop
at the first exit and 1.6% at the second, because its heads are so
underconfident. Confidence thresholds give it almost nothing.

**INT8 was the larger lever.** The weighted run in INT8, with no early exit at
all, reaches 96.46% at 5.12 ms mean and 5.45 ms p95. Every float32 early-exit
configuration of either model above 72% accuracy is slower on average. The
only configurations more accurate are the weighted run's own float32 ones, by
at most 0.43 points, at 1.8x the latency. Early exit still matters where the
exit is chosen for a deadline or an offloading decision rather than by
confidence, which is the setting the next measurements address.

---

## 9. What crosses the network when splitting

Every exit but the last is also a split point: the device runs the network up to
it and sends the feature map for a server to finish. Median payload per image,
equal-weights model, in decimal kilobytes:

| split after |  float32 | + zstd | 8-bit | + zstd |
| ----------- | -------: | -----: | ----: | -----: |
| `layer2`    |    204.8 |  114.3 |  51.2 |   29.5 |
| `layer3`    |    102.4 |   52.7 |  25.6 |   13.3 |

Full offload sends the JPEG instead: **7.9 KB** median, 12.0 KB at p95.

**Naive splitting sends more than offloading.** Even the smallest payload
measured, 10.8 KB after `layer3` from INT8 device stages sent as 8 bits and
compressed, is larger than the median JPEG. A feature map holds fewer values
than the image, 51,200 or 25,600 against 76,800, but JPEG is lossy coding built
around how images look, and nothing comparable exists for feature maps.
Splitting a standard ResNet only pays with a trained bottleneck that shrinks
the feature map, which is what the split computing literature adds.

**An 8-bit payload is nearly free.** Each feature map is mapped onto 256 levels
using its own range, with no calibration. Across all four split profiles it
changed one prediction in 3925. For the equal-weights model, quantizing and
compressing costs the device a median 0.19 to 0.40 ms.

**About half of each feature map is zero**, 45% after `layer2` and 49% after
`layer3`, left by the ReLU at the end of each block. That is what zstd exploits.
Feature maps from INT8 stages compress far better even as float32, 42.8 KB
against 114.3 KB after `layer2`, because their values take only 256 distinct
levels.

---

## 10. Server time

Median time per image on the server, from bytes arriving to an answer, for the
equal-weights model. *Receive* is decoding the JPEG with the same preprocessing
the device uses, or decompressing and dequantizing a feature map:

| arrives as                | laptop CPU, receive | compute | total |  p95 | T4, receive | compute | total |   p95 |
| ------------------------- | ------------------: | ------: | ----: | ---: | ----------: | ------: | ----: | ----: |
| JPEG                      |                1.47 |    4.79 |  6.27 | 7.12 |        1.95 |    3.20 |  5.17 | 20.14 |
| float32 after `layer2`    |                0.03 |    2.45 |  2.48 | 3.00 |        0.03 |    1.86 |  1.89 |  4.05 |
| 8-bit zstd after `layer2` |                0.18 |    2.40 |  2.57 | 3.14 |        0.22 |    1.88 |  2.11 |  4.36 |
| float32 after `layer3`    |                0.02 |    1.31 |  1.33 | 1.73 |        0.02 |    1.16 |  1.19 |  1.87 |
| 8-bit zstd after `layer3` |                0.11 |    1.25 |  1.36 | 1.77 |        0.14 |    1.18 |  1.32 |  2.55 |

The laptop CPU server is ONNX Runtime on 6 threads, sustaining about 3.39 GHz.
The GPU server is a free Colab Tesla T4 through PyTorch, decoding JPEGs on the
runtime's 2 vCPUs.

**The T4 is barely faster than the laptop for this network.** The network alone
runs 1.5x faster, 3.20 against 4.79 ms. One image at a time, eager PyTorch
issues each layer as its own GPU launch, and a network as small as ResNet-18
spends most of its time on launch overhead rather than arithmetic. A server
built with CUDA graphs or TensorRT would do better; this is the simple setup.

**The free T4's tail belongs to the shared machine.** Its JPEG p95 is 20.14 ms
against a 5.17 ms median, mostly in decoding on the shared vCPUs, and it comes in
bursts: the p95 of successive 400-image windows alternates between about 6 and
25 ms, at different moments in each run. That is the draft's server-load
uncertainty appearing on its own. The VM exposes no CPU clock or temperature, so
those columns are empty for it.

---

## 11. Local, split or offload

Each image's end-to-end time per way of processing it is its measured device,
preparation and server time plus a network: a round trip, and the payload
over the uplink. The networks are **illustrative assumptions**, not
measurements, chosen to span a range. The per-image components are stored
without any network, in `offload_components_*.csv.gz`, so any channel model can
replace them. Equal-weights model, laptop at 2 threads as the device:

| way of processing                    | accuracy | Wi-Fi median | 5G median | 4G median | poor 4G median |
| ------------------------------------ | -------: | -----------: | --------: | --------: | -------------: |
| local, float32                       |   95.44% |         9.11 |      9.11 |      9.11 |           9.11 |
| local, INT8                          |   93.81% |         5.10 |      5.10 |      5.10 |           5.10 |
| local INT8, exit 1                   |   56.79% |         2.71 |      2.71 |      2.71 |           2.71 |
| split after `layer3`, 8-bit zstd, T4 |   95.41% |        14.20 |     30.27 |     68.84 |         161.34 |
| offload to T4                        |   95.44% |        10.85 |     26.54 |     61.88 |         137.88 |
| offload to laptop CPU server         |   95.44% |        11.95 |     27.61 |     62.77 |         138.29 |

Wi-Fi is 5 ms round trip and 100 Mbps uplink, 5G 20 ms and 50 Mbps, 4G 50 ms
and 10 Mbps, poor 4G 100 ms and 2 Mbps.

**With this laptop as the device, running locally always wins.** Even over
Wi-Fi, offloading takes 10.85 ms against 9.11 ms locally in float32 and 5.10 ms
in INT8: the round trip and the server's own 5 ms already exceed what the
laptop needs. Offloading the JPEG is the fastest way of sending on every
network, and splitting never wins, as section 9 predicts.

**The real question is how slow the device is.** Sending pays only for a device
this many times slower than the laptop, by median:

| network | against local float32 | against local INT8 |
| ------- | --------------------: | -----------------: |
| Wi-Fi   |                 1.19x |              2.13x |
| 5G      |                 2.91x |              5.20x |
| 4G      |                 6.79x |             12.13x |
| poor 4G |                15.13x |             27.04x |

A boosting laptop CPU is far faster than an in-vehicle board or a single-board
computer, so the factors, not the raw comparison, are the transferable result.
They assume a slower device is slower by a constant factor, which holds only
roughly.

**Deadlines bring the tail back.** Over Wi-Fi the T4 has the better median,
10.85 against 11.95 ms for the laptop server, but answers 94.2% of frames within
20 ms against the laptop server's 99.6%, because of its shared-machine tail. A
deadline-driven controller would prefer the slower, steadier server.

The split rows use float32 device stages only. A feature map from INT8 stages
is smaller, but how a float32 server answers it has not been measured.

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

A slower device. The offloading question turns on how much slower than this
laptop the edge device is, so the next measurement caps the laptop's clock and
cores to approximate a Raspberry Pi 5-class board. After that, distorted inputs
(lighting, blur, weather), which the draft names as an uncertainty and which
should push images to later exits, and a harder dataset.
