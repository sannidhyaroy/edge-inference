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

## In brief

Sections 1 to 11 measure what each way of running the network costs. Sections
12 to 17 use those measurements to decide between them. *Confirms* marks a
result already known in general and measured here. Everything else is specific
to this setup and is reported as such.

**What each option costs**

- **The runtime was a larger lever than quantization:** 3.16x from PyTorch to
  ONNX Runtime, and a further 1.76x from INT8 (sections 3 and 4).
- **Early exit by confidence cuts the mean, not the tail:** 20% against 2.5% at
  a threshold of 0.55. INT8 with no early exit was faster than every float32
  early-exit configuration above 72% accuracy (section 8).
- **A feature map is a bigger payload than the photo:** 10.8 KB at best against
  7.9 KB (section 9). *Confirms* what the split computing literature reports
  for networks without a trained bottleneck (Kang et al., 2017; Matsubara et
  al., 2022).
- **One image at a time, a T4 barely beats a laptop CPU as the server:** 5.17
  against 6.27 ms, with a worse tail (section 10).

**When to send, and how to decide**

- **Sending pays only for a slow device on a good network.** On the laptop it
  never does. For a device 8 times slower it wins 8 of 16 scenarios. Splitting
  wins none of 160 (section 12).
- **Watching the network is worth a lot, in a narrow band.** In 3 of 16
  device-and-deadline cells, switching with the network answers 6 to 20 points
  more frames correctly in time than the best fixed option. In the other 13
  the two are within a point (sections 13 and 15).
- **A published learning rule, SplitEE, does not get there on these costs.** As
  published it answers no frame in time in 5 of the 16 cells, because its menu
  holds only float32 exits. Given the full menu it settles near the best fixed
  option, because it never looks at the network (section 14).
- **A table lookup built from the measurements reaches the fair target,**
  within 0.2 points in all 16 cells, when it is told the network. It needs a
  measuring phase on labelled images, which SplitEE does not (section 15).
- **Carried out between two processes, the ranking holds:** the lookup rule
  scores 72.32% against 73.50% looked up (section 16).
- **Two real Wi-Fi links to an internet server behaved like the assumed
  degraded network,** never like the good one (section 17).

Read "What these numbers are not" before quoting any of these.

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
25 ms, at different moments in each run. That is uncertainty in server load
appearing on its own. The VM exposes no CPU clock or temperature, so
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

## 12. Where sending pays

Section 11 compared medians on one device. A deployment asks something else:
for a given device, network and deadline, which single way of processing
answers the most frames **correctly and in time**? That share is called
*useful* from here on.

A device `k` times slower than the laptop is modelled by multiplying the
laptop's measured device time by `k`. Server time and payload sizes stay as
measured. Five device speeds, the four networks of section 11 and four
deadlines give 80 scenarios per server, with 11 ways of processing competing in
each (`decide_map_*.csv`, `decide_options_*.csv`).

How many of the 16 network-and-deadline scenarios sending the photo wins,
equal-weights model:

| device         | T4 server | laptop CPU server |
| -------------- | --------: | ----------------: |
| 1x, the laptop |         0 |                 0 |
| 2x slower      |         2 |                 3 |
| 4x slower      |         4 |                 7 |
| 8x slower      |         8 |                 8 |
| 16x slower     |         9 |                 9 |

The winner and its useful share for the device 8 times slower, T4 server:

| deadline | 5 ms, 100 Mbps | 20 ms, 50 Mbps     | 50 ms, 10 Mbps      | 100 ms, 2 Mbps        |
| -------- | -------------- | ------------------ | ------------------- | --------------------- |
| 20 ms    | send, 89.76%   | INT8 exit 1, 0.69% | INT8 exit 1, 0.69%  | INT8 exit 1, 0.69%    |
| 30 ms    | send, 91.06%   | send, 85.91%       | INT8 exit 1, 56.79% | INT8 exit 1, 56.79%   |
| 50 ms    | send, 95.41%   | send, 94.55%       | local INT8, 93.73%  | local INT8, 93.73%    |
| 100 ms   | send, 95.41%   | send, 95.41%       | send, 95.41%        | local float32, 95.44% |

**The best choice depends on the network only inside a band of deadlines.** At
30 ms this device should send on the best network, for 91.06%, and stop at its
first INT8 exit on the third, for 56.79%. At 100 ms sending wins on three
networks of four and the choice barely matters. At 20 ms nothing local fits at
all: the first INT8 exit takes 21.7 ms on this device.

**Splitting wins none of the 160 scenarios,** for either model. No split option
is the best in its scenario in any of 640 entries. Options within 0.1 points
of the top count as tied and the faster one wins, which matters here: section
15 shows a split coming out ahead by a few frames in ten thousand when the
deadline is loose.

**Early exit wins only where nothing else fits.** It is the best option in 14
of the 80 scenarios per server, always in INT8. In 7 of them it answers a
useful 56% to 78% of frames: the deadline is too tight to run INT8 to the end
and the network too slow to send. In the other 7 it answers under 1% and is
only the fastest way of being late, as is one of the 9 sending wins at 16x.

**The steadier server wins the tight deadlines.** For the 8x device over the
best network at 20 ms, the laptop CPU server gives 95.08% useful and the T4
89.76%. That is the T4's shared-machine tail from section 10, and it is why the
laptop server wins 7 scenarios at 4x where the T4 wins 4.

The weighted model gives the same counts with the T4 and 0, 2, 6, 8 and 9 with
the laptop server.

---

## 13. A changing network and four yardsticks

Section 12 holds the network still. Here the measurements are replayed as a
stream of frames while the network changes underneath (`edge decide evaluate`).
Nothing is run again: each frame is one of the profiled images, and its device
time, confidence at every exit, payload size and server time are looked up.

**The setup.** The network has two states and each frame stays in the current
one with probability 0.95, so a state lasts 20 frames on average. The good
state is 5 ms and 100 Mbps, the degraded one 50 ms and 10 Mbps: the "Wi-Fi" and
"4G" rows of section 11, which are assumed numbers and not measurements.
Half of each class's images, 1963, are for choosing settings. The other 1962
are the test, shown five times in a seeded random order: 9810 frames. A rule
is never shown a label.

**Four yardsticks, each allowed to know a little more:**

| yardstick                           | allowed to know                        |
| ----------------------------------- | -------------------------------------- |
| 1. best fixed option                | nothing: one option for the whole run  |
| 2. best option per network          | the network, not the image             |
| 3. best confidence rule per network | the network and each exit's confidence |
| 4. best possible per frame          | which options answer correctly         |

The third is the **fair target**: no rule that decides from the network and
confidence can beat it. The fourth needs the answer in advance, so nothing can
reach it. All four are found afterwards on the test frames themselves, which
makes them optimistic by construction.

The default scenario: a device 8 times slower, a 30 ms deadline, the T4 server.

| rule or yardstick                   | useful | accuracy | in time | mean ms |
| ----------------------------------- | -----: | -------: | ------: | ------: |
| always local, float32               |  0.00% |   95.67% |   0.00% |   72.85 |
| always local, INT8                  |  0.00% |   94.34% |   0.00% |   41.23 |
| always send the photo               | 45.43% |   95.67% |  47.36% |   37.91 |
| early exit in INT8, one threshold   | 56.68% |   56.68% | 100.00% |   22.01 |
| 1. best fixed option                | 56.68% |   56.68% | 100.00% |   22.01 |
| 2. best option per network          | 73.75% |   76.06% |  97.64% |   17.25 |
| 3. best confidence rule per network | 73.75% |   76.06% |  97.64% |   17.25 |
| 4. best possible per frame          | 77.74% |   77.74% | 100.00% |   17.19 |

The accuracies are those of the test half, which differ slightly from the
whole-set figures of section 6.

**Watching the network is worth 17.07 points here.** No fixed option can do
better than stop at the first INT8 exit on every frame, for 56.68%. Sending on
the good network and stopping early on the degraded one gives 73.75%.

**The tuned early-exit rule is not an early-exit rule.** Its threshold came out
at 0.05, so it answers at the first exit every time. With a deadline this
tight there is no time to go deeper.

**Not knowing the answer costs 3.99 points,** the gap from the third yardstick
to the fourth. On the degraded network only the first INT8 exit fits, and it
is right 57% of the time. No rule closes that.

The same four yardsticks in three other scenarios:

| device, deadline | 1. fixed | 2. per network | 3. fair target | 4. per frame |
| ---------------- | -------: | -------------: | -------------: | -----------: |
| 8x, 30 ms        |   56.68% |         73.75% |         73.75% |       77.74% |
| 8x, 50 ms        |   94.29% |         94.90% |         94.92% |       96.65% |
| 4x, 30 ms        |   94.34% |         94.34% |         94.39% |       96.71% |
| 4x, 50 ms        |   95.67% |         95.67% |         95.70% |       97.25% |

**Outside the first row there is almost nothing to decide.** When some fixed
option fits the deadline on both networks, it is within 0.7 points of the fair
target.

**Confidence adds at most 0.05 points to this score.** A deadline pays nothing
for finishing early, so the third yardstick barely moves above the second.
What confidence buys is time: at 4x and 30 ms the third yardstick averages
19.03 ms against 20.62 ms for the second, and at 8x and 50 ms 25.05 against
26.91 ms.

---

## 14. A published rule on measured costs

SplitEE (Bajpai et al., 2023) is a rule from the literature for exactly this
choice. It treats each exit as an arm of a bandit: run to that exit, answer if
it is confident, otherwise send the feature map. It learns which arm pays best
while running, from a reward of confidence minus cost, with the cost counted in
layers and one constant charged for sending. It needs no labels and no
measuring phase.

It was ported from the paper's algorithm, checked against the authors' code,
and run on the measurements of this repository, first as published and then
with one assumption changed at a time. Useful share on the test frames:

| variant                            | 8x, 30 ms | 8x, 50 ms | 4x, 30 ms | 4x, 50 ms |
| ---------------------------------- | --------: | --------: | --------: | --------: |
| as published                       |     0.00% |    59.91% |    61.10% |    95.43% |
| a threshold per exit               |     0.00% |    59.57% |    61.13% |    95.48% |
| rewarded with the right answer     |     0.00% |    60.60% |    61.50% |    94.64% |
| measured time as the cost          |     0.00% |    61.99% |    79.87% |    95.64% |
| measured cost, sliding window      |     0.00% |    60.95% |    75.99% |    94.43% |
| measured cost, INT8 and photo arms |    56.60% |    78.74% |    93.91% |    94.27% |
| every repair at once               |    56.69% |    71.16% |    92.69% |    92.41% |
| *1. best fixed option*             |    56.68% |    94.29% |    94.34% |    95.67% |
| *3. fair target*                   |    73.75% |    94.92% |    94.39% |    95.70% |

The published arms are float32 exits that fall back on the feature map. The
wider arms are INT8 exits that fall back on the photo, plus an arm that sends
the photo at once. "Rewarded with the right answer" replaces confidence with
correctness in the reward, which no device could do.

**This comparison favours the published rule in one way and handicaps it in
another.** Every variant's settings were chosen by running it on the tuning
images and keeping whichever answered the most frames correctly in time, which
hands it a goal its own reward does not contain. Against that, it is scored
here on meeting a deadline, which is not what it was designed to optimise, and
on an image network, where it was published for a text model.

**The menu is the main handicap.** At 8x and 30 ms no float32 option fits:
even the first float32 exit takes 36 ms. A rule that can choose nothing else
scores 0.00%. With INT8 arms the same rule scores 56.60%.

**With the full menu it settles near the best fixed option, and no higher.**
56.60% against 56.68% at 8x and 30 ms, and 93.91% against 94.34% at 4x and
30 ms. It keeps one favourite arm and never looks at the network, so the 17
points of section 13 are out of its reach. That matches its own measure of
success: its regret is counted against the best single arm.

**Its reward cannot express a deadline.** At 8x and 50 ms the full-menu
variant settles on the second INT8 exit and scores 78.74%, although running
INT8 to the end scores 94.29%. The reward charges steadily for time, while a
frame under a deadline is worth the same until 50 ms and nothing after. The
rule saves time it had no need to save.

**Two assumptions expected to matter did not.** One threshold for all exits,
and confidence standing in for correctness, each move the score by under a
point in every scenario.

**Forgetting helps only when the network changes slowly.** When a state lasts
1000 frames in place of 20, the variant with every repair reaches 70.22% at 8x
and 30 ms, above the best fixed option and 4.3 points short of that run's
fair target of 74.50%. At 20 frames per state the window adds 0.09 points in
that scenario and costs 1 to 8 points in the other three.

**A constant cost for sending does not describe these measurements.** The
paper's experiments charge sending a constant of between 1 and 5 layers'
worth. Measured, sending
the feature map after the first exit costs this many layers of device work:

| device    | 5 ms, 100 Mbps | 50 ms, 10 Mbps |
| --------- | -------------: | -------------: |
| 8x slower |            2.9 |           18.4 |
| 4x slower |            5.1 |           36.1 |

The range fits a slow device on a good network. It is six to seven times
larger on the degraded one, and it grows as the device gets faster, since a
layer of device work is then worth less time.

SplitEE-S, the paper's variant that uses side information, was not ported.

---

## 15. Measure, then decide

This project's own rule does what the measurements make possible. For each
network it knows about, it tries every plan on the tuning images and keeps the
one with the most useful frames, or the fastest among those within 0.1 points.
A plan is a precision, a depth, a confidence threshold per exit, and what to do
if no exit is confident: answer anyway, send the feature map, or send the
photo. There are 2523 of them. On each frame the rule takes the network as it
last saw it and plays that network's plan, so deciding is a table lookup.

With exact knowledge of the network, on the scenarios of section 13:

| device, deadline | this rule | 1. best fixed | 3. fair target | mean ms | best fixed, mean ms |
| ---------------- | --------: | ------------: | -------------: | ------: | ------------------: |
| 8x, 30 ms        |    73.75% |        56.68% |         73.75% |   17.25 |               22.01 |
| 8x, 50 ms        |    94.83% |        94.29% |         94.92% |   26.18 |               41.23 |
| 4x, 30 ms        |    94.19% |        94.34% |         94.39% |   19.83 |               20.62 |
| 4x, 50 ms        |    95.67% |        95.67% |         95.70% |   22.55 |               36.42 |

At 8x and 30 ms the table it built is two lines: send the photo on the good
network, stop at the first INT8 exit on the degraded one. That is the second
yardstick's own choice.

**A sweep shows where it holds.** Four device speeds by four deadlines, three
seeds each, means over the seeds
(`decide_sweep_resnet18_early_exit_imagenette_t4_wifi-4g_stay0.95.csv`):

| device | deadline | 1. best fixed | this rule | 3. fair target | rule minus fixed |
| ------ | -------- | ------------: | --------: | -------------: | ---------------: |
| 2x     | 20 ms    |        94.68% |    94.99% |         95.08% |            +0.31 |
| 2x     | 30 ms    |        95.63% |    95.51% |         95.66% |            -0.12 |
| 2x     | 50 ms    |        95.63% |    95.53% |         95.66% |            -0.10 |
| 2x     | 100 ms   |        95.63% |    95.53% |         95.66% |            -0.10 |
| 4x     | 20 ms    |        78.42% |    84.19% |         84.22% |            +5.76 |
| 4x     | 30 ms    |        94.12% |    93.97% |         94.14% |            -0.15 |
| 4x     | 50 ms    |        95.63% |    95.55% |         95.66% |            -0.08 |
| 4x     | 100 ms   |        95.63% |    95.55% |         95.66% |            -0.08 |
| 8x     | 20 ms    |        45.63% |    45.94% |         45.94% |            +0.31 |
| 8x     | 30 ms    |        57.12% |    74.23% |         74.24% |           +17.12 |
| 8x     | 50 ms    |        94.04% |    94.79% |         94.87% |            +0.76 |
| 8x     | 100 ms   |        95.63% |    95.53% |         95.66% |            -0.10 |
| 16x    | 20 ms    |        45.63% |    45.63% |         45.63% |             0.00 |
| 16x    | 30 ms    |        46.22% |    46.22% |         46.22% |             0.00 |
| 16x    | 50 ms    |        56.63% |    76.37% |         76.37% |           +19.75 |
| 16x    | 100 ms   |        95.60% |    95.60% |         95.60% |             0.00 |

**The rule is within 0.17 points of the fair target in every cell,** and
within 0.31 on any single seed. Its own score moves by a standard deviation
of 0.08 to 0.69 points across the three seeds.

**Deciding pays in 3 cells of 16,** by 5.76, 17.12 and 19.75 points. Each is a
case where the deadline is just too tight for the device alone and sending
still fits on the good network. In the other 13 a fixed option is within 0.8
points.

**In 7 cells the rule scores slightly below the best fixed option,** by at
most 0.15 points. The best fixed option is picked with hindsight on the test
frames, the rule's plans on the other half of the images, and the rule gives
up near-ties for speed. What it gets back is time: where both score above 94%,
its frames average 14.90 ms against 18.21 at 2x and 50 ms, and 25.03 against
36.43 at 4x and 50 ms.

**SplitEE as published scores 0.00% in 5 of these cells** and is within a
point of the best fixed option in 7. With every repair it is never at zero,
within a point of the best fixed option in 7, and 6.13 points behind the fair
target on average.

**A split plan does come out ahead here, by a hair.** With the deadline loose
and the network good, the yardsticks picked a plan that sends the feature map
in 9 of 48 runs and this rule did in 2. The margin is a few frames in 9810: in
the 4x, 50 ms scoreboard the second yardstick, with a split on the good
network, and this rule, sending the photo there, both score 9385 frames,
at 32.90 and 22.55 ms mean. So where a split comes out ahead it is by a few
frames in ten thousand, and it costs time.

**"Not confident, so send the photo" was chosen once in 48 runs.** With the
same network on both sides the server gives the answer the device would have,
so running part of the network first only spends device time. The option needs
a stronger model on the server to earn its place.

**The lead depends on how fresh the view of the network is.** A device does
not learn the network for free: sending reveals it and running locally
reveals nothing. At 8x and 30 ms:

| what the rule knows of the network                                 | useful |
| ------------------------------------------------------------------ | -----: |
| the state each frame is about to meet                              | 73.75% |
| the state at a probe every 5 frames                                | 69.83% |
| the state at a probe every 20 frames                               | 60.78% |
| only what its own sends revealed, sending one when 10 frames stale | 66.87% |
| *1. best fixed option, for reference*                              | 56.68% |

A state lasts 20 frames on average, so a probe every 20 frames is often
describing a state that has already ended. Learning from its own sends alone,
the rule keeps 10 of the 17 points.
Probes are free in this model, which flatters the two probe rows.

**A decision takes about a quarter of a microsecond:** a median of 0.19 to
0.30 microseconds across these runs, timed in Python on the laptop. That is
the lookup alone. Building the table, once per network and deadline, and
observing the network are not included.

---

## 16. Carried out between two processes

Everything above looks outcomes up. Here a rule's plans are executed
(`edge serve`, `edge run`): a device process runs the stage graphs a plan
names and checks each exit's confidence, and when the plan says so it sends
the photo or the feature map over a TCP connection to a server process, which
finishes the network and answers. Each frame is timed end to end.

**Two things are emulated,** because both processes share one laptop. The
device 8 times slower waits, after each stage, 7 times as long as the stage
took. The network's delay is a wait before each send, of that network's round
trip and transfer time. The computing, the bytes, the socket and the answers
are real. The server is the laptop CPU at 6 threads.

One pass of the test images, 1962 frames, 8x device, 30 ms deadline. The
network was good on 812 frames and degraded on 1150. Each rule's frames were
then looked up in the simulator for comparison (`live_*.csv.gz`):

| rule                  | useful, looked up | carried out | same plan | same right or wrong | median time gap |
| --------------------- | ----------------: | ----------: | --------: | ------------------: | --------------: |
| lookup, this project  |            73.50% |      72.32% |      100% |                100% |        +0.12 ms |
| always send the photo |            40.16% |      39.91% |      100% |                100% |        +3.51 ms |
| always local, INT8    |             0.00% |       0.00% |      100% |                100% |        +3.64 ms |
| SplitEE, every repair |            56.07% |      52.70% |    71.05% |              91.74% |        +0.28 ms |

**The ranking holds when carried out:** 72.32%, 52.70%, 39.91%, 0.00%.

**The stage graphs and the wire reproduce the table's answers.** For the three
rules that do not learn, every one of 1962 frames was answered right or wrong
exactly as looked up. That they played the same plan is true by construction,
since each was told the same network sequence in both runs.

**The simulator's medians hold and its tails are too thin.** For the lookup
rule, 45 frames were in time looked up and late carried out, 25 of them with
the right answer, and 2 went the other way: 23 useful frames, the 1.2 points
between the two scores. 44 of the 45 were frames answered locally at the first
INT8 exit. Their median was 21.59 ms carried out against 21.74 looked up, but
their 99th percentile was 40.27 against 24.61 ms. The simulator replays one
recorded time per image and so has no bad moments of its own. Part of this
tail belongs to the emulation, since waiting 7 times as long multiplies any
hiccup by 8.

**A server that has been idle is slower.** Profiled back to back, the server
answered a photo in a median 6.27 ms. In the run that sends every frame it
took 6.41 ms when the frame was on the good network, about 12 ms after the
last request, and 10.17 ms on the degraded one, after a 56 ms wait. The lookup
rule only sends on the good network, and its 812 requests took 6.20 ms.

**A rule that learns does not replay.** SplitEE played the same plan as its
looked-up self on 71.05% of frames. It learns from what each frame cost, so a
millisecond's difference changes what it tries next and the two runs drift
apart. Its score still lands within 3.4 points.

**The network can be named from a send's duration,** at least when the
candidates are this far apart. After each send the device compared its
duration with what each known network would have taken for that many bytes.
In the run that sends every frame it named the network correctly on all 1962
sends. The rules were not acting on that guess in these runs: each was told
the state.

---

## 17. Two real links

The networks above are assumed. To see what a real one looks like, a recorder
(`scripts/record_network.py`) uploaded to a public speed-test endpoint once a
second for ten minutes, over two Wi-Fi links (`network_trace_*.csv`):

| measure                                              | public Wi-Fi | shared Wi-Fi |
| ---------------------------------------------------- | -----------: | -----------: |
| round trip, 1-byte upload, median                    |            - |      56.2 ms |
| 8 KB upload, a photo: median                         |      65.2 ms |      59.9 ms |
| 8 KB upload: 95th percentile                         |     146.7 ms |      99.9 ms |
| 8 KB upload: slowest 1% took at least                |     222.7 ms |     211.1 ms |
| 30 KB upload, a feature map: median                  |     115.2 ms |      99.9 ms |
| uplink, from a 250 KB against an 8 KB upload, median |    14.0 Mbps |    13.1 Mbps |
| 8 KB uploads within 30 ms                            |     0 of 600 |     0 of 600 |
| 8 KB uploads within 100 ms                           |   529 of 600 |   571 of 600 |

The first trace has no usable round trip: it was recorded with a request the
endpoint answers slowly, and the recorder was changed before the second. For
the same reason its `uplink_mbps` column is not used here.

**Both links resemble the assumed degraded network,** 50 ms and 10 Mbps, on
which a photo would take about 56 ms. Neither resembles the good one, 5 ms and
100 Mbps. That network stands for a server close by on a clean link, and no
such link was recorded.

**On links like these the 30 ms cell has nothing to decide.** No photo reached
the endpoint in 30 ms on either link, so a device there would answer locally
on every frame. The band where deciding pays moves with the link. Replaying
these traces through the simulator in place of the two-state network has not
been done.

---

## What these numbers are not

**The reference machine is a laptop, not an edge board.** It stands in for one,
and constrained thread counts are the approximation. A real edge SoC has
different cache sizes, memory bandwidth, and thermal behaviour, so this shows
scaling behaviour rather than a prediction for any particular device.

**A slower device is the laptop's times multiplied by a constant.** No slow
device was measured. A real one is not uniformly slower: INT8 in particular
gains more or less depending on the instructions the processor has.

**The networks are assumed, and the changing one has two states.** A round
trip and an uplink speed, with no loss, no jitter and nothing in between. The
two links recorded in section 17 both behaved like the degraded state.

**Sending buys time here, never accuracy.** Both ends run the same network, so
the server gives the answer the device would have. Imagenette is also easy:
the full model is wrong on 179 of 3925 images. Whether to send the hard images
to a stronger model is a question this setup cannot ask.

**The yardsticks are found on the test frames,** and the rules' settings on
the other half of the same dataset, profiled on the same machine. The halves
differ only in which images they hold. A deployed device would need its own
profile.

**The live run is one laptop with two emulations,** in one scenario. It checks
that the simulator's answers and typical times can be reproduced. It is not a
measurement of a slow device or of a network.

**One published method, in one port.** SplitEE was written for a text model
and a different objective. Its results here describe how its assumptions fare
on these measurements, not the method in its own setting.

**Reproducibility is weaker than a fixed seed suggests.** Two runs of the
identical training command gave 96.94% and 96.89%. GPU arithmetic is not
bit-deterministic by default, and the dataloader worker count changes the
augmentation stream. Runs land within about a tenth of a point.

**Latency is never portable.** Every row records the machine, CPU, and thread
count it was measured under, so figures from different hardware cannot be
compared by accident.

---

## Next

A harder task with a stronger model on the server, so that sending can buy
accuracy and not only time. A real slow device in place of a multiplier. And
the recorded links replayed in place of the two-state network.
