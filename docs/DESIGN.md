# Design

The shape of the work from here on, written down before the code so that the
code follows it. Each section says what is decided and why. Anything not here
is either already built (see the README) or future work.

---

## 1. The claim under test

Measured honestly, choosing between answering locally, answering early and
sending to a server is the easy part: simple rules reach the best that rules
of their kind can do. What actually improves inference at the edge is the
menu of options: the runtime, the numeric precision, a better local fallback,
a stronger server, and doing two things at once.

The early-exit measurements on Imagenette produced this claim. The
model-menu measurements on ImageNetV2 are built to test it on different
models, a different dataset, real devices and real links.

---

## 2. What a frame is worth

A frame is not one choice. It is a timeline: from the moment it arrives the
device can start local work, send it, or both, and each finished piece of
work is a better answer than the one before. What that timeline is worth
depends on the application.

| application | example | score | deadline D, horizon H |
| ----------- | ------- | ----- | --------------------- |
| act once | braking | `useful`: correct by D | D = 50 or 100 ms |
| display | a live overlay | `useful` at D, plus `corrected`: the share fixed between D and H | D = 200 ms, H = 1000 ms |
| record | indexing stored footage | `record@H`: the server's answer if it arrives by H, else the local one | H = 1000 ms, and unbounded as a reference |

**Hedging** follows from this. The device sends the photo and runs its best
local plan at the same time. For act once it answers with the server if the
reply arrives by D, and with its own answer at D otherwise. For record, a late
reply still replaces the local answer if it arrives by H.

Hedging is known: hedged requests are an old idea in datacentre systems, and
SPINN (MobiCom 2020) and UniExit (arXiv 2611.09159) run local and remote work
together. What this project adds is measuring it against the yardsticks below,
on measured costs, under all three kinds of application.

---

## 3. Two sets of measurements, one table

| | early-exit measurements | model-menu measurements |
| --- | --- | --- |
| question | how good is choosing; what is early exit worth | which options are worth having |
| data | Imagenette | ImageNetV2, matched frequency |
| model | ResNet-18 with two early exits, the same model on device and server | pretrained classifiers without exits; a small one on the device, a larger one on the server |
| device | a laptop, with a slower device modelled by a constant factor | a laptop slowed down for real, or a single-board computer |
| servers | the laptop's CPU and a Tesla T4, profiled | a second computer on the local network, and a GPU over the internet |
| network | two assumed states, switching as a Markov chain | links recorded with the project's own protocol |

**Imagenette's provenance.** 3,791 of its 3,925 validation images are ImageNet
training images, which the ImageNet-pretrained backbone has seen.
`edge data origin` compares them with the 134 that ImageNet kept for
validation: the validation-origin images score higher at every exit, in both
precisions and both models, and every interval overlaps
(`results/accuracy_by_origin.csv`). There is no sign of inflation, but the
caveat stands. ImageNetV2 was collected after ImageNet and no ImageNet-trained
model has seen it.

---

## 4. Decisions

1. **Links are named, and every route uses exactly one.** With a local server
   and a remote one, a frame can meet two links at once. What a rule knows
   about the network is a mapping from link to observation.
2. **The simulator is queue-free.** Each frame is resolved alone. The live
   runs that check the simulator are synchronous: a frame starts only after
   the previous reply has arrived or its horizon has passed. A separate run
   at a fixed frame period, where work can overlap, is reported as a stress
   result, not as a check.
3. **One manifest per dataset fixes tuning, test and calibration images.** A
   seed only changes the frame order and the network. Before this, each seed
   drew its own tuning half, so calibration images could leak into a test.
4. **The analysis core is general; measurement is per model.** `simulate`,
   `policies`, `yardsticks`, `scoreboard` and the device half of `runtime`
   serve both sets of measurements. The early-exit profilers stay as they
   are, and their table enters the new format through an adapter.
5. **A golden test protects the published results.** After the core is
   generalised, the committed decision map and two committed scoreboards
   must reproduce exactly.
6. **A fifth yardstick.** Rung 3, the best confidence rule per network, stays
   as it was. Rung 3h is the same search over a family that also contains
   hedged plans, so that 1 ≤ 2 ≤ 3 ≤ 3h ≤ 4 holds by construction, and each
   kind of application has its own rung 4.
7. **Predictions come first.** `results/PREDICTIONS.md` is committed before
   any model-menu measurement exists.
8. **The device side does not import torch,** so that any 64-bit Linux board
   with Python, NumPy and ONNX Runtime can be the device.
9. **Choices are made on tuning images only:** which small model, which
   resolutions, which JPEG quality.
10. **Comparisons are paired.** Two rules are compared on the same frames
    under the same network, with a bootstrap interval over frames. The margin
    within which plans count as tied is a parameter.

---

## 5. The table

One long table per device configuration, one row per image and local option
and per image and route.

| column | local rows | route rows |
| ------ | ---------- | ---------- |
| `image`, `label`, `split` | yes | yes |
| `kind` | `local` | `route` |
| `variant` | `{model}@{resolution}/{precision}`, e.g. `mnv3l@160/int8`; the early-exit table keeps `float32` and `int8` | the variant a feature route continues from |
| `exit`, `exit_name` | an index; a model without exits has only `0`, `final` | the exit a feature route follows |
| `route`, `server`, `link` | | e.g. `jpeg224q85>edge/resnet50`, `edge/resnet50`, `lan-wifi` |
| `device_config` | e.g. `laptop-asis-t2`, `laptop-1.4g-2c` | the same |
| `device_ms`, `preprocess_ms` | to the exit, cumulative; resize and normalise | 0 |
| `prepare_ms`, `payload_bytes` | 0 | encoding on the device; bytes sent |
| `server_receive_ms`, `server_compute_ms` | 0 | measured on the server |
| `predicted`, `confidence`, `correct` | the device's | the server's own |

The decoded frame is the camera's output and sits outside every timer. A local
answer pays for preprocessing, a sent photo for resizing and encoding, the
server for decoding and its own preprocessing.

Energy is a separate table per device configuration: joules per inference for
each variant and each encoding, with an interval, the idle power it was
measured against, and the RAPL domain.

---

## 6. The simulator

- **`Measurements`** holds a block per local variant (time, confidence and
  correctness per exit) and a block per route (bytes, preparation, server
  time, the server's correctness and confidence, its server and link). It is
  built from the long table by joining on `image`, and refuses missing or
  duplicated images and labels that disagree.
- **`Plan`** names a variant, how deep to run, a threshold per exit, what to do
  when no exit is confident, a route to send to, and an optional route to
  hedge with.
- **`resolve`** records events, not only a total: when the local answer is
  ready and whether it is right, when the server's arrives and whether it is
  right. Each kind of application reads its score from those events. For a
  hedged plan the device encodes, then sends and computes in parallel. With
  cancellation the local run stops at the next stage boundary once the reply
  is in.
- **Recorded links** replace the Markov states with per-frame round trips and
  uplink rates. For grouping, frames fall into at most three bins per link,
  cut on quantiles of the tuning recording, so that the yardsticks that pick
  a plan per group never get groups small enough to act as an answer key.

---

## 7. Live runtime and probing

- Every request and reply carries an id, and one reader thread per connection
  hands replies to the frame that asked. A late reply can never be read as
  the next frame's.
- Hedging runs the send on a worker thread while the device computes.
- `echo` requests let the same server answer link probes. `edge probe` sends,
  at the frame period, a small echo for the round trip and a large one for
  the uplink rate, and every tenth tick a photo-sized echo to check that the
  two numbers predict a real send.
- A server holds one or more models behind a header field, on ONNX Runtime
  (CPU) or PyTorch (CUDA).

---

## 8. The advisor

`edge advise` takes a device's table, the servers' profiles, link recordings,
a deadline, the kind of application and, optionally, prices on device time,
bytes and energy. It reports:

- every fixed option under the recorded link;
- whether adapting is needed at all: the best fixed option against the lookup
  rule and rungs 3 and 3h, with paired intervals; if they overlap, the advice
  is to play the fixed option;
- what hedging gains and what it costs, with and without cancellation;
- with prices, the frontier between score and cost;
- the plan for each link bin, in a file that `edge run` can load.

Settings are chosen on the first recording and reported on a later one.

---

## 9. Measuring a slower device

The laptop is slowed for real rather than multiplied:

- boost is turned off;
- two physical cores are pinned to one fixed frequency, 1.4 or 2.0 GHz;
- the device process runs on those cores only, with two runtime threads.

Each row records the affinity, the policy of each pinned core and the measured
average frequency. This is the same processor design, slower and narrower, and
not a different kind of device. Comparing each variant's slowed time with its
normal time tests the constant-factor assumption the early-exit measurements
relied on.

---

## 10. Energy

The processor's own RAPL counters (package and cores) are read around batches
of 200 inferences, repeated five times, with idle power measured before and
after. Marginal energy per inference is (E − P_idle · t) / N. On this processor
the counters are estimates from a model, so they are compared only within this
machine. Radio energy is not measured; bytes sent are reported instead.

---

## 11. Invariants the tests hold

1. The committed decision map and scoreboards are unchanged by the refactor.
2. On random tables, under Markov and recorded links, the rungs never cross,
   for every score.
3. No plan, hedged or not, scores above rung 4 on any frame.
4. Hedging over a link that never delivers scores as its local plan; over an
   instant link, as sending; cancellation never adds device time; with H = D,
   record equals act once.
5. The table join refuses missing, duplicated or contradictory images.
6. No tuning or calibration image is ever a test image.
7. The device's Pillow and NumPy preprocessing matches torchvision's.
8. The device modules import without torch.
9. Replies reach the frame that asked, never the next one.
10. Recorded links are read frame by frame, and bins are fitted on tuning data
    only.
11. The energy reader survives counter wrap-around.
12. The photos the server decodes are byte-identical to the device's.
