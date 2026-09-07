# Where Early Exit Helps, and Where It Does Not

Early exit attaches classifiers partway through a network so an easy input can
be answered without running the rest of it. That is only ever worth doing under
particular conditions, and this note sets out which.

The measurements behind the claims here are in
[`results/RESULTS.md`](results/RESULTS.md).

---

## The governing principle

One sentence covers most of it:

> Early exit pays when the compute skipped is large relative to the overhead of
> deciding to skip it.

Everything below follows from that ratio, plus one requirement: **inputs must
vary in difficulty**. A network that exits early on every input was simply too
big, and one that never exits early has paid for exit heads it does not use.

---

## Where it helps

**Input difficulty genuinely varies.** A camera watching a loading bay sees an
empty frame most of the time and a partially occluded forklift occasionally.
Spending equal compute on both is the waste early exit removes.

**Latency is bounded, not averaged.** When a deadline exists, what matters is
finishing in time, not average throughput. A model that answers easy inputs in
a third of the time leaves headroom for the hard ones. This is the setting the
Angelucci et al. controller operates in.

**Streaming perception.** Consecutive video frames are highly correlated, so
most frames are easy given what the previous one contained. Per-frame cost
varies enormously while the deadline stays fixed at the frame interval.

**Edge and cloud cascades.** A small local model handles what it is confident
about and escalates the rest. An exit head is exactly the confidence estimate
that decision needs, and it doubles as a bandwidth control: escalating 10% of
inputs costs a tenth of the uplink.

**Large models, especially language models.** Skipping twenty transformer
layers dwarfs the cost of one branch decision. This is why confident adaptive
decoding and layer skipping are active techniques in language model serving,
on datacenter accelerators rather than edge devices. The ratio is what matters,
not the hardware.

---

## Where it does not help

**Uniformly hard inputs.** Fine-grained classification, medical imaging, and
anything adversarial rarely produce confident early predictions. The exits fire
almost never, and you have paid for heads that do nothing.

**Large-batch serving.** Samples in a batch exit at different depths, so the
batch runs ragged: the hardware waits on the slowest member while finished
lanes idle. This wastes exactly the parallelism batching exists to gain. It is
a batching problem rather than a GPU problem, and it disappears at batch size
one.

**Fixed-function accelerators.** NPUs typically want a static graph compiled
ahead of time, and many run quantized models only. Data-dependent branching is
either unsupported, forces a fallback to the CPU, or requires splitting the
network into separately compiled subgraphs with host-side control between them,
which reintroduces the per-exit overhead the technique was meant to save.

**Small models on accelerators.** Our own case, and the sharpest version of the
argument. On a mobile GPU every exit decision means reading a confidence score
back to the CPU to branch on it. That synchronisation can cost more than
skipping the remaining layers of a small network saves. The problem is worse
here than on an NPU: there it is compilation, here it is a round trip in the
middle of an inference that was supposed to be getting shorter.

**When the model was simply oversized.** If nearly everything exits at the
first head, the honest fix is a smaller network, not a bigger one with an
escape hatch.

---

## What our own measurements suggest

Three findings bear on where this technique belongs.

**Six threads beat twelve**, and the reason was tail latency rather than the
median. Early exit is a tail-latency technique too: it does not make the hard
cases faster, it stops easy cases from paying for them. Both matter in the same
deployments and for the same reason.

**The runtime mattered more than the optimization.** Moving from PyTorch to
ONNX Runtime was a 3.72x speedup; INT8 quantization then added 1.67x. Before
reaching for early exit, it is worth asking whether the execution stack has
been chosen at all. Early exit is a sophisticated lever, and sophisticated
levers are worth pulling after the blunt ones.

**INT8 cost 0.48 points of accuracy.** Early exit trades accuracy for latency
too, but adaptively rather than uniformly: the loss is concentrated on inputs
the network finds hard, whereas quantization degrades every input a little.
Those are different shapes of trade, and which is preferable depends on whether
the application tolerates occasional larger errors or uniform smaller ones.

---

## The short version

Early exit suits **variable-difficulty inputs, deadline-bound inference, and
models large enough that skipping layers outweighs the branch decision**.

It suits neither uniformly hard workloads, nor batched throughput serving, nor
hardware that wants a static graph.

It is not a general speedup. It is a way to spend less compute on easy inputs,
and it is worth exactly as much as the variation in your inputs.
