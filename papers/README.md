# Reference papers

The PDFs in this directory are **not committed**; only this file is tracked.
They can be fetched from the links below. Licenses differ per paper and are
recorded here. Note that "free to read" and "free to redistribute" are not the
same thing.

---

## 1. Angelucci et al. (2026): the paper this project builds on

> S. Angelucci, R. Valentini, M. Levorato, F. Santucci, C. F. Chiasserini.
> **"Distributional Reinforcement Learning for task offloading, resource
> allocation and early exit selection at the edge."**
> *Computer Networks* **288** (2026) 112652.
> DOI: [10.1016/j.comnet.2026.112652](https://doi.org/10.1016/j.comnet.2026.112652)

**Open access, CC BY 4.0.** The PDF states: "© 2026 The Authors. Published by
Elsevier B.V. This is an open access article under the CC BY license." That
permits redistribution with attribution, so this is the one paper here that
could legally be committed to the repo if we want it version-controlled
alongside the code.

**Relevance.** The paper treats the neural network as a black box described by
two vectors per model:

- `c`: MOPs (millions of operations) required to reach each exit
- `a`: accuracy achieved at each exit

obtained by *"offline profiling of a DNN architecture modified with explicitly
trained intermediate classifiers."* The worked example uses three exits with
`c = [22.6, 26.0, 59.5]` MOP and `a = [56.4, 78.0, 86.0]`%.

This repo implements **that offline-profiling step**: producing `c` and `a`
from real measurements on real hardware, and studying how INT8 quantization
changes them. The paper's reinforcement-learning controller and wireless
channel model are out of scope here.

---

## 2. Wang et al. (2025): on-device AI survey

> X. Wang, Z. Tang, J. Guo, T. Meng, C. Wang, T. Wang, W. Jia.
> **"Empowering Edge Intelligence: A Comprehensive Survey on On-Device AI Models."**
> *ACM Computing Surveys* **57**(9), Article 228, April 2025.
> DOI: [10.1145/3724420](https://doi.org/10.1145/3724420)
> Preprint: [arXiv:2503.06027](https://arxiv.org/abs/2503.06027)

Free to read and download from the ACM Digital Library, but **not** open access
in the licensing sense. The PDF carries ACM's standard notice: "© 2025
Copyright held by the owner/author(s). Publication rights licensed to ACM,"
granting copies "for personal or classroom use," while "to post on servers or
to redistribute to lists, requires prior specific permission and/or a fee."
Link to the arXiv preprint when sharing.

**Relevance.** Survey of on-device AI: model compression, hardware
acceleration, and deployment constraints. Source material for `DOMAINS.md`
(where early exit does and does not pay off) and for positioning quantization
among the other available optimization techniques.

---

## 3. Frankle & Carbin (2019): pruning background

> J. Frankle, M. Carbin.
> **"The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks."**
> ICLR 2019. [arXiv:1803.03635](https://arxiv.org/abs/1803.03635)

Freely available on arXiv. No explicit redistribution license in the PDF, so it
is linked rather than committed.

**Relevance.** Background only. Covers *pruning*, a different optimization
technique from the quantization used in this project. Kept as context for the
report's discussion of alternative techniques. **Pruning is not implemented
here.**
