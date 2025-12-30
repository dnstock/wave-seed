# Spectral Mixing: An Attention-Free Sequence Model

This repository explores whether the quadratic interaction at the core of modern Transformers can be replaced, rather than optimized, using a frequency-domain mixing operator based on the Fast Fourier Transform (FFT). 

Any operator that changes the asymptotic behavior of sequence interaction directly attacks the long-context and compute bottlenecks that constrain current architectures. Thus, a goal is to invent fundamentally new interaction operators that alter these asymptotic properties.

The short-term goal is to demonstrate a different scaling regime that remains trainable end-to-end, while avoiding n×n token interactions. Ultimately this could enable long-context models whose cost grows subquadratically with context length, creating a novel language modeling primitive.

---

## Why This Exists

Self-attention scales quadratically with sequence length due to explicit token–token interactions. This creates hard ceilings on context length, memory usage, and compute cost (training and inference).

This project asks a narrower question:

> Can we replace the quadratic interaction operator itself with something that scales subquadratically, while still supporting gradient-based learning?

---

## Core Idea

At a high level, the system replaces explicit token–token interaction with a single global mixing operator that acts uniformly over the entire sequence.

```
text → embeddings → spectral mixing (FFT) → reconstruction → logits
```

Everything downstream of spectral mixing is deliberately simple and local; all long-range interaction is handled by the frequency-domain operator.

The key properties are:

- Global mixing happens in the frequency domain
- No attention matrices
- No n×n interactions
- Complexity ~ O(n log n) in sequence length

---

## Repository Structure

The project has a modular structure, with a clean separation of concerns.

Here are the top-level directories:
```
bench/   →  benchmarking scripts + outputs
layers/  →  spectral + reconstruction primitives
model/   →  wiring + gradients
optim/   →  update rules
data/    →  corpus + encoding
```

Here is the file structure:
```
wave-seed/
│
├── bench/
│   ├── bench_mixers.py    # operator-level scaling benchmarks
│   ├── plot_bench.py      # benchmark plotting utilities
│   └── results/           # benchmark CSVs, plots, and artifacts
│
├── layers/
│   ├── spectral.py        # FFT mixing + backward
│   ├── norm.py            # RMSNorm + backward
│   ├── mlp.py             # position-wise MLP (+ backward)
│   └── output_head.py     # reconstruction head forward/backward
│
├── model/
│   ├── embedding.py       # embedding lookup + grads
│   └── lm.py              # forward graph wiring only
│
├── optim/
│   └── sgd.py             # parameter update rules
│
├── data/
│   └── char_dataset.py    # corpus + encoding
│
├── train.py               # experiments (training loop)
├── sample.py              # inference (sampling only)
├── memo.md                # working research notes
└── README.md
```

This layout keeps experimental logic, mathematical primitives, and training infrastructure clearly separated, making the system easier to reason about and extend. Each file stays small, auditable, and replaceable.

---

## What’s Implemented

### 1. Scaling & Memory Benchmarks (Operator-Level)

A minimal benchmark compares:

- **Global Attention:** single-head self-attention core (QKᵀ → softmax → AV)
- **Local Attention:** sliding window attention with fixed window size
- **Attention-Free Spectral Mixing (AFSM):** FFT-based replacement

**Results (CPU / NumPy, single block, d=256):**

| Seq. Length (`T`) | Global Attention | Local Attention | AFSM |
|---:|---:|---:|---:|
| | latency / memory | latency / memory | latency / memory |
| 128 | 0.06 ms / 0.3 MB | 2.44 ms / 0.3 MB | 0.08 ms / 0.5 MB |
| 256 | 0.22 ms / 1.1 MB | 5.07 ms / 0.4 MB | 0.14 ms / 1.1 MB |
| 512 | 0.90 ms / 4.2 MB | 10.35 ms / 0.7 MB | 0.28 ms / 2.1 MB |
| 1024 | 3.94 ms / 16.8 MB | 20.93 ms / 1.2 MB | 0.54 ms / 4.2 MB |
| 2048 | 17.38 ms / 67.2 MB | 42.03 ms / 2.3 MB | 1.22 ms / 8.4 MB |
| 4096 | 80.07 ms / 268.5 MB | 84.53 ms / 4.4 MB | 2.70 ms / 16.8 MB |

The log-log [bench scaling plot](bench/results/bench_scaling_20251229_1801.png) shows attention accelerating sharply while spectral mixing follows a much gentler curve consistent with `O(n log n)` behavior. The [bench memory plot](bench/results/bench_scaling_20251229_1801_mem.png) is perhaps even more significant, since attention leads to superlinear memory growth while the AFSM peak allocation grows slowly and predictably. This addresses the primary bottleneck for long-context models. The most compelling insight comes when comparing to local attention, since AFSM scales more favorably while providing the global interaction that local attention sacrifices.

---

### 2. Learnability Check (End-to-End)

A tiny character-level next-token model is implemented in NumPy only (no ML frameworks), and trained using *no attention* with a learned spectral filter (FFT → filter → inverse FFT).

**Observations:**
- Loss decreases monotonically from ~3.05 → ~2.68 over 500 steps
- The model learns non-trivial character structure
- Training is stable and end-to-end

This establishes that spectral mixing is both fast and usable as a sequence modeling primitive.

---

## Scope of Claim

This work demonstrates a concrete replacement for the quadratic interaction operator, a different asymptotic scaling regime, and practical trainability in a minimal setting.

This work does not claim superior language quality, full replacement of attention in all regimes, or immediate production readiness.

---

## How to Run

Follow these steps to run and test the FFT LM. No GPUs or ML frameworks required.

### Train and sample

```bash
# Run minimal training loop
python train.py

# Then create sampling data
python sample.py
```

### Scaling benchmarks

To run with the default parameters:
```bash
# Calculate the benchmarks
python bench/bench_mixers.py

# Then plot the results
python bench/plot_bench.py --latest
```

Custom model & sequence dimentions and benchmarking settings are also available.
```bash
# View the runtime options and descriptions
python bench/bench_mixers.py --help
python bench/plot_bench.py --help
```

---

## Why This Matters

Scaling limits dominate the cost of large language models, not constant factors.

This project is a seed. It serves to prove that a different scaling law is both mathematically grounded and empirically observable on modest hardware.

---

## Next Directions

- [ ] Residual + normalization layers
- [ ] Hybrid models (spectral + sparse/local attention)
- [ ] Longer-context experiments (n ≫ 4096)
- [ ] Better spectral parameterizations

---

## Status

Exploratory research. Shared for discussion, critique, and collaboration.

<br>
<br>
<br>
<br>

---

Author: [Dan Harcsztark](https://github.com/dnstock), 2025
