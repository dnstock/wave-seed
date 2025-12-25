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

### 1. Scaling Benchmark (Operator-Level)

A minimal benchmark compares:

- **Baseline:** single-head self-attention core (QKᵀ → softmax → AV)
- **Replacement:** FFT-based spectral mixing

**Results (CPU / NumPy, single block):**

| n | attention (ms) | spectral mixing (ms) |
|---:|---:|---:|
| 128 | 2.39 | 0.24 |
| 256 | 2.83 | 0.47 |
| 512 | 4.18 | 0.96 |
| 1024 | 9.58 | 1.92 |
| 2048 | 29.35 | 3.88 |
| 4096 | 106.21 | 9.50 |

The log-log plot [bench_scaling_loglog.png](bench_scaling_loglog.png) shows attention accelerating sharply while spectral mixing follows a much gentler curve consistent with O(n log n) behavior.

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
# Run first:
python train.py     # minimal training loop

# Then run:
python sample.py    # create sampling
```

### Scaling benchmarks

```bash
# Run first:
python bench/bench_mixers.py --d 256 --w 128 --tmin 128 --tmax 4096 --warmup 5 --iters 20 --csv bench_results.csv

# Then run:
python bench/plot_bench.py --csv bench_results.csv --out bench_scaling.png
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
