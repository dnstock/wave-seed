# Seed Note: Spectral Mixing as a Drop-in Replacement for Quadratic Attention

## Claim
Replacing quadratic self‑attention with FFT‑based spectral mixing changes sequence‑length scaling from ~O(n²) to ~O(n log n), while remaining trainable end‑to‑end in a minimal language modeling setup.

---

## Motivation
Modern Transformer attention scales quadratically with sequence length due to explicit n×n token interactions. This creates a hard ceiling on context length, memory usage, and compute cost. The goal of this experiment is not to improve language quality, but to demonstrate a *fundamentally different scaling regime* using a global mixing operator derived from signal processing.

---

## Minimal Benchmark: Scaling Behavior

**Baseline**
- Single‑head self‑attention core: QKᵀ → softmax → AV  
- Complexity: ~O(n²·d)

**Replacement**
- Spectral mixing via FFT → learned frequency‑domain filter → inverse FFT  
- Complexity: ~O(n·d·log n)

### Results (CPU / NumPy, single block)
| n | attention (ms) | spectral mixing (ms) |
|---:|---:|---:|
| 128 | 2.39 | 0.24 |
| 256 | 2.83 | 0.47 |
| 512 | 4.18 | 0.96 |
| 1024 | 9.58 | 1.92 |
| 2048 | 29.35 | 3.88 |
| 4096 | 106.21 | 9.50 |

See: [bench_scaling_loglog.png](bench_scaling_loglog.png)

**Observation:**  
Attention cost accelerates rapidly as n grows, while spectral mixing follows a much gentler curve consistent with O(n log n) behavior.

---

## Learnability Check (End‑to‑End Training)

To verify that spectral mixing is not merely fast but *usable*, a tiny character‑level next‑token model was trained using:

- No attention
- No ML frameworks (NumPy only)
- A learned spectral filter (FFT → filter → inverse FFT)

**Training signal**
- Loss decreases monotonically from ~3.05 → ~2.68 over 500 steps
- Model produces non‑trivial character distributions and structure

This establishes that attention‑free spectral mixing can participate in gradient‑based learning without collapsing.

---

## What This Is / Is Not

**This work demonstrates**
- A concrete replacement for the quadratic interaction operator
- A different asymptotic scaling regime
- Practical trainability in a minimal setting

**This work does NOT claim**
- State‑of‑the‑art language quality
- Full replacement of attention in all regimes
- Immediate production readiness

---

## Why This Matters
Scaling limits, not constant factors, dominate the cost of large language models. Any operator that changes the asymptotic behavior of sequence interaction directly attacks the long‑context and compute‑cost bottlenecks faced by current architectures.

This suggests a path toward long-context models whose cost grows subquadratically with context length, rather than exploding with it.

---

## Next Steps
1. Extend spectral mixing with residuals and normalization
2. Evaluate hybrid stacks (spectral + sparse/local attention)
3. Measure scaling on longer contexts (n ≫ 4096)

This note represents a *seed*: a proof that a different scaling law is both mathematically grounded and empirically observable on modest hardware.
