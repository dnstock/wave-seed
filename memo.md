# Seed Note: Spectral Mixing as a Drop-in Replacement for Quadratic Attention

## Claim
Replacing self-attention’s quadratic token interaction with spectral mixing changes scaling from ~O(n^2) to ~O(n log n) in sequence length n (empirically demonstrated in a minimal benchmark).

## Minimal Benchmark
- Baseline: single-head self-attention core (QK^T + softmax + AV)
- Replacement: FFT-based 2D mixing of (n, d) activations

### Results (CPU / NumPy)
| n | attention (ms) | FFT (ms) |
|---:|---:|---:|
| 128 | 2.39 | 0.24 |
| 256 | 2.83 | 0.47 |
| 512 | 4.18 | 0.96 |
| 1024 | 9.58 | 1.92 |
| 2048 | 29.35 | 3.88 |
| 4096 | 106.21 | 9.50 |

See: [bench_scaling_loglog.png](bench_scaling_loglog.png)

## Why It Matters
Attention scales quadratically in context length due to n×n interactions; spectral mixing scales subquadratically and remains tractable at longer contexts.

## Next Experiment
Train a tiny next-token model using spectral mixing in place of attention to validate end-to-end learnability.
