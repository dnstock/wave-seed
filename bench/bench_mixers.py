import argparse
import csv
import math
import time
import tracemalloc
from pathlib import Path
from datetime import datetime

import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def attention_mix(X: np.ndarray) -> np.ndarray:
    """
    Minimal attention-style mixer (no learned QKV projection).
    Y = softmax( X X^T / sqrt(d) ) X

    X: (T, d)
    returns Y: (T, d)
    """
    T, d = X.shape
    scores = (X @ X.T) / math.sqrt(d)    # (T, T)
    P = softmax(scores, axis=-1)         # (T, T)
    Y = P @ X                            # (T, d)
    return Y

def attention_local_mix(X: np.ndarray, w: int) -> np.ndarray:
    """
    Windowed attention mixer with fixed window size w.

    For each position i, attends only to [i-w//2, i+w//2).
    Complexity ~ O(T * w * d). Memory ~ O(T * w).

    X: (T, d)
    returns Y: (T, d)
    """
    T, d = X.shape
    half = w // 2
    Y = np.empty((T, d), dtype=np.float32)

    for i in range(T):
        j0 = max(0, i - half)
        j1 = min(T, i - half + w)
        K = X[j0:j1]  # (L, d)
        q = X[i]      # (d,)

        scores = (K @ q) / math.sqrt(d)         # (L,)
        p = softmax(scores[None, :], axis=-1).ravel()  # (L,)
        Y[i] = (p[:, None] * K).sum(axis=0)     # (d,)

    return Y

def fft_mix(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Spectral mixer: rFFT -> elementwise multiply -> irFFT
    X: (T, d) real
    F: (T//2+1, d) complex
    """
    T = X.shape[0]
    Z = np.fft.rfft(X, axis=0, norm="ortho")          # (T//2+1, d)
    Zf = Z * F
    Y = np.fft.irfft(Zf, n=T, axis=0, norm="ortho")   # (T, d)
    return Y.real.astype(np.float32)

def bench_one(fn, X, warmup: int, iters: int):
    # warmup (JIT/caches/FFT planning effects)
    for _ in range(warmup):
        fn(X)

    # timing
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(X)
    dt = time.perf_counter() - t0
    return (dt / iters) * 1000.0  # ms/op

def peak_mem_one(fn, X, warmup: int, iters: int) -> int:
    # tracemalloc measures Python allocations; for NumPy it’s not perfect,
    # but it’s still useful for comparative peak trends on the same machine.
    tracemalloc.start()

    for _ in range(warmup):
        fn(X)
    for _ in range(iters):
        fn(X)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak  # bytes

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark performance and memory for Attention, Local Attention, and FFT mixers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Grouping arguments for better readability in --help
    dims = ap.add_argument_group("Model & Sequence Dimensions")
    dims.add_argument("--d", type=int, default=256, help="Hidden dimension size")
    dims.add_argument("--tmin", type=int, default=128, help="Minimum sequence length")
    dims.add_argument("--tmax", type=int, default=4096, help="Maximum sequence length")
    dims.add_argument("--w", type=int, default=128, help="Window size for local attention")

    timing = ap.add_argument_group("Benchmarking Settings")
    timing.add_argument("--warmup", type=int, default=5, help="Warmup iterations per sequence length")
    timing.add_argument("--iters", type=int, default=20, help="Benchmark iterations to average per run")
    timing.add_argument("--seed", type=int, default=0, help="Random seed for data generation")

    output = ap.add_argument_group("Output")
    output.add_argument("--csv", type=str, default="bench_results.csv", help="Filename to store results")

    args = ap.parse_args()

    # --- Directory & Path Handling ---
    script_dir = Path(__file__).parent.absolute()
    results_dir = script_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    results_file = results_dir / f"{Path(args.csv).stem}_{ts}.csv"

    # --- Execution Logic ---
    rng = np.random.default_rng(args.seed)

    rows = []
    T = args.tmin
    while T <= args.tmax:
        X = (rng.standard_normal((T, args.d)).astype(np.float32)) / math.sqrt(args.d)
        F = np.ones((T // 2 + 1, args.d), dtype=np.complex64)

        att_fn = lambda x: attention_mix(x)
        loc_fn = lambda x: attention_local_mix(x, args.w)
        fft_fn = lambda x: fft_mix(x, F)

        att_ms = bench_one(att_fn, X, args.warmup, args.iters)
        loc_ms = bench_one(loc_fn, X, args.warmup, args.iters)
        fft_ms = bench_one(fft_fn, X, args.warmup, args.iters)

        att_peak = peak_mem_one(att_fn, X, max(1, args.warmup // 2), max(3, args.iters // 3))
        loc_peak = peak_mem_one(loc_fn, X, max(1, args.warmup // 2), max(3, args.iters // 3))
        fft_peak = peak_mem_one(fft_fn, X, max(1, args.warmup // 2), max(3, args.iters // 3))

        rows.append((T, args.d, att_ms, loc_ms, fft_ms, att_peak, loc_peak, fft_peak))
        print(
            f"T={T:5d} d={args.d:4d} | "
            f"att {att_ms:8.2f} ms | loc(w={args.w}) {loc_ms:8.2f} ms | fft {fft_ms:8.2f} ms | "
            f"peak att {att_peak/1e6:7.1f} MB | peak loc {loc_peak/1e6:7.1f} MB | peak fft {fft_peak/1e6:7.1f} MB"
        )

        T *= 2

    # --- Save File ---
    with open(results_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "T", "d",
            "attention_ms", "local_attention_ms", "fft_ms",
            "attention_peak_bytes", "local_attention_peak_bytes", "fft_peak_bytes"
        ])
        w.writerows(rows)

    # --- Summary Printout ---
    print(f"\n" + "="*50)
    print(f"BENCHMARK COMPLETE")
    print("-"*50)
    print(f"Results:\nfile://{results_file.resolve()}")
    print("="*50)

if __name__ == "__main__":
    main()
