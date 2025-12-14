import time
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def attention_mixer(X):
    """
    X: (n, d)
    Vanilla (single-head) attention block core:
        A = softmax(QK^T / sqrt(d))
        Y = A V
    Complexity ~ O(n^2 d)
    """
    n, d = X.shape
    # fixed random projections to keep it deterministic-ish per run
    Wq = np.random.randn(d, d).astype(np.float32) / np.sqrt(d)
    Wk = np.random.randn(d, d).astype(np.float32) / np.sqrt(d)
    Wv = np.random.randn(d, d).astype(np.float32) / np.sqrt(d)

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    scores = (Q @ K.T) / np.sqrt(d)          # (n, n)
    A = softmax(scores, axis=-1)             # (n, n)
    Y = A @ V                                # (n, d)
    return Y

def fft_mixer(X):
    """
    FFT token-mixing style:
        Y = Re(FFT2(X))
    Complexity ~ O(n d log n) (plus constants)
    """
    Y = np.fft.fft2(X).real.astype(np.float32)
    return Y

def time_fn(fn, X, warmup=2, iters=5):
    for _ in range(warmup):
        fn(X)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(X)
    t1 = time.perf_counter()
    return (t1 - t0) / iters

def main():
    d = 256
    ns = [128, 256, 512, 1024, 2048, 4096]   # bump up/down based on your machine
    print(f"d={d}")
    print("n, attention_ms, fft_ms")
    for n in ns:
        X = np.random.randn(n, d).astype(np.float32)

        # Important: reseed projections per n to avoid weird caching effects
        np.random.seed(0)
        att_ms = 1000 * time_fn(attention_mixer, X)

        np.random.seed(0)
        fft_ms = 1000 * time_fn(fft_mixer, X)

        print(f"{n}, {att_ms:.2f}, {fft_ms:.2f}")

if __name__ == "__main__":
    main()
