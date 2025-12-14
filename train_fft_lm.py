import numpy as np
import time

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def one_hot(idx, vocab):
    x = np.zeros((idx.size, vocab), dtype=np.float32)
    x[np.arange(idx.size), idx] = 1.0
    return x

def spectral_mix(X, F, T):
    """
    X: (T, d) real
    F: (T//2+1, d) complex spectral filter (learned)
    returns H: (T, d) real
    """
    assert X.shape[0] == T, f"Expected X.shape[0]=={T}, got {X.shape[0]}"
    Z = np.fft.rfft(X, axis=0, norm="ortho")          # (T//2+1, d) complex
    Zf = Z * F                                        # elementwise complex
    H = np.fft.irfft(Zf, n=T, axis=0, norm="ortho")   # (T, d) real
    return H, Z

def cross_entropy(probs, y_idx):
    # probs: (T, V), y_idx: (T,)
    eps = 1e-9
    return -np.mean(np.log(probs[np.arange(y_idx.size), y_idx] + eps))

def main():
    # Tiny corpus
    text = (
        "in the beginning there was only waves. "
        "language is a construct, patterns are nature.\n"
    ) * 200

    # Char-level vocab
    chars = sorted(set(text))
    stoi = {c:i for i,c in enumerate(chars)}
    itos = {i:c for c,i in stoi.items()}
    V = len(chars)

    # Encode
    data = np.array([stoi[c] for c in text], dtype=np.int64)

    # Hyperparams (tiny)
    d = 128
    T = 256               # sequence length per step
    lr = 0.05
    steps = 500

    rng = np.random.default_rng(0)

    # Params: embedding + output
    E = (rng.standard_normal((V, d)).astype(np.float32) / np.sqrt(d))
    W = (rng.standard_normal((d, V)).astype(np.float32) / np.sqrt(d))
    b = np.zeros((V,), dtype=np.float32)

    # Learned spectral filter, start near-identity
    F = np.ones((T // 2 + 1, d), dtype=np.complex64)
    F += (0.01 * (rng.standard_normal(F.shape) + 1j * rng.standard_normal(F.shape))).astype(np.complex64)

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        # Sample a random window
        start = rng.integers(0, len(data) - (T + 1))
        x_idx = data[start : start + T]       # input
        y_idx = data[start + 1 : start + T + 1]  # next-char targets

        # Forward
        X = E[x_idx]                        # (T, d)
        H, Z = spectral_mix(X, F, T)
        logits = (H @ W) / np.sqrt(d) + b
        probs = softmax(logits, axis=-1)
        loss = cross_entropy(probs, y_idx)

        # Backprop (manual, simple)
        dlogits = probs
        dlogits[np.arange(T), y_idx] -= 1.0
        dlogits /= T  # mean

        dW = H.T @ dlogits                   # (d, V)
        db = np.sum(dlogits, axis=0)         # (V,)
        dH = dlogits @ W.T                   # (T, d)

        # Backprop through irfft(Z * F):
        # H = irfft(Zf), so dZf = rfft(dH)
        dZf = np.fft.rfft(dH, axis=0, norm="ortho")          # (T//2+1, d) complex

        # Zf = Z * F
        dF = dZf * np.conj(Z)                                # (T//2+1, d) complex
        dZ = dZf * np.conj(F)                                # (T//2+1, d) complex

        # Z = rfft(X) so dX = irfft(dZ)
        dX = np.fft.irfft(dZ, n=T, axis=0, norm="ortho").real.astype(np.float32)

        dE = np.zeros_like(E)
        np.add.at(dE, x_idx, dX)             # accumulate per embedding row

        # SGD update
        E -= lr * dE
        W -= lr * dW
        b -= lr * db
        F -= (lr * dF).astype(np.complex64)

        if step % 25 == 0:
            elapsed = time.perf_counter() - t0
            print(f"step {step:4d} | loss {loss:.4f} | elapsed {elapsed:.1f}s")

    # Quick sample
    seed = "in the "
    pad_id = stoi.get(" ", 0)
    context = [stoi[c] for c in seed]
    for _ in range(200):
        window = context[-T:]
        if len(window) < T:
            window = [pad_id] * (T - len(window)) + window
        x = np.array(window, dtype=np.int64)
        X = E[x]
        H, Z = spectral_mix(X, F, T)
        logits = (H @ W) / np.sqrt(d) + b
        p = softmax(logits[None, :], axis=-1).ravel()
        nxt = int(rng.choice(np.arange(V), p=p))
        context.append(nxt)

    print("\n--- sample ---")
    print("".join(itos[i] for i in context))

if __name__ == "__main__":
    main()
