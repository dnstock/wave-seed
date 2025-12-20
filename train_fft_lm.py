import numpy as np
import time

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def rmsnorm(x, w, eps=1e-6):
    # x: (n, d), w: (d,)
    r2 = np.mean(x * x, axis=-1, keepdims=True) + eps
    inv_r = 1.0 / np.sqrt(r2)
    y = x * inv_r * w
    return y, inv_r

def rmsnorm_backward(dy, x, w, inv_r):
    # dy: (n, d), x: (n, d), w: (d,), inv_r: (n, 1)
    # y = x * inv_r * w
    # dx = dy*(w*inv_r) - x * (S/(d)) * inv_r^3, where S = sum(dy*x*w) per row
    n, d = x.shape
    wx = w * inv_r  # (n, d) via broadcast
    dwd = dy * (x * inv_r)  # (n, d)
    dw = np.sum(dwd, axis=0)  # (d,)

    S = np.sum(dy * x * w, axis=-1, keepdims=True)  # (n, 1)
    dx = dy * wx - x * (S * (inv_r ** 3) / d)
    return dx.astype(np.float32), dw.astype(np.float32)

def gelu(x):
    # tanh approximation
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def gelu_backward(dy, x):
    # derivative of tanh-approx GELU
    a = np.sqrt(2.0 / np.pi)
    x3 = x**3
    t = a * (x + 0.044715 * x3)
    th = np.tanh(t)
    sech2 = 1.0 - th * th
    dt_dx = a * (1.0 + 3.0 * 0.044715 * x * x)
    dgelu_dx = 0.5 * (1.0 + th) + 0.5 * x * sech2 * dt_dx
    return dy * dgelu_dx

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
    d_ff = 4 * d
    T = 256               # sequence length per step
    lr = 0.05
    steps = 500

    rng = np.random.default_rng(0)

    # Params: embedding + output
    E = (rng.standard_normal((V, d)).astype(np.float32) / np.sqrt(d))
    W = (rng.standard_normal((d, V)).astype(np.float32) / np.sqrt(d))
    b = np.zeros((V,), dtype=np.float32)

    # Output reconstruction head params
    w_norm1 = np.ones((d,), dtype=np.float32)
    w_norm2 = np.ones((d,), dtype=np.float32)

    W1 = (rng.standard_normal((d, d_ff)).astype(np.float32) / np.sqrt(d))
    b1 = np.zeros((d_ff,), dtype=np.float32)
    W2 = (rng.standard_normal((d_ff, d)).astype(np.float32) / np.sqrt(d_ff))
    b2 = np.zeros((d,), dtype=np.float32)

    # Learned spectral filter, start near-identity
    F = np.ones((T // 2 + 1, d), dtype=np.complex64)
    F += (0.01 * (rng.standard_normal(F.shape) + 1j * rng.standard_normal(F.shape))).astype(np.complex64)

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        # Sample a random window
        start = rng.integers(0, len(data) - (T + 1))
        x_idx = data[start : start + T]             # input
        y_idx = data[start + 1 : start + T + 1]     # next-char targets

        # Forward
        X = E[x_idx]                         # (T, d)
        H0, Z = spectral_mix(X, F, T)        # H0: (T, d)

        # Output reconstruction head (position-wise, O(n) in sequence length)
        H1, inv_r1 = rmsnorm(H0, w_norm1)            # (T, d)
        U = H1 @ W1 + b1                              # (T, d_ff)
        A = gelu(U)                                   # (T, d_ff)
        Vmid = A @ W2 + b2                             # (T, d)
        H2 = H1 + Vmid                                 # residual
        H3, inv_r2 = rmsnorm(H2, w_norm2)             # (T, d)

        logits = (H3 @ W) / np.sqrt(d) + b            # (T, V)
        probs = softmax(logits, axis=-1)
        loss = cross_entropy(probs, y_idx)

        # Backprop (manual, simple)
        dlogits = probs.copy()  # avoid mutating probs in-place during training backprop
        dlogits[np.arange(T), y_idx] -= 1.0
        dlogits /= T  # mean

        # Vocab projection grads (logits = (H3 @ W)/sqrt(d) + b)
        scale = 1.0 / np.sqrt(d)
        dW = (H3.T @ dlogits) * scale        # (d, V)
        db = np.sum(dlogits, axis=0)         # (V,)
        dH3 = (dlogits @ W.T) * scale        # (T, d)

        # Backprop through RMSNorm2
        dH2, dw_norm2 = rmsnorm_backward(dH3, H2, w_norm2, inv_r2)

        # Backprop through residual MLP
        dH1 = dH2.copy()                     # residual path into H1
        dVmid = dH2                           # (T, d)

        dW2 = A.T @ dVmid                     # (d_ff, d)
        db2 = np.sum(dVmid, axis=0)           # (d,)
        dA = dVmid @ W2.T                     # (T, d_ff)
        dU = gelu_backward(dA, U)             # (T, d_ff)
        dW1 = H1.T @ dU                       # (d, d_ff)
        db1 = np.sum(dU, axis=0)              # (d_ff,)
        dH1 += dU @ W1.T                      # accumulate into H1

        # Backprop through RMSNorm1
        dH0, dw_norm1 = rmsnorm_backward(dH1, H0, w_norm1, inv_r1)

        # Backprop through spectral mixing
        dZf = np.fft.rfft(dH0, axis=0, norm="ortho")          # (T//2+1, d) complex
        dF = dZf * np.conj(Z)                                  # (T//2+1, d) complex
        dZ = dZf * np.conj(F)                                  # (T//2+1, d) complex
        dX = np.fft.irfft(dZ, n=T, axis=0, norm="ortho").real.astype(np.float32)

        dE = np.zeros_like(E)
        np.add.at(dE, x_idx, dX)

        # SGD update
        E -= lr * dE
        W -= lr * dW
        b -= lr * db
        F -= (lr * dF).astype(np.complex64)
        w_norm1 -= lr * dw_norm1
        w_norm2 -= lr * dw_norm2
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

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
        H0, Z = spectral_mix(X, F, T)
        H1, _inv_r1 = rmsnorm(H0, w_norm1)
        U = H1 @ W1 + b1
        A = gelu(U)
        Vmid = A @ W2 + b2
        H2 = H1 + Vmid
        H3, _inv_r2 = rmsnorm(H2, w_norm2)
        logits = (H3 @ W) / np.sqrt(d) + b       # (T, V)

        temperature = 0.9
        top_k = 12

        logits_last = logits[-1] / temperature  # (V,)
        p = softmax(logits_last[None, :], axis=-1).ravel()

        # top-k filter
        top_k = min(top_k, V)  # guard against small vocab sizes
        top_idx = np.argpartition(p, -top_k)[-top_k:]
        p2 = np.zeros_like(p)
        p2[top_idx] = p[top_idx]
        p2 /= p2.sum()

        nxt = int(rng.choice(np.arange(V), p=p2))
        context.append(nxt)

    print("\n--- sample ---")
    print("".join(itos[i] for i in context))

if __name__ == "__main__":
    main()
