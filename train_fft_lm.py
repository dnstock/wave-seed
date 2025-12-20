import numpy as np
import time

# Optional modular imports
try:
    from layers.spectral import spectral_mix_forward, spectral_mix_backward  # type: ignore
    from layers.output_head import head_forward, head_backward, softmax      # type: ignore
    USING_LAYERS_MODULES = True
except Exception:
    USING_LAYERS_MODULES = False

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

    def spectral_mix_forward(X, F, T):
        """
        X: (T, d) real
        F: (T//2+1, d) complex spectral filter (learned)
        returns H: (T, d) real
        """
        assert X.shape[0] == T, f"Expected X.shape[0]=={T}, got {X.shape[0]}"
        Z = np.fft.rfft(X, axis=0, norm="ortho")          # (T//2+1, d) complex
        Zf = Z * F                                        # elementwise complex
        H = np.fft.irfft(Zf, n=T, axis=0, norm="ortho")   # (T, d) real
        return H.astype(np.float32), {"Z": Z, "T": T}

    def spectral_mix_backward(dH, F, cache):
        Z = cache["Z"]
        T = cache["T"]

        dZf = np.fft.rfft(dH, axis=0, norm="ortho")
        dF = dZf * np.conj(Z)
        dZ = dZf * np.conj(F)
        dX = np.fft.irfft(dZ, n=T, axis=0, norm="ortho").real.astype(np.float32)
        return dX, dF.astype(np.complex64)

    def head_forward(H0, params, d):
        # params: dict with w_norm1,w_norm2,W1,b1,W2,b2,Wv,bv
        w_norm1 = params["w_norm1"]
        w_norm2 = params["w_norm2"]
        W1, b1 = params["W1"], params["b1"]
        W2, b2 = params["W2"], params["b2"]
        Wv, bv = params["Wv"], params["bv"]

        H1, inv_r1 = rmsnorm(H0, w_norm1)
        U = H1 @ W1 + b1
        A = gelu(U)
        Vmid = A @ W2 + b2
        H2 = H1 + Vmid
        H3, inv_r2 = rmsnorm(H2, w_norm2)

        logits = (H3 @ Wv) / np.sqrt(d) + bv

        cache = {
            "H0": H0, "H1": H1, "U": U, "A": A, "H2": H2, "H3": H3,
            "inv_r1": inv_r1, "inv_r2": inv_r2,
        }
        return logits.astype(np.float32), cache


    def head_backward(dlogits, params, cache, d):
        w_norm1 = params["w_norm1"]
        w_norm2 = params["w_norm2"]
        W1 = params["W1"]
        W2 = params["W2"]
        Wv = params["Wv"]

        H0 = cache["H0"]
        H1 = cache["H1"]
        U = cache["U"]
        A = cache["A"]
        H2 = cache["H2"]
        H3 = cache["H3"]
        inv_r1 = cache["inv_r1"]
        inv_r2 = cache["inv_r2"]

        scale = 1.0 / np.sqrt(d)

        dWv = (H3.T @ dlogits) * scale
        dbv = np.sum(dlogits, axis=0)
        dH3 = (dlogits @ Wv.T) * scale

        dH2, dw_norm2 = rmsnorm_backward(dH3, H2, w_norm2, inv_r2)

        dH1 = dH2.copy()
        dVmid = dH2

        dW2 = A.T @ dVmid
        db2 = np.sum(dVmid, axis=0)
        dA = dVmid @ W2.T
        dU = gelu_backward(dA, U)
        dW1 = H1.T @ dU
        db1 = np.sum(dU, axis=0)
        dH1 += dU @ W1.T

        dH0, dw_norm1 = rmsnorm_backward(dH1, H0, w_norm1, inv_r1)

        grads = {
            "w_norm1": dw_norm1,
            "w_norm2": dw_norm2,
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "Wv": dWv, "bv": dbv,
        }
        return dH0.astype(np.float32), grads

def one_hot(idx, vocab):
    x = np.zeros((idx.size, vocab), dtype=np.float32)
    x[np.arange(idx.size), idx] = 1.0
    return x

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

    # Params: embedding + vocab projection
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

    head_params = {
        "w_norm1": w_norm1,
        "w_norm2": w_norm2,
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2,
        "Wv": W,
        "bv": b,
    }

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
        H0, spec_cache = spectral_mix_forward(X, F, T)        # (T, d)
        logits, head_cache = head_forward(H0, head_params, d)  # (T, V)

        probs = softmax(logits, axis=-1)
        loss = cross_entropy(probs, y_idx)

        # Backprop (manual, simple)
        dlogits = probs.copy()  # avoid mutating probs in-place during training backprop
        dlogits[np.arange(T), y_idx] -= 1.0
        dlogits /= T  # mean

        # Backprop through output head (includes vocab projection)
        dH0, head_grads = head_backward(dlogits, head_params, head_cache, d)

        # Backprop through spectral mixing
        dX, dF = spectral_mix_backward(dH0, F, spec_cache)

        dE = np.zeros_like(E)
        np.add.at(dE, x_idx, dX)

        # SGD update
        E -= lr * dE

        # Apply head grads (includes vocab projection)
        head_params["Wv"] -= lr * head_grads["Wv"]
        head_params["bv"] -= lr * head_grads["bv"]
        head_params["w_norm1"] -= lr * head_grads["w_norm1"]
        head_params["w_norm2"] -= lr * head_grads["w_norm2"]
        head_params["W1"] -= lr * head_grads["W1"]
        head_params["b1"] -= lr * head_grads["b1"]
        head_params["W2"] -= lr * head_grads["W2"]
        head_params["b2"] -= lr * head_grads["b2"]

        # Keep convenience refs in sync (optional, but preserves existing variable usage elsewhere)
        W = head_params["Wv"]
        b = head_params["bv"]
        w_norm1 = head_params["w_norm1"]
        w_norm2 = head_params["w_norm2"]
        W1, b1 = head_params["W1"], head_params["b1"]
        W2, b2 = head_params["W2"], head_params["b2"]

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
        H0, _spec_cache = spectral_mix_forward(X, F, T)
        logits, _head_cache = head_forward(H0, head_params, d)

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
