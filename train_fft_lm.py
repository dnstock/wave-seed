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

def fft_mix(X):
    # X: (T, d)
    return np.fft.fft2(X).real.astype(np.float32)

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
    lr = 0.2
    steps = 300

    rng = np.random.default_rng(0)

    # Params: embedding + output
    E = (rng.standard_normal((V, d)).astype(np.float32) / np.sqrt(d))
    W = (rng.standard_normal((d, V)).astype(np.float32) / np.sqrt(d))
    b = np.zeros((V,), dtype=np.float32)

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        # Sample a random window
        start = rng.integers(0, len(data) - (T + 1))
        x_idx = data[start : start + T]       # input
        y_idx = data[start + 1 : start + T + 1]  # next-char targets

        # Forward
        X = E[x_idx]                 # (T, d)
        H = fft_mix(X)               # (T, d)  <-- attention replacement
        logits = H @ W + b           # (T, V)
        probs = softmax(logits, axis=-1)
        loss = cross_entropy(probs, y_idx)

        # Backprop (manual, simple)
        dlogits = probs
        dlogits[np.arange(T), y_idx] -= 1.0
        dlogits /= T  # mean

        dW = H.T @ dlogits                   # (d, V)
        db = np.sum(dlogits, axis=0)         # (V,)
        dH = dlogits @ W.T                   # (T, d)

        # IMPORTANT: fft_mix is not differentiable as written (real(fft2))
        # For this seed, we treat it as a fixed mixing operator and pass gradients through as identity.
        # (Next iteration: use a proper linear operator with known Jacobian.)
        dX = dH

        dE = np.zeros_like(E)
        np.add.at(dE, x_idx, dX)             # accumulate per embedding row

        # SGD update
        E -= lr * dE
        W -= lr * dW
        b -= lr * db

        if step % 25 == 0:
            elapsed = time.perf_counter() - t0
            print(f"step {step:4d} | loss {loss:.4f} | elapsed {elapsed:.1f}s")

    # Quick sample
    seed = "in the "
    context = [stoi[c] for c in seed]
    for _ in range(200):
        x = np.array(context[-T:], dtype=np.int64)
        X = E[x]
        H = fft_mix(X)
        logits = (H[-1] @ W + b)
        p = softmax(logits[None, :], axis=-1).ravel()
        nxt = int(rng.choice(np.arange(V), p=p))
        context.append(nxt)

    print("\n--- sample ---")
    print("".join(itos[i] for i in context))

if __name__ == "__main__":
    main()
