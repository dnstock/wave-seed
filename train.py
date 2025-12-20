import numpy as np
import time

from data.char_dataset import build_tiny_corpus, make_vocab, encode
from model.lm import forward, backward, loss_and_grad
from optim.sgd import sgd_step

def init_params(rng: np.random.Generator, V: int, d: int, d_ff: int, T: int):
    E = (rng.standard_normal((V, d)).astype(np.float32) / np.sqrt(d))
    Wv = (rng.standard_normal((d, V)).astype(np.float32) / np.sqrt(d))
    bv = np.zeros((V,), dtype=np.float32)

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
        "Wv": Wv,
        "bv": bv,
    }

    F = np.ones((T // 2 + 1, d), dtype=np.complex64)
    F += (0.01 * (rng.standard_normal(F.shape) + 1j * rng.standard_normal(F.shape))).astype(np.complex64)

    return E, F, head_params

def main():
    # Data
    text = build_tiny_corpus(repeats=200)
    _chars, stoi, itos = make_vocab(text)
    data = encode(text, stoi)
    V = len(stoi)

    # Hyperparams
    d = 128
    d_ff = 4 * d
    T = 256
    lr = 0.05
    steps = 500

    rng = np.random.default_rng(0)
    E, F, head_params = init_params(rng, V, d, d_ff, T)

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        start = rng.integers(0, len(data) - (T + 1))
        x_idx = data[start : start + T]
        y_idx = data[start + 1 : start + T + 1]

        logits, cache = forward(E, F, head_params, x_idx, T, d)
        loss, dlogits = loss_and_grad(logits, y_idx)

        dE, dF, head_grads = backward(E, F, head_params, dlogits, cache, T, d)

        E -= lr * dE
        sgd_step(head_params, head_grads, lr)
        F -= (lr * dF).astype(np.complex64)

        if step % 25 == 0:
            elapsed = time.perf_counter() - t0
            print(f"step {step:4d} | loss {loss:.4f} | elapsed {elapsed:.1f}s")

    # Save a simple checkpoint for sample.py
    np.savez(
        "ckpt_wave_seed.npz",
        E=E,
        F=F,
        **{k: v for k, v in head_params.items()},
        text=text,
    )
    print("saved: ckpt_wave_seed.npz")

if __name__ == "__main__":
    main()
