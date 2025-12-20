import numpy as np

from data.char_dataset import make_vocab, encode, decode
from layers.spectral import spectral_mix_forward
from layers.output_head import head_forward, softmax
from model.embedding import embed_lookup

def main():
    ckpt = np.load("ckpt_wave_seed.npz", allow_pickle=True)

    E = ckpt["E"]
    F = ckpt["F"]

    head_params = {
        "w_norm1": ckpt["w_norm1"],
        "w_norm2": ckpt["w_norm2"],
        "W1": ckpt["W1"],
        "b1": ckpt["b1"],
        "W2": ckpt["W2"],
        "b2": ckpt["b2"],
        "Wv": ckpt["Wv"],
        "bv": ckpt["bv"],
    }

    text = str(ckpt["text"])
    _chars, stoi, itos = make_vocab(text)
    V = len(stoi)

    d = E.shape[1]
    T = 256

    seed = "in the "
    pad_id = stoi.get(" ", 0)
    rng = np.random.default_rng(0)

    context = [stoi.get(c, pad_id) for c in seed]

    for _ in range(200):
        window = context[-T:]
        if len(window) < T:
            window = [pad_id] * (T - len(window)) + window

        x = np.array(window, dtype=np.int64)
        X = embed_lookup(E, x)
        H0, _ = spectral_mix_forward(X, F, T)
        logits, _ = head_forward(H0, head_params, d)

        temperature = 0.9
        top_k = 12

        logits_last = logits[-1] / temperature
        p = softmax(logits_last[None, :], axis=-1).ravel()

        top_k = min(top_k, V)
        top_idx = np.argpartition(p, -top_k)[-top_k:]
        p2 = np.zeros_like(p)
        p2[top_idx] = p[top_idx]
        p2 /= p2.sum()

        nxt = int(rng.choice(np.arange(V), p=p2))
        context.append(nxt)

    print(decode(context, itos))

if __name__ == "__main__":
    main()
