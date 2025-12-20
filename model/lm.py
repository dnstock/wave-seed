import numpy as np

from layers.spectral import spectral_mix_forward, spectral_mix_backward
from layers.output_head import head_forward, head_backward, softmax
from model.embedding import embed_lookup, embed_backward

def cross_entropy(probs: np.ndarray, y_idx: np.ndarray) -> float:
    eps = 1e-9
    return float(-np.mean(np.log(probs[np.arange(y_idx.size), y_idx] + eps)))

def forward(E, F, head_params, x_idx: np.ndarray, T: int, d: int):
    X = embed_lookup(E, x_idx)  # (T, d)
    H0, spec_cache = spectral_mix_forward(X, F, T)
    logits, head_cache = head_forward(H0, head_params, d)
    cache = {"x_idx": x_idx, "spec": spec_cache, "head": head_cache}
    return logits, cache

def backward(E, F, head_params, dlogits: np.ndarray, cache: dict, T: int, d: int):
    dH0, head_grads = head_backward(dlogits, head_params, cache["head"], d)
    dX, dF = spectral_mix_backward(dH0, F, cache["spec"])
    dE = embed_backward(E, cache["x_idx"], dX)
    return dE, dF, head_grads

def loss_and_grad(logits: np.ndarray, y_idx: np.ndarray):
    probs = softmax(logits, axis=-1)
    loss = cross_entropy(probs, y_idx)

    T = y_idx.size
    dlogits = probs.copy()
    dlogits[np.arange(T), y_idx] -= 1.0
    dlogits /= T
    return loss, dlogits
