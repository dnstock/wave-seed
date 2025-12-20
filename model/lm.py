import numpy as np

from layers.spectral import spectral_mix_forward, spectral_mix_backward
from layers.output_head import head_forward, head_backward, softmax

def cross_entropy(probs: np.ndarray, y_idx: np.ndarray) -> float:
    eps = 1e-9
    return float(-np.mean(np.log(probs[np.arange(y_idx.size), y_idx] + eps)))

def forward(E, F, head_params, x_idx, T, d):
    X = E[x_idx]  # (T,d)
    H0, spec_cache = spectral_mix_forward(X, F, T)
    logits, head_cache = head_forward(H0, head_params, d)
    return logits, {"spec": spec_cache, "head": head_cache, "x_idx": x_idx}

def backward(dlogits, F, head_params, cache, T):
    dH0, head_grads = head_backward(dlogits, head_params, cache["head"], head_params["Wv"].shape[0])
    dX, dF = spectral_mix_backward(dH0, F, cache["spec"])
    return dX, dF, head_grads
