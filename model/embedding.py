import numpy as np

def embed_lookup(E: np.ndarray, x_idx: np.ndarray) -> np.ndarray:
    return E[x_idx]

def embed_backward(E: np.ndarray, x_idx: np.ndarray, dX: np.ndarray) -> np.ndarray:
    dE = np.zeros_like(E)
    np.add.at(dE, x_idx, dX)
    return dE
