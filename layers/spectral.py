"""
This is the core architectural claim, in code.
No embeddings, no vocab, and no loss.
"""

import numpy as np

def spectral_mix_forward(X: np.ndarray, F: np.ndarray, T: int):
    """
    X: (T, d) real
    F: (T//2+1, d) complex spectral filter (learned)
    returns:
        H: (T, d) real
        cache: {"Z": Z, "T": T}
    """
    assert X.shape[0] == T, f"Expected X.shape[0]=={T}, got {X.shape[0]}"
    Z = np.fft.rfft(X, axis=0, norm="ortho")          # (T//2+1, d) complex
    Zf = Z * F                                        # elementwise complex
    H = np.fft.irfft(Zf, n=T, axis=0, norm="ortho")   # (T, d) real
    return H.astype(np.float32), {"Z": Z, "T": T}

def spectral_mix_backward(dH: np.ndarray, F: np.ndarray, cache: dict):
    """
    dH: (T, d) real gradient wrt H
    F:  (T//2+1, d) complex spectral filter
    cache: {"Z": Z, "T": T}

    returns:
        dX: (T, d) real
        dF: (T//2+1, d) complex
    """
    Z = cache["Z"]
    T = cache["T"]

    dZf = np.fft.rfft(dH, axis=0, norm="ortho")       # complex
    dF = dZf * np.conj(Z)                              # complex
    dZ = dZf * np.conj(F)                              # complex
    dX = np.fft.irfft(dZ, n=T, axis=0, norm="ortho").real.astype(np.float32)
    return dX, dF.astype(np.complex64)
