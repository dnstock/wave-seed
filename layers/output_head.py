"""
This file answers: 
“How do spectral representations become legible text?”

It is the bridge that wires:
H → RMSNorm → MLP → Residual → RMSNorm → logits

Crucially:
No FFT, no embeddings, no attention, and no sampling
"""

import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def rmsnorm(x, w, eps=1e-6):
    # x: (n, d), w: (d,)
    r2 = np.mean(x * x, axis=-1, keepdims=True) + eps
    inv_r = 1.0 / np.sqrt(r2)
    y = x * inv_r * w
    return y.astype(np.float32), inv_r.astype(np.float32)

def rmsnorm_backward(dy, x, w, inv_r):
    # dy: (n, d), x: (n, d), w: (d,), inv_r: (n, 1)
    n, d = x.shape
    wx = w * inv_r
    dwd = dy * (x * inv_r)
    dw = np.sum(dwd, axis=0)

    S = np.sum(dy * x * w, axis=-1, keepdims=True)
    dx = dy * wx - x * (S * (inv_r ** 3) / d)
    return dx.astype(np.float32), dw.astype(np.float32)

def gelu(x):
    # tanh approximation
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def gelu_backward(dy, x):
    # derivative of tanh-approx GELU
    a = np.sqrt(2.0 / np.pi)
    t = a * (x + 0.044715 * x**3)
    th = np.tanh(t)
    sech2 = 1.0 - th * th
    dt_dx = a * (1.0 + 3.0 * 0.044715 * x * x)
    dgelu_dx = 0.5 * (1.0 + th) + 0.5 * x * sech2 * dt_dx
    return (dy * dgelu_dx).astype(np.float32)

def head_forward(H0, params, d):
    """
    H0: (T, d)
    params: dict with w_norm1,w_norm2,W1,b1,W2,b2,Wv,bv
    returns logits (T, V), cache
    """
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
        "H0": H0,
        "H1": H1,
        "U": U,
        "A": A,
        "H2": H2,
        "H3": H3,
        "inv_r1": inv_r1,
        "inv_r2": inv_r2,
    }
    return logits.astype(np.float32), cache

def head_backward(dlogits, params, cache, d):
    """
    dlogits: (T, V)
    returns:
        dH0: (T, d)
        grads: dict matching params keys
    """
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
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "Wv": dWv,
        "bv": dbv,
    }
    return dH0.astype(np.float32), grads
