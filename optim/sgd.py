def sgd_step(params: dict, grads: dict, lr: float) -> None:
    for k, g in grads.items():
        params[k] -= lr * g
