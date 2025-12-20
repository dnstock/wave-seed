def sgd_step(params: dict, grads: dict, lr: float):
    for k, g in grads.items():
        params[k] -= lr * g
