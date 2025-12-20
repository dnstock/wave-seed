import numpy as np

def build_tiny_corpus(repeats: int = 200) -> str:
    return (
        "in the beginning there was only waves. "
        "language is a construct, patterns are nature.\n"
    ) * repeats

def make_vocab(text: str):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return chars, stoi, itos

def encode(text: str, stoi: dict) -> np.ndarray:
    return np.array([stoi[c] for c in text], dtype=np.int64)

def decode(ids, itos: dict) -> str:
    return "".join(itos[int(i)] for i in ids)
