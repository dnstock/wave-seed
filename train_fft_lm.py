"""Compatibility entrypoint.

This repo has been modularized. Use:
    - `python train.py` to train
    - `python sample.py` to sample

This file remains as a convenience alias so existing commands keep working.

Usage:
    - `python train_fft_lm.py`         -> runs train.main()
    - `python train_fft_lm.py --sample` -> runs sample.main()
"""

import sys


def main() -> None:
    argv = sys.argv[1:]

    if "--sample" in argv:
        from sample import main as sample_main

        sample_main()
        return

    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
