import argparse
import csv
import math

import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark plotter to graph performance and memory results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--csv", type=str, default="bench_results.csv", help="Base filename for results")
    ap.add_argument("--out", type=str, default="bench_scaling.png", help="Base filename for the plot")
    args = ap.parse_args()

    T, att_ms, loc_ms, fft_ms, att_mb, loc_mb, fft_mb = [], [], [], [], [], [], []

    with open(args.csv, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            T.append(int(row["T"]))
            att_ms.append(float(row["attention_ms"]))
            loc_ms.append(float(row["local_attention_ms"]))
            fft_ms.append(float(row["fft_ms"]))
            att_mb.append(float(row["attention_peak_bytes"]) / 1e6)
            loc_mb.append(float(row["local_attention_peak_bytes"]) / 1e6)
            fft_mb.append(float(row["fft_peak_bytes"]) / 1e6)

    plt.figure()
    plt.plot(T, att_ms, marker="o", label="attention (ms)")
    plt.plot(T, loc_ms, marker="o", label="local attention (ms)")
    plt.plot(T, fft_ms, marker="o", label="fft (ms)")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("sequence length T (log2)")
    plt.ylabel("time per op (ms, log)")
    plt.title("Scaling: Attention vs FFT mixer (NumPy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

    out2 = args.out.replace(".png", "_mem.png")
    plt.figure()
    plt.plot(T, att_mb, marker="o", label="attention peak (MB)")
    plt.plot(T, loc_mb, marker="o", label="local attention peak (MB)")
    plt.plot(T, fft_mb, marker="o", label="fft peak (MB)")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("sequence length T (log2)")
    plt.ylabel("peak bytes (MB, log)")
    plt.title("Peak alloc trend: Attention vs FFT mixer (tracemalloc)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out2, dpi=200)

    print(f"wrote: {args.out}")
    print(f"wrote: {out2}")

if __name__ == "__main__":
    main()
