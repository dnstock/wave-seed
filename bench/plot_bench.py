import argparse
import csv
import re
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark plotter to graph performance and memory results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--csv", type=str, default="bench_results.csv", help="Base filename for results")
    ap.add_argument("--latest", action="store_true", help="Use the newest timestamped CSV and sync output filenames")
    ap.add_argument("--out", type=str, default="bench_scaling.png", help="Base filename for the plot")
    args = ap.parse_args()

    # --- Path Resolution Logic ---
    script_dir = Path(__file__).parent.absolute()
    results_dir = script_dir / "results"

    csv_file = Path(args.csv)
    timestamp_suffix = ""

    if args.latest:
        # If --latest is set, search results/ for files matching "base_*.csv"
        base_name = csv_file.stem  # e.g., "bench_results"
        pattern = f"{base_name}_*.csv"

        # Filter out sample files and ensure they match the timestamp pattern
        regex = r"(\d{8}_\d{4})"
        matching_files = [
            f for f in results_dir.glob(pattern)
            if "_samples" not in f.name and re.search(regex, f.name)
        ]

        if not matching_files:
            print(f"Error: No timestamped files found for '{base_name}' in {results_dir}. Ensure timestamps are appended in the format '_YYYYMMDD_HHMM'")
            return

        # Sort based on the timestamp string extracted by the regex
        csv_file = max(matching_files, key=lambda f: re.search(regex, f.name).group(1))
        timestamp_suffix = f"_{re.search(regex, csv_file.name).group(1)}"

        print(f"Using most recent timestamped file: {csv_file.name}")
    else:
        # If not using --latest, check if the path is relative to results/ or literal
        if not csv_file.exists():
            csv_file = results_dir / args.csv

        if not csv_file.exists():
            print(f"Error: Could not find file {args.csv}")
            return

    # --- Data Loading ---
    T, att_ms, loc_ms, fft_ms, att_mb, loc_mb, fft_mb = [], [], [], [], [], [], []

    with open(csv_file, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            T.append(int(row["T"]))
            att_ms.append(float(row["attention_ms"]))
            loc_ms.append(float(row["local_attention_ms"]))
            fft_ms.append(float(row["fft_ms"]))
            att_mb.append(float(row["attention_peak_bytes"]) / 1e6)
            loc_mb.append(float(row["local_attention_peak_bytes"]) / 1e6)
            fft_mb.append(float(row["fft_peak_bytes"]) / 1e6)

    # --- Plotting Logic ---
    # Construct output names with timestamp if --latest was used
    out_path = Path(args.out)
    final_out_name = f"{out_path.stem}{timestamp_suffix}{out_path.suffix}"
    final_mem_name = f"{out_path.stem}{timestamp_suffix}_mem{out_path.suffix}"

    out_perf = results_dir / final_out_name
    out_mem = results_dir / final_mem_name

    plt.figure()
    plt.plot(T, att_ms, marker="o", label="attention (ms)")
    plt.plot(T, loc_ms, marker="o", label="local attention (ms)")
    plt.plot(T, fft_ms, marker="o", label="fft (ms)")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("sequence length T (log2)")
    plt.ylabel("time per op (ms, log)")
    plt.title(f"Scaling: Attention vs FFT mixer (NumPy)\nSource: {csv_file.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_perf, dpi=200)

    plt.figure()
    plt.plot(T, att_mb, marker="o", label="attention peak (MB)")
    plt.plot(T, loc_mb, marker="o", label="local attention peak (MB)")
    plt.plot(T, fft_mb, marker="o", label="fft peak (MB)")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("sequence length T (log2)")
    plt.ylabel("peak bytes (MB, log)")
    plt.title(f"Peak alloc trend: Attention vs FFT mixer (tracemalloc)\nSource: {csv_file.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_mem, dpi=200)

    # --- Summary Printout ---
    print(f"\n" + "="*50)
    print(f"PLOTS COMPLETE")
    print("-"*50)
    print(f"Data source:\n{csv_file}")
    print(f"Performance plot:\n{out_perf.resolve()}")
    print(f"Memory plot:\n{out_mem.resolve()}")
    print("="*50)

if __name__ == "__main__":
    main()
