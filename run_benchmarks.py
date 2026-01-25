import argparse
import os
from pyram_ptycho.benchmarking import run_full_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyRAM Benchmark Runner")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    if os.path.exists(args.config):
        run_full_benchmark(args.config)
    else:
        print(f"Error: Config file '{args.config}' not found.")
