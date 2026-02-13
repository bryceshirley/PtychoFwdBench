import argparse
import logging
import os

from ptycho_fwd_bench.dim_2.benchmarking import run_full_benchmark
from ptycho_fwd_bench.dim_2.utils.utils import parse_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyRAM Benchmark Runner")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    if os.path.exists(args.config):
        cfg, out_dir = parse_config(args.config)
        run_full_benchmark(cfg, out_dir)
    else:
        logging.error(f"Error: Config file '{args.config}' not found.")
