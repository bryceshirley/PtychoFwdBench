import argparse
import logging
import os

from PtychoFwdBench.ptycho_fwd_bench.utils import parse_config

from ptycho_fwd_bench.benchmarking import run_full_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyRAM Benchmark Runner")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    if os.path.exists(args.config):
        cfg, out_dir = parse_config(args.config)
        run_full_benchmark(cfg, out_dir)
    else:
        logging.error(f"Error: Config file '{args.config}' not found.")
