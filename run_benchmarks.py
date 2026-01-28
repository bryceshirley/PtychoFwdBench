import argparse
import logging
import os
import yaml
from PtychoFwdBench.ptycho_fwd_bench.utils import setup_logging, setup_output_directory
from ptycho_fwd_bench.benchmarking import run_full_benchmark


def parse_config(config_path: str) -> dict:
    """
    Parse the YAML configuration file and set up output directory and logging.

    Parameters:
    - config_path: Path to the YAML configuration file.

    Returns:
    - cfg: Parsed configuration dictionary.
    - out_dir: Output directory path.
    """
    # 1. Load & Setup
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = setup_output_directory(config_path, cfg["experiment"]["name"])
    setup_logging(out_dir)
    logging.info(f"Loaded Config: {config_path}")
    return cfg, out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyRAM Benchmark Runner")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    if os.path.exists(args.config):
        cfg, out_dir = parse_config(args.config)
        run_full_benchmark(cfg, out_dir)
    else:
        logging.error(f"Error: Config file '{args.config}' not found.")
