import logging
import os
import shutil
from datetime import datetime
import subprocess
from typing import Optional, Tuple
import numpy as np
import yaml


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


def get_git_revision_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except (subprocess.CalledProcessError, OSError):
        return "Unknown"


def setup_output_directory(config_path: str, experiment_name: str) -> str:
    """Creates timestamped output directory and saves simulation metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"{timestamp}_{experiment_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Archive config and version info
    shutil.copy(config_path, os.path.join(out_dir, "config_snapshot.yaml"))
    with open(os.path.join(out_dir, "commit_hash.txt"), "w") as f:
        f.write(get_git_revision_hash())

    return out_dir


def setup_logging(out_dir: str):
    """Initializes logging to both file and console."""
    log_file = os.path.join(out_dir, "benchmark.log")

    # Clear existing handlers to prevent duplication
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    logging.info(f"Log initialized: {log_file}")


# ==========================================
# I/O Helpers for Ground Truth
# ==========================================


def save_ground_truth(
    filepath: str,
    n_map_fine: np.ndarray,
    psi_0: np.ndarray,
    psi_gt: np.ndarray,
    beam_gt: Optional[np.ndarray],
):
    """Saves the high-res inputs and results to an .npz file."""
    logging.info(f"Saving Ground Truth to: {filepath}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    data_dict = {
        "n_map_fine": n_map_fine,
        "psi_0": psi_0,
        "psi_gt": psi_gt,
    }
    if beam_gt is not None:
        data_dict["beam_gt"] = beam_gt
    else:
        # Save a placeholder or handle None on load
        data_dict["beam_gt"] = np.array([])

    np.savez_compressed(filepath, **data_dict)


def load_ground_truth(
    filepath: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Loads n_map_fine, psi_0, psi_gt, and beam_gt from an .npz file."""
    logging.info(f"Loading Ground Truth from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"The specified Ground Truth file was not found: {filepath}"
        )

    with np.load(filepath) as data:
        n_map_fine = data["n_map_fine"]
        psi_0 = data["psi_0"]
        psi_gt = data["psi_gt"]

        # Handle optional beam_gt
        beam_gt = None
        if "beam_gt" in data:
            arr = data["beam_gt"]
            if arr.size > 0:
                beam_gt = arr

    return n_map_fine, psi_0, psi_gt, beam_gt
