import logging
import os
import shutil
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

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
    logging.info(f"Loaded 3D Config: {config_path}")
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
    out_dir = os.path.join("results_3d", f"{timestamp}_{experiment_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Archive config and version info
    shutil.copy(config_path, os.path.join(out_dir, "config_snapshot.yaml"))
    with open(os.path.join(out_dir, "commit_hash.txt"), "w") as f:
        f.write(get_git_revision_hash())

    return out_dir


def setup_logging(out_dir: str):
    """Initializes logging to both file and console."""
    log_file = os.path.join(out_dir, "benchmark_3d.log")

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
    logging.info(f"3D Log initialized: {log_file}")


# ==========================================
# I/O Helpers for 3D Ground Truth
# ==========================================


def save_ground_truth(
    filepath: str,
    n_map_fine: np.ndarray,
    psi_0: np.ndarray,
    psi_gt: np.ndarray,
    beam_gt: Optional[np.ndarray],
):
    """
    Saves the high-res 3D inputs and results to an .npz file.

    Shapes:
    - n_map_fine: (Ny, Nx, Nz)
    - psi_0: (Ny, Nx)
    - psi_gt: (Ny, Nx)
    - beam_gt: (Ny, Nx, Nz)
    """
    logging.info(f"Saving 3D Ground Truth to: {filepath}")
    logging.debug(f"Shapes -> n_map: {n_map_fine.shape}, psi_0: {psi_0.shape}")

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

    # Use savez_compressed for 3D data as it is usually highly compressible and saves massive disk space
    np.savez_compressed(filepath, **data_dict)

    # Calculate filesize for logging
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    logging.info(f"Ground truth saved successfully. File size: {file_size_mb:.2f} MB")


def load_ground_truth(
    filepath: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Loads n_map_fine, psi_0, psi_gt, and beam_gt from an .npz file.
    """
    logging.info(f"Loading 3D Ground Truth from: {filepath}")

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

    logging.info(
        f"Loaded 3D Data -> map: {n_map_fine.shape}, "
        f"probe: {psi_0.shape}, "
        f"exit_wave: {psi_gt.shape}"
    )

    return n_map_fine, psi_0, psi_gt, beam_gt


def parse_simulation_parameters(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and converts all physical parameters from YAML (Microns) to SI units (Meters).
    Assumes an isotropic transverse 3D grid (Nx = Ny, dx = dy).
    Returns a flat dictionary of simulation constants.
    """
    um = 1e-6

    # Physics
    phys = cfg["physics"]
    wavelength = phys["wavelength_um"] * um
    probe_dia = phys["probe_dia_um"] * um
    probe_focus = phys["probe_focus_um"] * um

    # Transverse Grid (Isotropic X/Y assumed for 3D)
    grid = cfg["grid"]
    n_physical = grid["n_physical"]
    n_pad = grid["n_padding"]
    physical_width = grid["physical_width_um"] * um

    # Derived Grid Props
    n_total = n_physical + 2 * n_pad
    dx = physical_width / n_physical  # Applies to both dx and dy
    total_width = dx * n_total

    # Sample (Longitudinal Z-axis)
    sample = cfg["sample"]
    thickness = sample["thickness_um"] * um

    return {
        "wavelength": wavelength,
        "probe_dia": probe_dia,
        "probe_focus": probe_focus,
        "n_physical": n_physical,
        "n_pad": n_pad,
        "n_total": n_total,
        "physical_width": physical_width,
        "dx": dx,  # Applied symmetrically to X and Y
        "total_width": total_width,
        "sample_thickness": thickness,
        "sample_type": sample["type"],
        "sample_params": sample.get("params", {}),
        "ground_truth_cfg": sample.get("ground_truth", {}),
    }
