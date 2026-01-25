import numpy as np
import logging
import warnings
from typing import Dict, Any


def parse_simulation_parameters(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and converts all physical parameters from YAML (Microns) to SI units (Meters).
    Returns a flat dictionary of simulation constants.
    """
    um = 1e-6

    # Physics
    phys = cfg["physics"]
    wavelength = phys["wavelength_um"] * um
    probe_dia = phys["probe_dia_um"] * um
    probe_focus = phys["probe_focus_um"] * um

    # Grid
    grid = cfg["grid"]
    n_physical = grid["n_physical"]
    n_pad = grid["n_padding"]
    physical_width = grid["physical_width_um"] * um

    # Derived Grid Props
    n_total = n_physical + 2 * n_pad
    dx = physical_width / n_physical
    total_width = dx * n_total

    # Sample
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
        "dx": dx,
        "total_width": total_width,
        "sample_thickness": thickness,
        "sample_type": sample["type"],
        "sample_params": sample.get("params", {}),
        "ground_truth_cfg": sample.get("ground_truth", {}),
    }


def validate_sampling_conditions(sim_params: Dict[str, Any]):
    """
    Performs a safety check to ensure grid resolution (dx) is sufficient
    for the probe's divergence angle (Nyquist Limit).
    """
    wl = sim_params["wavelength"]
    dx = sim_params["dx"]
    dia = sim_params["probe_dia"]
    foc = sim_params["probe_focus"]

    # 1. Grid Nyquist Limit: arcsin(lambda / 2dx)
    try:
        max_angle_deg = np.degrees(np.arcsin(wl / (2.0 * dx)))
    except FloatingPointError:
        max_angle_deg = 90.0  # dx is extremely small

    # 2. Probe Geometric Divergence
    if abs(foc) < 1e-12:
        probe_angle_deg = 0.0
    else:
        # Theta = 2 * atan( (D/2) / f )
        probe_angle_deg = np.degrees(2.0 * np.arctan((dia / 2.0) / abs(foc)))

    logging.info("--- SAMPLING CHECK ---")
    logging.info(f"  Grid Nyquist Limit: {max_angle_deg:.4f}°")
    logging.info(f"  Probe Divergence:   {probe_angle_deg:.4f}°")

    if probe_angle_deg > max_angle_deg:
        msg = (
            f"CRITICAL WARNING: Probe divergence ({probe_angle_deg:.2f}°) "
            f"exceeds grid limit ({max_angle_deg:.2f}°). Aliasing will occur!"
        )
        logging.warning(msg)
        warnings.warn(msg)
