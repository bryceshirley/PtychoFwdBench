import logging
import os

import numpy as np

import ptycho_fwd_bench.dim_2.recon.plotters as plotters
from ptycho_fwd_bench.dim_2.benchmarking import (
    compute_ground_truth,
    generate_simulation_inputs,
)
from ptycho_fwd_bench.dim_2.generators.generators import interpolate_to_coarse
from ptycho_fwd_bench.dim_2.solvers import ParallelMultisliceSolverBatched


def generate_synthetic_dataset(config, sim_params, output_dir):
    grid_cfg = config["grid"]
    phys_cfg = config["probe"]
    fwd_cfg = config["forward_model"]
    recon_cfg = config["reconstruction"]
    pad = grid_cfg["window_padding_nx"]
    n_probes = recon_cfg["n_probes"]

    # 1. Generate Fine Physics Inputs
    n_map_fine, psi_0_full_fine = generate_simulation_inputs(sim_params)
    compute_ground_truth(n_map_fine, psi_0_full_fine, sim_params, output_dir)

    # 2. Interpolate Object to Coarse Mesh for Reconstruction True Ref
    coarse_nz = fwd_cfg["coarse_nz"]
    logging.info(f"Interpolating object to Coarse Mesh (Z={coarse_nz})...")
    n_map_true_raw = interpolate_to_coarse(n_map_fine, coarse_nz)

    plotters.plot_true_object(
        n_map_true_raw,
        "True Object (Coarse)",
        os.path.join(output_dir, "object_true_coarse.png"),
    )

    # Pad Object
    n_map_fine_padded = np.pad(
        n_map_fine, ((pad, pad), (0, 0)), mode="constant", constant_values=1.0 + 0j
    )
    n_map_true_padded = np.pad(
        n_map_true_raw, ((pad, pad), (0, 0)), mode="constant", constant_values=1.0 + 0j
    )

    # 3. Setup Fine Solver for Data Generation
    fine_nz = config["ground_truth"]["fine_nz"]
    fine_solver = ParallelMultisliceSolverBatched(
        dx=grid_cfg["dx"],
        nx=grid_cfg["window_nx_padded"],
        nz_steps=fine_nz,
        dz=sim_params["sample_thickness"] / fine_nz,
        wavelength=phys_cfg["wavelength"],
        probe_dia=phys_cfg["diameter"],
        probe_focus=phys_cfg["focus"],
        alpha=fwd_cfg["params"]["alpha"],
        n_iter=fwd_cfg["params"]["n_iter"],
        solver_type=fwd_cfg["params"]["solver_type"],
    )

    # 4. Generate Diffraction Data
    logging.info("Generating synthetic Ptychography data on FINE mesh...")
    max_start = grid_cfg["global_nx"] - grid_cfg["window_nx"]
    scan_indices = np.linspace(0, max_start, n_probes).astype(int)

    psi_0_true_single = fine_solver.initialize_wavefront(None)
    psi_0_true_batch = np.tile(psi_0_true_single, (n_probes, 1))

    u_true_vol_padded = fine_solver.run_ptycho_batch(
        psi_0_true_batch, scan_indices, n_map_fine_padded, mode="forward"
    )

    # Crop Data
    exit_wave_data = u_true_vol_padded[:, pad:-pad, -1]

    # Generate Measured Constraints (Far Field)
    diffraction_intensities = np.abs(np.fft.fft(exit_wave_data, axis=-1)) ** 2
    measured_amplitudes = np.sqrt(diffraction_intensities)

    # Visualize Data Generation
    plotters.plot_exit_wave(
        u_true_vol_padded[n_probes // 2, pad:-pad, :],
        os.path.join(output_dir, "exit_wave_true_fine.png"),
    )
    plotters.plot_data_intensity(
        exit_wave_data[n_probes // 2],
        measured_amplitudes[n_probes // 2],
        os.path.join(output_dir, "data_intensity_profile.png"),
    )

    return {
        "n_map_true_padded": n_map_true_padded,
        "psi_0_true_single": psi_0_true_single,
        "exit_wave_data": exit_wave_data,
        "measured_amplitudes": measured_amplitudes,
        "scan_indices": scan_indices,
    }
