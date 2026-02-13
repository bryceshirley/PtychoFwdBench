import logging
import os

import numpy as np

import ptycho_fwd_bench.dim_2.recon.plotters as plotters
from ptycho_fwd_bench.dim_2.recon.config_parser import build_sim_params, load_config
from ptycho_fwd_bench.dim_2.recon.data_generator import generate_synthetic_dataset
from ptycho_fwd_bench.dim_2.recon.initialization import (
    initialize_object,
    initialize_probe,
)

# --- Modular Imports ---
from ptycho_fwd_bench.dim_2.solvers import ParallelMultisliceSolverBatched

# =============================================================================
# 0. Setup & Parse Config
# =============================================================================
config, output_dir = load_config()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(output_dir, "simulation.log")),
        logging.StreamHandler(),
    ],
)
logging.info(f"Experiment: {config['experiment']['name']}")

# Hyperparameters
n_probes = config["reconstruction"]["n_probes"]
n_recon_epochs = config["reconstruction"]["n_epochs"]
pad = config["grid"]["window_padding_nx"]

lr_obj = float(config["reconstruction"]["optimizer"]["lr_object"])
lr_probe = float(config["reconstruction"]["optimizer"]["lr_probe"])
lr_decay = float(config["reconstruction"]["optimizer"]["lr_decay"])

RECON_MODE = config["reconstruction"]["mode"]

# =============================================================================
# 1. Generate Ground Truth Data
# =============================================================================
sim_params = build_sim_params(config)
gt_data = generate_synthetic_dataset(config, sim_params, output_dir)

scan_indices = gt_data["scan_indices"]
measured_amplitudes = gt_data["measured_amplitudes"]
exit_wave_data = gt_data["exit_wave_data"]
n_map_true_padded = gt_data["n_map_true_padded"]
psi_0_true_single = gt_data["psi_0_true_single"]

# =============================================================================
# 2. Setup Reconstruction Solver & Initialization
# =============================================================================
# Initialize the Coarse (Reconstruction) Solver
solver = ParallelMultisliceSolverBatched(
    dx=config["grid"]["dx"],
    nx=config["grid"]["window_nx_padded"],
    nz_steps=config["forward_model"]["coarse_nz"],
    dz=config["grid"]["dz"],
    wavelength=config["probe"]["wavelength_um"] * 1e-6,
    probe_dia=config["probe"]["diameter_um"] * 1e-6,
    probe_focus=config["probe"]["focus_um"] * 1e-6,
    alpha=config["forward_model"]["params"]["alpha"],
    n_iter=config["forward_model"]["params"]["n_iter"],
    solver_type=config["forward_model"]["params"]["solver_type"],
)

# Initialize States
n_map_est = n_map_true_padded.copy()
psi_0_est = psi_0_true_single.copy()
probe_history = []

if "object" in RECON_MODE:
    n_map_est = initialize_object(
        config["reconstruction"]["initialization"]["object_mode"],
        n_map_est.shape,
        config["reconstruction"]["initialization"].get("seed", 0),
    )

if "probe" in RECON_MODE:
    psi_0_est = initialize_probe(
        config["reconstruction"]["initialization"]["probe_mode"],
        config["grid"]["window_nx_padded"],
        config["reconstruction"]["initialization"].get("seed", 0),
    )

probe_history.append(psi_0_est.copy())
psi_0_est_batch = np.tile(psi_0_est, (n_probes, 1))

# =============================================================================
# 3. Core Reconstruction Loop
# =============================================================================
logging.info("Starting iterative reconstruction...")

for epoch in range(n_recon_epochs):
    # 1. Forward Pass
    u_est_vol = solver.run_ptycho_batch(
        psi_0_est_batch, scan_indices, n_map_est, mode="forward"
    )
    u_est_exit_cropped = u_est_vol[:, pad:-pad, -1]

    # 2. Apply Data Constraints / Compute Residual
    if "phase_retrieval" in RECON_MODE:
        # Gerchberg-Saxton Modulus Projection
        G_est = np.fft.fft(u_est_exit_cropped, axis=-1)
        G_constrained = measured_amplitudes * np.exp(1j * np.angle(G_est))
        u_target = np.fft.ifft(G_constrained, axis=-1)

        residual_cropped = u_est_exit_cropped - u_target
        if epoch % 5 == 0:
            gs_error = np.mean(np.abs(measured_amplitudes - np.abs(G_est)) ** 2)
            logging.info(f"Epoch {epoch}: Fourier Modulus MSE = {gs_error:.2e}")
    else:
        # Standard complex residual
        residual_cropped = u_est_exit_cropped - exit_wave_data

    residual_padded = np.pad(
        residual_cropped, ((0, 0), (pad, pad)), mode="constant", constant_values=0j
    )

    # 3. Adjoint Pass (Backpropagation)
    v_adj_vol = solver.run_ptycho_batch(
        residual_padded, scan_indices, n_map_est, mode="adjoint"
    )

    # 4. Gradients & Updates
    if "object" in RECON_MODE:
        grad_n_obj = solver.compute_gradient_object(u_est_vol, v_adj_vol, scan_indices)
        max_grad = np.max(np.abs(grad_n_obj)) + 1e-12
        normalized_grad = grad_n_obj / max_grad

        lr_obj *= lr_decay
        n_map_est -= lr_obj * (-1j * normalized_grad)

    if "probe" in RECON_MODE:
        grad_probe = solver.compute_gradient_probe(v_adj_vol)
        lr_probe *= lr_decay
        psi_0_est -= lr_probe * grad_probe

        psi_0_est_batch = np.tile(psi_0_est, (n_probes, 1))
        probe_history.append(psi_0_est.copy())

    # 5. Monitoring & Plotting
    mse = np.mean(np.abs(residual_cropped) ** 2)
    if epoch % 5 == 0:
        logging.info(f"Epoch {epoch}: Near-Field Residual MSE = {mse:.2e}")
        if "probe" in RECON_MODE:
            plotters.plot_epoch_probe(
                psi_0_true_single,
                psi_0_est,
                epoch,
                os.path.join(output_dir, f"probe_est_epoch_{epoch}.png"),
            )
        if "object" in RECON_MODE:
            plotters.plot_epoch_object(
                n_map_est,
                epoch,
                os.path.join(output_dir, f"object_est_epoch_{epoch}.png"),
            )

# =============================================================================
# 4. Final Outputs
# =============================================================================
if "probe" in RECON_MODE:
    plotters.plot_final_probe(
        psi_0_true_single,
        probe_history,
        mse,
        os.path.join(output_dir, "probe_final_comparison.png"),
        os.path.join(output_dir, "probe_convergence.png"),
    )

if "object" in RECON_MODE:
    plotters.plot_final_object(
        n_map_true_padded[pad:-pad, :],
        n_map_est[pad:-pad, :],
        os.path.join(output_dir, "object_final_comparison.png"),
    )

logging.info(f"Simulation Complete. Results in {output_dir}")
