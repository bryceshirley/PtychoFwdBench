import logging
import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import yaml

from ptycho_fwd_bench.dim_2.benchmarking import generate_blob_phantom
from ptycho_fwd_bench.dim_2.solvers import (
    ParallelMultisliceSolverBatched,
)

SOLVER_MAP = {
    "pint": ParallelMultisliceSolverBatched,
}


def load_config(path="recon_config.yml"):
    # 1. Parse the YAML just to read the experiment name
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Create the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/recon/{cfg['experiment']['name']}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Copy the exact original file to the new directory
    dest_path = os.path.join(output_dir, "config_run.yml")
    shutil.copy2(path, dest_path)

    # Calculate derived parameters
    cfg["grid"]["dx"] = cfg["grid"]["physical_width"] / cfg["grid"]["global_nx"]
    cfg["grid"]["dz"] = cfg["grid"]["physical_thickness"] / cfg["grid"]["nz"]
    cfg["grid"]["window_nx_padded"] = (
        cfg["grid"]["window_nx"] + 2 * cfg["grid"]["pad_size"]
    )

    return cfg, output_dir


# =============================================================================
# 0. Setup & Initialization
# =============================================================================
config, output_dir = load_config()
exp_cfg = config["experiment"]
grid_cfg = config["grid"]
phys_cfg = config["physics"]
opt_cfg = config["optimizer"]
solver_cfg = config["solver"]

n_probes = exp_cfg["n_probes"]
n_recon_iters = exp_cfg["n_recon_iters"]

lr_object = float(opt_cfg["learning_rate_object"])
lr_probe = float(opt_cfg["learning_rate_probe"])
lr_decay = float(opt_cfg["lr_decay"])

RECON_MODE = exp_cfg["recon_mode"]
PROBE_INIT_MODE = exp_cfg["probe_init_mode"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(output_dir, "simulation.log")),
        logging.StreamHandler(),
    ],
)

logging.info(f"Mode: {RECON_MODE}")
if "probe" in RECON_MODE:
    logging.info(f"Probe Initialization: {PROBE_INIT_MODE}")

# =============================================================================
# 1. Generate Phantom & Solver
# =============================================================================
n_map_raw = generate_blob_phantom(
    nx=grid_cfg["global_nx"],
    nz=grid_cfg["nz"],
    n_background=1.0,
    delta_n=config["sample"]["delta_n"],
    beta_n=config["sample"]["beta_n"],
    n_blobs=config["sample"]["n_blobs"],
    seed=exp_cfg["seed"],
)
n_map_true_raw = np.squeeze(n_map_raw)

# Plot the true object for visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.angle(n_map_true_raw.T), cmap="inferno", aspect="auto")
plt.title("True Object Phase")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(np.abs(n_map_true_raw.T), cmap="inferno", aspect="auto")
plt.title("True Object Modulus")
plt.colorbar()
plt.savefig(os.path.join(output_dir, "object_true.png"))
plt.close()

# Pad Object
pad = grid_cfg["pad_size"]
n_map_true_padded = np.pad(
    n_map_true_raw, ((pad, pad), (0, 0)), mode="constant", constant_values=1.0 + 0j
)


solver = SOLVER_MAP.get(solver_cfg["name"], ParallelMultisliceSolverBatched)(
    dx=grid_cfg["dx"],
    nx=grid_cfg["window_nx_padded"],
    nz_steps=grid_cfg["nz"],
    dz=grid_cfg["dz"],
    wavelength=phys_cfg["wavelength"],
    probe_dia=phys_cfg["probe_dia"],
    probe_focus=phys_cfg["probe_focus"],
    alpha=solver_cfg["params"]["alpha"],
    n_iter=solver_cfg["params"]["n_iter"],
    solver_type=solver_cfg["params"]["solver_type"],
    z_cutoff=solver_cfg["params"].get("z_cutoff", 1.0),
)


# C. Generate Probe Ground Truth
# We generate the "True" probe using the solver's internal physics model
psi_0_true_single = solver.initialize_wavefront(None)
psi_0_true_batch = np.tile(psi_0_true_single, (n_probes, 1))

# =============================================================================
# 4. Generate Synthetic Data
# =============================================================================
logging.info("Generating synthetic ground truth data...")
max_start = grid_cfg["global_nx"] - grid_cfg["window_nx"]
scan_indices = np.linspace(0, max_start, n_probes).astype(int)


# Run forward pass with TRUE Probe and TRUE Object
u_true_vol_padded = solver.run_ptycho_batch(
    psi_0_true_batch, scan_indices, n_map_true_padded, mode="forward"
)

# Crop to detector view
exit_wave_data = u_true_vol_padded[:, pad:-pad, -1]

# Save one example of the full wavefield for visualization phase and modulus
plt.subplots(1, 2, figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(
    np.angle(u_true_vol_padded[n_probes // 2, pad:-pad, :].T),
    cmap="inferno",
    aspect="auto",
)
plt.title("Exit Wave Phase (True)")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(
    np.abs(u_true_vol_padded[n_probes // 2, pad:-pad, :].T),
    cmap="inferno",
    aspect="auto",
)
plt.title("Exit Wave Amplitude (True)")
plt.colorbar()
plt.savefig(os.path.join(output_dir, "exit_wave_true.png"))
plt.close()


# --- For Phase Retrieval: Generate Far-Field Intensities ---
# FFT to get Far-Field
exit_wave_fourier_true = np.fft.fft(exit_wave_data, axis=-1)
# Calculate Intensity (Modulus Squared)
diffraction_intensities = np.abs(exit_wave_fourier_true) ** 2
# The "measured" amplitude constraint for GS
measured_amplitudes = np.sqrt(diffraction_intensities)

# Save Data visualization
plt.figure(figsize=(10, 4))
plt.plot(np.abs(exit_wave_data[n_probes // 2]), label="Magnitude (Near Field)")
plt.plot(
    np.fft.fftshift(measured_amplitudes[n_probes // 2]) / 100,
    label="Diffraction Amp (Scaled)",
)
plt.title("Exit Wave Data (Central Probe)")
plt.legend()
plt.savefig(os.path.join(output_dir, "data_intensity_profile.png"))
plt.close()


# =============================================================================
# 5. Initialization logic based on Mode
# =============================================================================
n_map_est_padded = n_map_true_padded.copy()
psi_0_est_batch = psi_0_true_batch.copy()
psi_0_est_single = psi_0_true_single.copy()
probe_history = [psi_0_est_single.copy()]

# Variables specific to Phase Retrieval
gs_exit_wave_est = None
gs_error_history = []

if "object" in RECON_MODE:
    # Initialize Object to Vacuum (1.0)
    n_map_est_padded = np.full_like(n_map_true_padded, 1.0 + 0j)

if "probe" in RECON_MODE:
    nx_p = grid_cfg["window_nx_padded"]
    if PROBE_INIT_MODE == "ones":
        psi_0_est_single = np.ones(nx_p, dtype=np.complex128)
    elif PROBE_INIT_MODE == "zeros":
        psi_0_est_single = np.zeros(nx_p, dtype=np.complex128)
    elif PROBE_INIT_MODE == "gaussian":
        x = np.arange(nx_p)
        center = nx_p // 2
        width = nx_p // 6
        amp_profile = 1.0 * np.exp(-((x - center) ** 2) / (2 * width**2))
        psi_0_est_single = amp_profile.astype(np.complex128)
    elif PROBE_INIT_MODE == "random":
        rng = np.random.default_rng(exp_cfg["seed"])
        psi_0_est_single = (rng.random(nx_p) + 1j * rng.random(nx_p)) * 0.1 + 0.5
    elif PROBE_INIT_MODE == "cauchy":
        x = np.arange(nx_p)
        center = nx_p // 2
        width = nx_p // 10
        amp_profile = 1.0 / (1.0 + ((x - center) / width) ** 2)
        psi_0_est_single = amp_profile.astype(np.complex128)
    else:
        raise ValueError(f"Unknown probe init: {PROBE_INIT_MODE}")

    psi_0_est_batch = np.tile(psi_0_est_single, (n_probes, 1))
    probe_history = [psi_0_est_single.copy()]

# =============================================================================
# 6. Reconstruction Loop
# =============================================================================

logging.info("Starting reconstruction...")

for epoch in range(n_recon_iters):
    if np.any(n_map_est_padded < 1e-9):
        logging.warning("Found zeros/negatives in n_map! Replacing with 1.0 (Vacuum).")
        n_map_est_padded = np.where(n_map_est_padded < 1e-9, 1.0, n_map_est_padded)

    # --- A. Forward Pass ---
    u_est_vol_padded = solver.run_ptycho_batch(
        psi_0_est_batch, scan_indices, n_map_est_padded, mode="forward"
    )

    # --- B. Compute Residual ---
    u_est_exit_cropped = u_est_vol_padded[:, pad:-pad, -1]
    if "phase_retrieval" in RECON_MODE:
        # --- PHASE RETRIEVAL LOGIC (GS / Error Reduction) ---

        # 1. Propagate to Far-Field (Fourier Domain)
        G_est = np.fft.fft(u_est_exit_cropped, axis=-1)

        # 2. Apply Modulus Constraint
        # Keep the Phase from the Estimate, but force Amplitude to match Data
        G_phase = np.exp(1j * np.angle(G_est))
        G_constrained = measured_amplitudes * G_phase

        # 3. Propagate back to Near-Field (Inverse Fourier)
        u_target = np.fft.ifft(G_constrained, axis=-1)

        # 4. Compute Residual
        # The "error" driving the update is the difference between
        # what we estimated vs. what the physics (data) says it should be.
        residual_exit_cropped = u_est_exit_cropped - u_target

        # Optional: Track Error
        gs_error = np.mean(np.abs(measured_amplitudes - np.abs(G_est)) ** 2)
        if epoch % 5 == 0:
            logging.info(f"Epoch {epoch}: Fourier MSE = {gs_error:.2e}")

    else:
        # --- STANDARD RECONSTRUCTION (Known Complex Truth) ---
        # This is for debugging/benchmarking where we know the answer
        residual_exit_cropped = u_est_exit_cropped - exit_wave_data

    # Pad residual for solver
    residual_exit_padded = np.pad(
        residual_exit_cropped, ((0, 0), (pad, pad)), mode="constant", constant_values=0j
    )

    # --- C. Back-propagation (Adjoint) ---
    v_adj_vol_padded = solver.run_ptycho_batch(
        residual_exit_padded, scan_indices, n_map_est_padded, mode="adjoint"
    )

    # --- D. Gradients & Updates ---
    if "object" in RECON_MODE:
        overlap_sum_padded = solver.compute_gradient_object(
            u_est_vol_padded, v_adj_vol_padded, scan_indices
        )

        # Save for viz
        grad_viz_padded = overlap_sum_padded

        # --- Normalization (Object) ---
        # Normalize by the max magnitude so lr_object is the max step size
        normalize = np.max(np.abs(u_est_vol_padded)) + 1e-12

        # Decay LR
        lr_object *= lr_decay
        grad_n = -1j * overlap_sum_padded

        # Update Object
        n_map_est_padded -= lr_object * grad_n

    if "probe" in RECON_MODE:
        # 1. Compute Probe Gradient
        grad_probe_batch = solver.compute_gradient_probe(v_adj_vol_padded)

        # 2. Update Probe
        lr_probe *= lr_decay
        psi_0_est_single -= lr_probe * grad_probe_batch

        # Broadcast back to batch
        psi_0_est_batch = np.tile(psi_0_est_single, (n_probes, 1))

        probe_history.append(psi_0_est_single.copy())

    # --- E. Monitoring ---
    mse = np.mean(np.abs(residual_exit_cropped) ** 2)

    if epoch % 5 == 0:
        logging.info(f"Epoch {epoch}: MSE = {mse:.2e}")

        if "probe" in RECON_MODE:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].plot(
                np.abs(psi_0_true_single), "k--", label="Ground Truth", alpha=0.6
            )
            ax[0].plot(np.abs(psi_0_est_single), "r-", label=f"Epoch {epoch}")
            ax[0].set_title("Probe Amplitude")
            ax[0].legend()

            ax[1].plot(
                np.angle(psi_0_true_single), "k--", label="Ground Truth", alpha=0.6
            )
            ax[1].plot(np.angle(psi_0_est_single), "r-", label=f"Epoch {epoch}")
            ax[1].set_title("Probe Phase")

            plt.suptitle(f"Probe Reconstruction - Epoch {epoch}")
            plt.savefig(os.path.join(output_dir, f"probe_est_epoch_{epoch}.png"))
            plt.close()
        if "object" in RECON_MODE:
            # Plot Object 2D
            # Crop padding for visualization
            n_viz = n_map_est_padded[pad:-pad, :]
            grad_viz = grad_viz_padded[pad:-pad, :]

            fig, axs = plt.subplots(2, 2, figsize=(18, 5))

            # 1. Modulus (Refractive Index)
            im1 = axs[0, 0].imshow(np.abs(n_viz.T), cmap="inferno", aspect="auto")
            axs[0, 0].set_title(f"Modulus(n) - Epoch {epoch}")
            plt.colorbar(im1, ax=axs[0, 0])

            # 2. Phase (Absorption)
            im2 = axs[0, 1].imshow(np.angle(n_viz.T), cmap="inferno", aspect="auto")
            axs[0, 1].set_title(f"Phase(n) - Epoch {epoch}")
            plt.colorbar(im2, ax=axs[0, 1])

            # 3. Gradient Magnitude
            im3 = axs[1, 0].imshow(np.abs(grad_viz.T), cmap="viridis", aspect="auto")
            axs[1, 0].set_title(f"Gradient Mag - Epoch {epoch}")
            plt.colorbar(im3, ax=axs[1, 0])

            # 4. Gradient Phase
            im4 = axs[1, 1].imshow(np.angle(grad_viz.T), cmap="viridis", aspect="auto")
            axs[1, 1].set_title(f"Gradient Phase - Epoch {epoch}")
            plt.colorbar(im4, ax=axs[1, 1])

            plt.savefig(
                os.path.join(output_dir, f"object_est_epoch_{epoch}.png"),
                bbox_inches="tight",
            )
            plt.close()

# =============================================================================
# 7. Final Output
# =============================================================================

if "probe" in RECON_MODE:
    logging.info("Generating Final Probe Comparison Plot...")

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(np.abs(psi_0_true_single), "k", linewidth=2, label="Ground Truth")
    axs[0].plot(np.abs(probe_history[0]), "g--", linewidth=1.5, label="Initial Guess")
    axs[0].plot(
        np.abs(probe_history[-1]), "r-.", linewidth=1.5, label="Final Reconstruction"
    )
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title(f"Probe Reconstruction Results (MSE: {mse:.2e})")
    axs[0].legend()

    axs[1].plot(np.angle(psi_0_true_single), "k", linewidth=2, label="Ground Truth")
    axs[1].plot(np.angle(probe_history[0]), "g--", linewidth=1.5, label="Initial Guess")
    axs[1].plot(
        np.angle(probe_history[-1]), "r-.", linewidth=1.5, label="Final Reconstruction"
    )
    axs[1].set_ylabel("Phase (rad)")
    axs[1].set_xlabel("Pixel Index")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "probe_final_comparison.png"))
    plt.close()

    err_history = [np.mean(np.abs(p - psi_0_true_single) ** 2) for p in probe_history]
    plt.figure()
    plt.plot(err_history)
    plt.xlabel("Epoch")
    plt.ylabel("Probe Error (MSE vs GT)")
    plt.title("Probe Convergence")
    plt.savefig(os.path.join(output_dir, "probe_convergence.png"))
    plt.close()

if "object" in RECON_MODE:
    logging.info("Generating Final Object Comparison...")

    # Compare Ground Truth vs Reconstruction
    n_true_crop = n_map_true_padded[pad:-pad, :]
    n_est_crop = n_map_est_padded[pad:-pad, :]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Real Part
    axs[0, 0].imshow(np.angle(n_true_crop.T), cmap="inferno", aspect="auto")
    axs[0, 0].set_title("Ground Truth (Phase)")

    axs[0, 1].imshow(np.angle(n_est_crop.T), cmap="inferno", aspect="auto")
    axs[0, 1].set_title("Reconstruction (Phase)")

    # Imaginary Part
    axs[1, 0].imshow(np.abs(n_true_crop.T), cmap="inferno", aspect="auto")
    axs[1, 0].set_title("Ground Truth (Modulus)")

    axs[1, 1].imshow(np.abs(n_est_crop.T), cmap="inferno", aspect="auto")
    axs[1, 1].set_title("Reconstruction (Modulus)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "object_final_comparison.png"))
    plt.close()

logging.info(f"Simulation Complete. Results in {output_dir}")
