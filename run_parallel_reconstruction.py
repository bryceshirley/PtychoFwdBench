import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# Assuming these are available in your environment, otherwise we define the generator below
from ptycho_fwd_bench.generators import generate_blob_phantom
from ptycho_fwd_bench.solvers.batched_pint_multislice import (
    ParallelMultisliceSolverBatched,
)

# =============================================================================
# 0. Setup Timestamped Output Directory
# =============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"simulation_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(output_dir, "simulation.log")),
        logging.StreamHandler(),
    ],
)
logging.info(f"Starting simulation. Results will be saved to: {output_dir}")

# =============================================================================
# 1. Configuration
# =============================================================================
# Define padding size (pixels) to add to LEFT and RIGHT of the window
pad_size = 32

sim_params = {
    "global_nx": 1024,
    "nz": 100,
    "window_nx": 256,
    # The solver will use a wider window: window_nx + 2 * pad_size
    "window_nx_padded": 256 + 2 * pad_size,
    "pad_size": pad_size,
    "dx": 150.0 / 1024,  # ~146.5 nm
    "dz": 200.0 / 100,  # 2.0 um
    "wavelength": 0.635,
    "probe_dia": 5.0,
    "probe_focus": -10.0,
    "sample_params": {"n_blobs": 5, "delta_n": -0.001, "beta_n": 0.005},
}

# =============================================================================
# 2. Generate Phantom (Original Size)
# =============================================================================
# Generate the "Physical" unpadded map first for Ground Truth reference
n_map_raw = generate_blob_phantom(
    nx=sim_params["global_nx"],
    nz=sim_params["nz"],
    n_background=1.0,
    delta_n=sim_params["sample_params"]["delta_n"],
    beta_n=sim_params["sample_params"]["beta_n"],
    n_blobs=sim_params["sample_params"]["n_blobs"],
    seed=0,
)
n_map_true_raw = np.squeeze(n_map_raw)
if n_map_true_raw.ndim == 1:
    n_map_true_raw = np.tile(n_map_true_raw[:, np.newaxis], (1, sim_params["nz"]))

# Save Ground Truth (Unpadded view)
plt.figure(figsize=(10, 10))
plt.imshow(np.real(n_map_true_raw.T), cmap="inferno", aspect="auto")
plt.title("Real Part of True Refractive Index Map (Physical)")
plt.xlabel("Transverse Position (pixels)")
plt.ylabel("Depth (slices)")
plt.savefig(os.path.join(output_dir, "true_n_map.png"), bbox_inches="tight")
plt.close()

# =============================================================================
# 3. Apply Padding to Global Maps
# =============================================================================
# We pad the global map so the solver can pull valid data even at the edges
# of the physical window.
pad = sim_params["pad_size"]

# Pad Ground Truth with 1.0 (Free Space)
n_map_true_padded = np.pad(
    n_map_true_raw, ((pad, pad), (0, 0)), mode="constant", constant_values=1.0 + 0j
)

logging.info(f"Global Map Padded: {n_map_true_raw.shape} -> {n_map_true_padded.shape}")
logging.info(
    f"Solver Window Padded: {sim_params['window_nx']} -> {sim_params['window_nx_padded']}"
)

# =============================================================================
# 4. Define Scan Path and Solver
# =============================================================================
n_probes = 50
# Calculate max start index for the UNPADDED logic, to keep physical locations consistent
max_start = sim_params["global_nx"] - sim_params["window_nx"]
scan_indices = np.linspace(0, max_start, n_probes).astype(int)

# Initialize Solver with PADDED window size
solver = ParallelMultisliceSolverBatched(
    dx=sim_params["dx"],
    nx=sim_params["window_nx_padded"],
    nz_steps=sim_params["nz"],
    dz=sim_params["dz"],
    wavelength=sim_params["wavelength"],
    probe_dia=sim_params["probe_dia"],
)

# Initial Wavefront (Created at padded size)
psi_0_batch = np.tile(solver.initialize_wavefront(None), (n_probes, 1))

# --- GENERATE SYNTHETIC DATA ---
# Run forward pass on the PADDED map
u_true_vol_padded = solver.run_ptycho_batch(
    psi_0_batch, scan_indices, n_map_true_padded, mode="forward"
)

# Crop the result to simulate a detector with limited field of view
# We discard the 'pad' pixels from left and right
# Shape: [Batch, Padded_NX, NZ] -> [Batch, Window_NX] (taking last slice -1)
exit_wave_data = u_true_vol_padded[:, pad:-pad, -1]

# --- FIX START ---
# Visualization of Data: We must use the VOLUME variable, not the exit_data variable
idx = n_probes // 2
# Crop the padding out of the volume for visualization
wavefield_viz = u_true_vol_padded[idx, pad:-pad, :]

plt.figure(figsize=(10, 10))
plt.imshow(np.abs(wavefield_viz.T), cmap="inferno", aspect="auto")
plt.title("True Forward Wavefield (Cropped/Detector View)")
plt.xlabel("Transverse Position (pixels)")
plt.ylabel("Depth (slices)")
plt.savefig(os.path.join(output_dir, "true_wavefields.png"), bbox_inches="tight")
plt.close()
# =============================================================================
# 5. Reconstruction Loop
# =============================================================================
n_recon_iters = 50
learning_rate = 1e-4

# Initialize estimate (Padded size) with free space
n_map_est_padded = np.full_like(n_map_true_padded, 1.0 + 0j)

for epoch in range(n_recon_iters):
    # --- A. Forward Pass (Padded) ---
    u_est_vol_padded = solver.run_ptycho_batch(
        psi_0_batch, scan_indices, n_map_est_padded, mode="forward"
    )

    # --- B. Compute Residual (Cropped) ---
    # Crop the estimate to match the detector data
    u_est_exit_cropped = u_est_vol_padded[:, pad:-pad, -1]

    # Calculate residual on the valid physical window only
    residual_exit_cropped = u_est_exit_cropped - exit_wave_data

    # --- C. Pad Residual for Adjoint ---
    # We must feed the solver a padded residual (zeros in the padded region)
    # because the solver expects 'window_nx_padded' width.
    residual_exit_padded = np.pad(
        residual_exit_cropped,
        ((0, 0), (pad, pad)),  # Pad 2nd dim (nx)
        mode="constant",
        constant_values=0j,
    )

    # --- D. Back-propagate Residual (Padded) ---
    v_adj_vol_padded = solver.run_ptycho_batch(
        residual_exit_padded, scan_indices, n_map_est_padded, mode="adjoint"
    )

    # --- E. Compute Gradient (Padded) ---
    # This returns the gradient on the PADDED global grid
    overlap_sum_padded = solver.compute_gradient(
        u_est_vol_padded, v_adj_vol_padded, scan_indices
    )

    # --- F. Separate Gradients ---
    grad_n_real_padded = np.imag(overlap_sum_padded)
    grad_n_imag_padded = np.real(overlap_sum_padded)

    # --- G. Update Step ---
    # Updates happen on the full padded grid, but since residual was 0 in pads,
    # gradient should be effectively 0 there (ignoring minor numerical bleed)
    # decay learning rate
    learning_rate *= 0.99
    n_map_est_padded.real -= learning_rate * grad_n_real_padded
    n_map_est_padded.imag -= (learning_rate * 0.1) * grad_n_imag_padded

    # --- H. Constraint Projection ---
    n_map_est_padded.imag = np.maximum(n_map_est_padded.imag, 0.0)

    # --- I. Monitoring ---
    mse = np.mean(np.abs(residual_exit_cropped) ** 2)

    if epoch % 5 == 0:
        logging.info(f"Epoch {epoch}: MSE = {mse:.2e}")

        # Crop maps back to original size for clean visualization
        n_map_est_viz = n_map_est_padded[pad:-pad, :]
        grad_viz = overlap_sum_padded[pad:-pad, :]

        # 1. Refractive Index Map
        plt.figure(figsize=(10, 10))
        im = plt.imshow(np.abs(n_map_est_viz.T), cmap="inferno", aspect="auto")
        plt.colorbar(im, label="Refractive Index Magnitude")
        plt.title(f"Refractive Index Magnitude (Physical Region) - Epoch {epoch}")
        plt.xlabel("Transverse Position")
        plt.ylabel("Depth")
        plt.savefig(
            os.path.join(output_dir, f"n_map_est_epoch_{epoch}.png"),
            bbox_inches="tight",
        )
        plt.close()

        # 2. Gradient Magnitude
        plt.figure(figsize=(10, 10))
        im = plt.imshow(np.abs(grad_viz.T), cmap="inferno", aspect="auto")
        plt.colorbar(im, label="Gradient Magnitude")
        plt.title(f"Gradient Magnitude (Physical Region) - Epoch {epoch}")
        plt.savefig(
            os.path.join(output_dir, f"gradient_epoch_{epoch}.png"), bbox_inches="tight"
        )
        plt.close()

        # 3. Wavefields (Forward/Adjoint) - Just show one batch item
        idx = n_probes // 2
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Show unpadded wavefield for clarity
        ax[0].imshow(
            np.abs(u_est_vol_padded[idx, pad:-pad, :].T), cmap="inferno", aspect="auto"
        )
        ax[0].set_title(f"Forward (Cropped) Ep {epoch}")

        ax[1].imshow(
            np.abs(v_adj_vol_padded[idx, pad:-pad, :].T), cmap="inferno", aspect="auto"
        )
        ax[1].set_title(f"Adjoint (Cropped) Ep {epoch}")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"wavefields_epoch_{epoch}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)

logging.info("Simulation Complete.")
