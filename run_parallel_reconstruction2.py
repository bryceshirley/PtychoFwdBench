import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# GPU Handling
try:
    import cupy as cp

    HAS_GPU = True
    logging.info("GPU Detected: Using CuPy")
except ImportError:
    import numpy as cp

    HAS_GPU = False
    logging.info("No GPU Detected: Using NumPy")
from ptycho_fwd_bench.generators import generate_blob_phantom
from ptycho_fwd_bench.solvers import ParallelMultisliceSolverBatched

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"joint_recon_{timestamp}"
iter_dir = os.path.join(results_dir, "iterations")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(iter_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(os.path.join(results_dir, "run.log")))

params = {
    "global_nx": 1024,
    "nz": 50,
    "window_nx": 256,
    "pad_size": 32,
    "dx": 1.0,  # nm
    "dz": 20.0,  # nm
    "wavelength": 1.5,
    "probe_dia": 80.0,
    "probe_focus": -200.0,
    "n_blobs": 12,
    "delta_n": -0.005,
    "beta_n": 0.0001,
}
params["window_padded"] = params["window_nx"] + 2 * params["pad_size"]

# =============================================================================
# 3. GENERATORS & PLOTTING
# =============================================================================


def get_structured_probe(nx, dx, wavelength, dia, focus):
    x = (np.arange(nx) - nx // 2) * dx
    k = 2 * np.pi / wavelength
    radius = dia / 2.0
    aperture = (np.abs(x) < radius).astype(np.complex128)
    phase = (k * x**2) / (2 * focus)  # Defocus
    phase += 1e-7 * (x / radius) ** 4  # Spherical
    phase += 0.2 * (x / radius) ** 3  # Coma
    return aperture * np.exp(1j * phase)


def generate_ground_truth():
    logging.info("Generating Phantom Object...")
    n_map_raw = generate_blob_phantom(
        nx=params["global_nx"],
        nz=params["nz"],
        n_background=1.0,
        delta_n=params["delta_n"],
        beta_n=params["beta_n"],
        n_blobs=params["n_blobs"],
        seed=42,
    )
    n_map_raw = np.squeeze(n_map_raw)
    pad = params["pad_size"]
    n_map_padded = np.pad(
        n_map_raw, ((pad, pad), (0, 0)), mode="constant", constant_values=1.0
    )

    logging.info("Generating Probe...")
    probe_gt = get_structured_probe(
        params["window_padded"],
        params["dx"],
        params["wavelength"],
        params["probe_dia"],
        params["probe_focus"],
    )

    if HAS_GPU:
        n_map_padded = cp.asarray(n_map_padded)
        probe_gt = cp.asarray(probe_gt)
    return n_map_padded, probe_gt


def save_plot(data, filename, title, xlabel="x", ylabel="y", mode="real"):
    """Helper to save clean plots."""
    data_cpu = cp.asnumpy(data) if HAS_GPU else data
    plt.figure(figsize=(8, 5))

    if mode == "real":
        im = plt.imshow(np.real(data_cpu).T, aspect="auto", cmap="inferno")
        plt.colorbar(im)
    elif mode == "abs":
        im = plt.imshow(np.abs(data_cpu), aspect="auto", cmap="viridis")
        plt.colorbar(im)
    elif mode == "mag":
        im = plt.imshow(np.abs(data_cpu).T, aspect="auto", cmap="hot")
        plt.colorbar(im)
    elif mode == "1d_complex":
        plt.plot(np.abs(data_cpu), label="Amp", color="blue")
        plt.ylabel("Amplitude", color="blue")
        ax2 = plt.gca().twinx()
        ax2.plot(np.angle(data_cpu), label="Phase", color="orange", alpha=0.5)
        ax2.set_ylabel("Phase", color="orange")

    plt.title(title)
    plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=150)
    plt.close()


# =============================================================================
# 4. RECONSTRUCTION ENGINE
# =============================================================================


class JointReconstructor:
    def __init__(self, solver, scan_indices, measurements):
        self.solver = solver
        self.indices = scan_indices
        self.meas_amp = measurements
        self.pad = params["pad_size"]
        self.n_pos = len(scan_indices)

    def run(self, n_epochs=100, batch_size=10, use_gs=True):
        logging.info(f"Starting Reconstruction (GS={use_gs})...")

        # Init Estimates
        n_map_est = cp.ones_like(n_map_gt) + (0.0 + 0j)
        x = (
            cp.arange(params["window_padded"]) - params["window_padded"] // 2
        ) * params["dx"]
        if HAS_GPU:
            x = cp.asarray(x)
        # Blind Probe Guess (Simple Gaussian)
        probe_est = cp.exp(-(x**2) / (2 * (params["probe_dia"] / 4) ** 2)).astype(
            cp.complex128
        )

        # Step Sizes (Normalized)
        # Higher alpha because we normalize by Intensity
        alpha_obj = 0.5
        alpha_probe = 0.9

        loss_history = []
        perm = np.arange(self.n_pos)

        for epoch in range(n_epochs):
            np.random.shuffle(perm)
            epoch_loss = 0

            # Placeholders for visualization
            last_grad_O = None
            last_grad_P = None

            for i in range(0, self.n_pos, batch_size):
                batch_idx = perm[i : i + batch_size]
                curr_B = len(batch_idx)
                idx_ten = self.indices[batch_idx]
                target_amp = self.meas_amp[batch_idx]

                # 1. Forward
                probe_batch = cp.tile(probe_est, (curr_B, 1))
                vol_est = self.solver.run_ptycho_batch(
                    probe_batch, idx_ten, n_map_est, mode="forward", n_iter=1
                )

                # 2. Residual
                exit_est = vol_est[:, self.pad : -self.pad, -1]
                model_amp = cp.abs(exit_est)
                epoch_loss += cp.sum((model_amp - target_amp) ** 2)

                if use_gs:
                    # GS Projection: Force Amplitude
                    projected = (target_amp / (model_amp + 1e-9)) * exit_est
                    residual_cropped = projected - exit_est
                else:
                    # Gradient Descent on L2
                    term = 1.0 - target_amp / (model_amp + 1e-9)
                    residual_cropped = -1.0 * term * exit_est

                # 3. Adjoint
                res_padded = cp.pad(
                    residual_cropped, ((0, 0), (self.pad, self.pad)), mode="constant"
                )
                vol_adj = self.solver.run_ptycho_batch(
                    res_padded, idx_ten, n_map_est, mode="adjoint", n_iter=1
                )

                # 4. Gradients
                grad_O = self.solver.compute_gradient_object(vol_est, vol_adj, idx_ten)
                grad_P = self.solver.compute_gradient_probe(vol_adj)

                last_grad_O = grad_O
                last_grad_P = grad_P

                # 5. Normalization (CRITICAL FIX)
                # Norm Object by Probe Intensity (Max approx)
                norm_O = cp.max(cp.abs(probe_est) ** 2) + 1e-9
                # Norm Probe by Object Intensity (approx 1.0 * BatchSize)
                norm_P = cp.max(cp.abs(n_map_est) ** 2) * curr_B + 1e-9

                # 6. Updates
                # Object (Phase update)
                n_map_est.real += alpha_obj * cp.imag(grad_O) / norm_O

                # Probe (Complex update)
                probe_est += alpha_probe * cp.conj(grad_P) / norm_P

            loss_history.append(epoch_loss.item())

            # --- SAVE ITERATION FIGURES ---
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: Loss {epoch_loss.item():.4e}")

                # Save Object Est
                save_plot(
                    n_map_est,
                    f"iterations/obj_est_ep{epoch:03d}.png",
                    f"Object Est (Real) - Epoch {epoch}",
                    mode="real",
                )

                # Save Object Grad
                save_plot(
                    last_grad_O,
                    f"iterations/grad_obj_ep{epoch:03d}.png",
                    f"Object Grad Mag - Epoch {epoch}",
                    mode="mag",
                )

                # Save Probe Grad
                save_plot(
                    last_grad_P,
                    f"iterations/grad_probe_ep{epoch:03d}.png",
                    f"Probe Grad - Epoch {epoch}",
                    mode="1d_complex",
                    ylabel=None,
                )

        return n_map_est, probe_est, loss_history


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    solver = ParallelMultisliceSolverBatched(
        dx=params["dx"],
        nx=params["window_padded"],
        nz_steps=params["nz"],
        dz=params["dz"],
        wavelength=params["wavelength"],
        probe_dia=params["probe_dia"],
    )

    n_map_gt, probe_gt = generate_ground_truth()

    # 1. Generate Data
    max_start = params["global_nx"] - params["window_nx"]
    scan_indices = np.linspace(0, max_start, 64).astype(int)
    if HAS_GPU:
        scan_indices = cp.asarray(scan_indices)

    logging.info("Running Forward Simulation (GT)...")
    probe_batch = cp.tile(probe_gt, (64, 1))
    vol_out = solver.run_ptycho_batch(probe_batch, scan_indices, n_map_gt, n_iter=2)

    pad = params["pad_size"]
    exit_waves = vol_out[:, pad:-pad, -1]
    meas_amp = cp.abs(exit_waves)

    # 2. Save Initial GT Figures
    save_plot(n_map_gt, "GT_Object.png", "Ground Truth Object", mode="real")
    save_plot(probe_gt, "GT_Probe.png", "GT Probe", mode="1d_complex", ylabel=None)
    save_plot(
        meas_amp,
        "GT_Data_ExitWaves.png",
        "Exit Wave Intensities (Data)",
        xlabel="Detector Pixels",
        ylabel="Scan Pos",
        mode="abs",
    )

    # 3. Run Reconstruction
    reconstructor = JointReconstructor(solver, scan_indices, meas_amp)
    n_est, p_est, loss = reconstructor.run(n_epochs=100, batch_size=8, use_gs=True)

    # 4. Final Plots
    save_plot(n_est, "Final_Object.png", "Final Reconstructed Object", mode="real")
    save_plot(
        p_est,
        "Final_Probe.png",
        "Final Reconstructed Probe",
        mode="1d_complex",
        ylabel=None,
    )

    # Loss Curve
    plt.figure()
    plt.plot(loss)
    plt.yscale("log")
    plt.title("Reconstruction Error")
    plt.xlabel("Epoch")
    plt.ylabel("L2 Loss")
    plt.savefig(os.path.join(results_dir, "loss_curve.png"))
    plt.close()

    logging.info(f"Done. Check {results_dir} for results.")
