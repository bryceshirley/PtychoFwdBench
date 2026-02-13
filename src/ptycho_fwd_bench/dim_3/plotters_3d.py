import logging

import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(probe, p_wave, s_wave):
    # Calculate difference wave (Residual)
    diff_wave = p_wave - s_wave

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # --- Row 1: Intensities (Squared Magnitudes) ---
    # We synchronize the vmin/vmax based on the Standard exit wave
    vmax_int = np.max(np.abs(s_wave) ** 2)

    axes[0, 0].imshow(np.abs(probe) ** 2, cmap="inferno", vmax=vmax_int)
    axes[0, 0].set_title("Probe Intensity")

    im01 = axes[0, 1].imshow(np.abs(p_wave) ** 2, cmap="inferno", vmax=vmax_int)
    axes[0, 1].set_title("Parallel Intensity")

    axes[0, 2].imshow(np.abs(s_wave) ** 2, cmap="inferno", vmax=vmax_int)
    axes[0, 2].set_title("Standard Intensity")

    # Difference plot uses its own auto-scaled color bar to show subtle errors
    im03 = axes[0, 3].imshow(np.abs(diff_wave) ** 2, cmap="magma")
    axes[0, 3].set_title("Intensity Difference")

    fig.colorbar(im01, ax=axes[0, :3], location="bottom", shrink=0.6, label="Intensity")
    fig.colorbar(
        im03, ax=axes[0, 3], location="bottom", shrink=0.8, label="Res. Intensity"
    )

    # --- Row 2: Phases (Angles) ---
    # Phase is naturally bounded [-pi, pi]
    phase_bounds = {"vmin": -np.pi, "vmax": np.pi, "cmap": "twilight"}

    axes[1, 0].imshow(np.angle(probe), **phase_bounds)
    axes[1, 0].set_title("Probe Phase")

    im11 = axes[1, 1].imshow(np.angle(p_wave), **phase_bounds)
    axes[1, 1].set_title("Parallel Phase")

    axes[1, 2].imshow(np.angle(s_wave), **phase_bounds)
    axes[1, 2].set_title("Standard Phase")

    # Phase difference (Residual Phase)
    im13 = axes[1, 3].imshow(np.angle(diff_wave), cmap="coolwarm")
    axes[1, 3].set_title("Phase Difference")

    fig.colorbar(im11, ax=axes[1, :3], location="bottom", label="Phase (rad)")
    fig.colorbar(im13, ax=axes[1, 3], location="bottom", label="Res. Phase")

    for ax in axes.flatten():
        ax.axis("off")
    plt.savefig("solver_comparison_detailed.png", dpi=300)
    plt.show()

    # Calculate and log RRMSE (Root Mean Square Error)
    rrmse = np.linalg.norm(diff_wave) / np.linalg.norm(s_wave)
    logging.info(f"Exit Wave RRMSE: {rrmse:.2e}")


def plot_phantom_slices(n_map):
    """
    Plots and saves a comparison of refractive decrement and absorption
    for selected slices of the 3D phantom.
    """
    slice_indices = [0, n_map.shape[2] // 2, -1]
    num_slices = len(slice_indices)
    fig, axes = plt.subplots(2, num_slices, figsize=(4 * num_slices, 8))

    # Calculate delta from n = 1 - delta + i*beta
    # We subtract 1 and take the real part to isolate the perturbation
    delta_map = -(np.real(n_map) - 1.0)
    beta_map = np.imag(n_map)

    for i, idx in enumerate(slice_indices):
        # Row 1: Refractive Decrement (Phase contrast)
        im_delta = axes[0, i].imshow(delta_map[:, :, idx], cmap="viridis")
        axes[0, i].set_title(f"Slice {idx}: Delta ($\delta$)")
        fig.colorbar(im_delta, ax=axes[0, i], shrink=0.7)

        # Row 2: Absorption (Beta)
        im_beta = axes[1, i].imshow(beta_map[:, :, idx], cmap="inferno")
        axes[1, i].set_title(f"Slice {idx}: Beta ($\\beta$)")
        fig.colorbar(im_beta, ax=axes[1, i], shrink=0.7)

    for ax in axes.flatten():
        ax.axis("off")

    plt.suptitle("3D Phantom Internal Structure (X-ray Refractive Index)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("phantom_slices.png")
    plt.show()
