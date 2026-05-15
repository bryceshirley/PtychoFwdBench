import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_wave_slice(wave, filepath="exit_wave.png", title="Exit Wave Slice"):
    """Plots the initial probe intensity and phase in a separate figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Intensity
    im0 = axes[0].imshow(np.abs(wave), cmap="inferno")
    axes[0].set_title(f"{title} Modulus")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    # Phase
    im1 = axes[1].imshow(np.angle(wave), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f"{title} Phase")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_comparison(psi_gt, final_waves, output_dir="."):
    """
    Plots the ground truth vs each solver exit wave on separate figures.
    """
    phase_bounds = {"vmin": -np.pi, "vmax": np.pi, "cmap": "twilight"}
    vmax_int = np.max(np.abs(psi_gt) ** 2)

    for name, s_wave in final_waves.items():
        diff_wave = psi_gt - s_wave
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Comparison: Ground Truth vs {name}", fontsize=16)

        # --- Row 1: Intensities ---
        axes[0, 0].imshow(np.abs(psi_gt) ** 2, cmap="inferno", vmax=vmax_int)
        axes[0, 0].set_title("GT Intensity")

        axes[0, 1].imshow(np.abs(s_wave) ** 2, cmap="inferno", vmax=vmax_int)
        axes[0, 1].set_title(f"{name} Intensity")

        im_diff_int = axes[0, 2].imshow(np.abs(diff_wave) ** 2, cmap="magma")
        axes[0, 2].set_title("Intensity Difference (Residual)")
        fig.colorbar(im_diff_int, ax=axes[0, 2], shrink=0.7)

        # --- Row 2: Phases ---
        axes[1, 0].imshow(np.angle(psi_gt), **phase_bounds)
        axes[1, 0].set_title("GT Phase")

        axes[1, 1].imshow(np.angle(s_wave), **phase_bounds)
        axes[1, 1].set_title(f"{name} Phase")

        im_diff_phs = axes[1, 2].imshow(np.angle(diff_wave), cmap="coolwarm")
        axes[1, 2].set_title("Phase Difference")
        fig.colorbar(im_diff_phs, ax=axes[1, 2], shrink=0.7)

        for ax in axes.flatten():
            ax.axis("off")

        # Clean the name for a filename
        clean_name = (
            name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        )
        logging.info(
            f"Saving comparison plot for {output_dir}/compare_{clean_name}.png"
        )
        plt.savefig(f"{output_dir}/compare_{clean_name}.png", dpi=300)
        plt.close()

        # Calculate and log RRMSE
        rrmse = np.linalg.norm(diff_wave) / np.linalg.norm(psi_gt)
        logging.info(f"{name} - Exit Wave RRMSE: {rrmse:.2e}")


def plot_phantom_slices(n_map, filepath="phantom_data.npz"):
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
    plt.savefig(filepath)
    plt.show()


def plot_convergence_metrics(
    dz_values: List[float],
    methods_data: Dict,
    output_dir: str,
    test_case_name: str,
):
    """
    Plots metric convergence analysis.

    Subplot 1: Relative Error vs Step Size
    Subplot 2: Relative Error vs Execution Time

    Expects methods_data[name] to contain keys: "err", "times", "style"
    """
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2)

    # --- 1. Convergence (Error vs Step Size) ---
    ax1 = fig.add_subplot(gs[0, 0])
    for name, data in methods_data.items():
        if len(data["err"]) > 0:
            # Assuming dz_values corresponds to iterations (smaller dz = more iterations)
            ax1.loglog(
                dz_values, data["err"][: len(dz_values)], data["style"], label=name
            )

    ax1.set_xlabel(r"Step Size $\Delta z$ (m)")
    ax1.set_ylabel("Relative Error")
    ax1.set_title("Convergence: Accuracy vs Step Size")
    # Invert X axis so smaller steps (more computation) are on the right
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend()

    # --- 2. Efficiency (Error vs Time) ---
    ax2 = fig.add_subplot(gs[0, 1])
    for name, data in methods_data.items():
        if len(data["err"]) > 0:
            # Check if time data exists in the dictionary
            if "times" in data and len(data["times"]) == len(data["err"]):
                ax2.loglog(data["times"], data["err"], data["style"], label=name)
            else:
                logging.warning(
                    f"Warning: No timing data found for {name}, skipping time plot."
                )

    ax2.set_xlabel("Execution Time (s)")
    ax2.set_ylabel("Relative Error")
    ax2.set_title("Efficiency: Accuracy vs Time")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{test_case_name}_2_convergence.png"), dpi=150
    )
    plt.close(fig)
