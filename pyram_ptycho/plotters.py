import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional


def plot_setup_and_exit_wave(
    n_map_fine: np.ndarray,
    n_map_coarse: np.ndarray,
    psi_gt: np.ndarray,
    final_waves: Dict[str, np.ndarray],
    physical_width_um: float,
    sample_thick_um: float,
    test_case_name: str,
    output_dir: str,
    padding_px: int,
):
    """
    Plots the Refractive Index maps (Fine vs Coarse) and the Exit Wave Amplitude comparison.
    """
    # Grid setup for plotting
    nx_fine, nr_fine = n_map_fine.shape
    extent_map = [0, sample_thick_um, physical_width_um, 0]

    # X-axis for 1D plots
    n_physical = nx_fine - 2 * padding_px
    x_axis = np.linspace(0, physical_width_um, n_physical)

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2)

    # 1. Refractive Index Map (Fine)
    ax1 = fig.add_subplot(gs[0, 0])
    # Crop padding for visualization
    n_view = n_map_fine[padding_px:-padding_px, :]
    im = ax1.imshow(np.real(n_view), extent=extent_map, aspect="auto", cmap="bone")
    ax1.set_title(f"Refractive Index Fine ({test_case_name})")
    ax1.set_ylabel("X (um)")
    ax1.set_xlabel("Z (um)")
    plt.colorbar(im, ax=ax1, label="Re[n]")

    # 2. Exit Wave Amplitude
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x_axis, np.abs(psi_gt), "k-", lw=2, label="GT")

    for name, wave in final_waves.items():
        ax2.plot(x_axis, np.abs(wave), "--", label=name)

    ax2.set_title("Exit Wave Amplitude")
    ax2.set_xlabel("X (um)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_setup_exit.png"), dpi=150)
    plt.close(fig)


def plot_beam_propagation(
    beam_gt: np.ndarray,
    beam_ms_fail: Optional[np.ndarray],
    beam_pade: Optional[np.ndarray],
    physical_width_um: float,
    sample_thick_um: float,
    step_count_disp: int,
    pade_fine_nz: int,
    output_dir: str,
):
    """
    Plots the full beam propagation history for GT, Multislice, and Pade.
    """
    extent_map = [0, sample_thick_um, physical_width_um, 0]

    # Determine number of subplots needed
    rows = 1
    if beam_ms_fail is not None:
        rows += 1
    if beam_pade is not None:
        rows += 1

    fig = plt.figure(figsize=(16, 4 * rows))
    gs = fig.add_gridspec(rows, 1)

    row_idx = 0

    # 1. GT Beam
    ax1 = fig.add_subplot(gs[row_idx, 0])
    im1 = ax1.imshow(
        np.abs(beam_gt).T, extent=extent_map, aspect="auto", cmap="inferno"
    )
    ax1.set_title(f"Ground Truth Beam (Padé Fine, nz={pade_fine_nz})")
    ax1.set_ylabel("X (um)")
    ax1.set_xticks([])
    plt.colorbar(im1, ax=ax1, label="|u|")
    row_idx += 1

    # 2. Multislice Failure
    if beam_ms_fail is not None:
        ax2 = fig.add_subplot(gs[row_idx, 0])
        im2 = ax2.imshow(
            np.abs(beam_ms_fail).T, extent=extent_map, aspect="auto", cmap="inferno"
        )
        ax2.set_title(f"Multislice Beam (N={step_count_disp})")
        ax2.set_ylabel("X (um)")
        if beam_pade is not None:
            ax2.set_xticks([])  # Hide x-ticks if not last
        else:
            ax2.set_xlabel("Z (um)")
        plt.colorbar(im2, ax=ax2, label="|u|")
        row_idx += 1

    # 3. Pade Coarse
    if beam_pade is not None:
        ax3 = fig.add_subplot(gs[row_idx, 0])
        im3 = ax3.imshow(
            np.abs(beam_pade).T, extent=extent_map, aspect="auto", cmap="inferno"
        )
        ax3.set_title(f"Padé Beam (N={step_count_disp})")
        ax3.set_ylabel("X (um)")
        ax3.set_xlabel("Z (um)")
        plt.colorbar(im3, ax=ax3, label="|u|")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_beams.png"), dpi=150)
    plt.close(fig)


def plot_metrics_and_phase(
    dz_values: List[float],
    methods_data: Dict,
    psi_gt: np.ndarray,
    final_waves: Dict[str, np.ndarray],
    physical_width_um: float,
    output_dir: str,
):
    """
    Plots convergence error vs step size and the phase of the exit waves.
    """
    n_pts = len(psi_gt)
    x_axis = np.linspace(0, physical_width_um, n_pts)

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2)

    # 1. Convergence
    ax1 = fig.add_subplot(gs[0, 0])
    for name, data in methods_data.items():
        if len(data["err"]) > 0:
            ax1.loglog(dz_values, data["err"], data["style"], label=name)

    ax1.set_xlabel(r"Step Size $\Delta z$ (m)")
    ax1.set_ylabel("Relative Error")
    ax1.set_title("Convergence Analysis")
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend()

    # 2. Phase
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x_axis, np.angle(psi_gt), "k-", lw=2, label="GT")

    for name, wave in final_waves.items():
        ax2.plot(x_axis, np.angle(wave), "--", label=name)

    ax2.set_title("Exit Wave Phase")
    ax2.set_xlabel("X (um)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_metrics.png"), dpi=150)
    plt.close(fig)
