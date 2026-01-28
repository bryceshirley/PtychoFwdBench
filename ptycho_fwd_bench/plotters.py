import logging
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_fine_vs_coarse(
    n_map_fine: np.ndarray,
    n_map_coarse: np.ndarray,
    test_case_name: str,
    output_dir: str,
    padding_px: int,
    physical_width_um: float,
    sample_thick_um: float,
):
    """
    Plots the comparison between the Fine Ground Truth Refractive Index map
    and the Coarse approximation used for the lowest resolution step.

    Restores the 'Figure 4' functionality from the original script.
    """
    # Grid extent
    extent_map = [
        0,
        sample_thick_um,
        physical_width_um,
        0,
    ]  # [Left, Right, Bottom, Top]

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2)

    # 1. Fine Map
    ax1 = fig.add_subplot(gs[0, 0])
    n_view_fine = n_map_fine[padding_px:-padding_px, :]
    im1 = ax1.imshow(
        np.real(n_view_fine), extent=extent_map, aspect="auto", cmap="bone"
    )
    ax1.set_title(f"Refractive Index Fine ({test_case_name})")
    ax1.set_ylabel("X (um)")
    ax1.set_xlabel("Z (um)")
    # colorbar limits
    plt.colorbar(im1, ax=ax1, label="Re[n]")
    n_max = np.max(np.real(n_view_fine))
    n_min = np.min(np.real(n_view_fine))
    im1.set_clim(n_min, n_max)

    # 2. Coarse Map
    ax2 = fig.add_subplot(gs[0, 1])
    n_view_coarse = n_map_coarse[padding_px:-padding_px, :]
    im2 = ax2.imshow(
        np.real(n_view_coarse), extent=extent_map, aspect="auto", cmap="bone"
    )
    ax2.set_title(f"Refractive Index Coarse ({test_case_name})")
    ax2.set_ylabel("X (um)")
    ax2.set_xlabel("Z (um)")
    # colorbar limits
    plt.colorbar(im2, ax=ax2, label="Re[n]")
    im2.set_clim(n_min, n_max)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{test_case_name}_3_fine_coarse.png"), dpi=150
    )
    plt.close(fig)


def plot_exit_wave_comparison(
    psi_gt: np.ndarray,
    final_waves: Dict[str, np.ndarray],
    physical_width_um: float,
    test_case_name: str,
    output_dir: str,
):
    """
    Plots the Exit Wave Amplitude and Phase on a single figure with two subplots.
    """
    # X-axis for 1D plot
    n_physical_plot = len(psi_gt)
    x_axis = np.linspace(0, physical_width_um, n_physical_plot)

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2)

    # --- 1. Exit Wave Amplitude ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x_axis, np.abs(psi_gt), "k-", lw=2, label="GT")
    for name, wave in final_waves.items():
        ax1.plot(x_axis, np.abs(wave), "--", label=name)

    ax1.set_title(f"Exit Wave Amplitude ({test_case_name})")
    ax1.set_xlabel("X (um)")
    ax1.set_ylabel(r"|\psi|")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- 2. Exit Wave Phase ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x_axis, np.angle(psi_gt), "k-", lw=2, label="GT")
    for name, wave in final_waves.items():
        ax2.plot(x_axis, np.angle(wave), "--", label=name)

    ax2.set_title(f"Exit Wave Phase ({test_case_name})")
    ax2.set_xlabel("X (um)")
    ax2.set_ylabel("Phase (rad)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{test_case_name}_1_exit_waves.png"), dpi=150)
    plt.close(fig)


def plot_beam_propagation(
    beam_gt: np.ndarray,
    beam_histories: Dict[str, np.ndarray],
    physical_width_um: float,
    sample_thick_um: float,
    step_count_disp: int,
    output_dir: str,
    test_case_name: str,
):
    """
    Plots the beam propagation history (Z-X plane).
    Creates a 'beam_comparisons' folder and saves a separate plot
    for each solver compared against the Ground Truth.
    """
    # 1. Setup Output Directory
    beams_dir = os.path.join(output_dir, "beam_comparisons")
    os.makedirs(beams_dir, exist_ok=True)

    # 2. Define Extents and Limits
    # Extent: [Left, Right, Bottom, Top] -> [Z_min, Z_max, X_max, X_min]
    extent_map = [0, sample_thick_um, physical_width_um, 0]

    # Establish consistent color limits based on Ground Truth
    # This ensures all plots share the exact same intensity scale.
    v_min = 0
    v_max = np.max(np.abs(beam_gt)) * 1.1
    # Phase limits
    v_min_phase = -np.pi
    v_max_phase = np.pi

    # 3. Iterate through each solver and plot against GT
    for name, beam_solver in beam_histories.items():
        # Create a figure with 2 rows: Top=GT, Bottom=Solver
        fig = plt.figure(figsize=(12, 12), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)

        # --- Top: Ground Truth ---
        ax_gt = fig.add_subplot(gs[0, 0])
        im_gt = ax_gt.imshow(
            np.abs(beam_gt),
            extent=extent_map,
            aspect="auto",
            cmap="inferno",
            vmin=v_min,
            vmax=v_max,
        )
        ax_gt.set_title("Reference: Ground Truth Amplitude (Fine Padé)")
        ax_gt.set_ylabel("X (um)")
        ax_gt.set_xlabel("Z (um)")
        plt.colorbar(im_gt, ax=ax_gt, label="|u|")

        ax_gt = fig.add_subplot(gs[0, 1])
        im_gt = ax_gt.imshow(
            np.angle(beam_gt),
            extent=extent_map,
            aspect="auto",
            cmap="inferno",
            vmin=v_min_phase,
            vmax=v_max_phase,
        )
        ax_gt.set_title("Reference: Ground Truth Phase (Fine Padé)")
        ax_gt.set_ylabel("X (um)")
        ax_gt.set_xlabel("Z (um)")
        plt.colorbar(im_gt, ax=ax_gt, label="Phase (rad)")

        # --- Bottom: Specific Solver ---
        ax_sol = fig.add_subplot(gs[1, 0])
        im_sol = ax_sol.imshow(
            np.abs(beam_solver),
            extent=extent_map,
            aspect="auto",
            cmap="inferno",
            vmin=v_min,
            vmax=v_max,
        )
        ax_sol.set_title(f"Solver: {name} (N={step_count_disp}) Amplitude")
        ax_sol.set_ylabel("X (um)")
        ax_sol.set_xlabel("Z (um)")
        plt.colorbar(im_sol, ax=ax_sol, label="|u|")

        ax_sol = fig.add_subplot(gs[1, 1])
        im_sol = ax_sol.imshow(
            np.angle(beam_solver),
            extent=extent_map,
            aspect="auto",
            cmap="inferno",
            vmin=v_min_phase,
            vmax=v_max_phase,
        )
        ax_sol.set_title(f"Solver: {name} (N={step_count_disp}) Phase")
        ax_sol.set_ylabel("X (um)")
        ax_sol.set_xlabel("Z (um)")
        plt.colorbar(im_sol, ax=ax_sol, label="Phase (rad)")

        # --- Save ---
        # Clean filename: remove spaces, parentheses, etc.
        clean_name = re.sub(r"[^\w\-_]", "_", name)
        filename = f"{test_case_name}_vs_{clean_name}.png"
        save_path = os.path.join(beams_dir, filename)

        plt.savefig(save_path, dpi=150)
        plt.close(fig)


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
            ax1.loglog(dz_values, data["err"], data["style"], label=name)

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
                logger.warning(
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
