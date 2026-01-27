import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List


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
        os.path.join(output_dir, f"{test_case_name}_4_fine_coarse.png"), dpi=150
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
    Plots the full beam propagation history (Z-X plane).
    Dynamically creates subplots for GT + every solver in beam_histories.
    """
    # Extent: [Left, Right, Bottom, Top] -> [Z_min, Z_max, X_max, X_min]
    extent_map = [0, sample_thick_um, physical_width_um, 0]

    # Calculate rows: 1 for GT + 1 for each solver
    n_solvers = len(beam_histories)
    rows = 1 + n_solvers

    # Adjust figure height based on number of solvers
    fig = plt.figure(figsize=(16, 3 * rows))
    gs = fig.add_gridspec(rows, 1, hspace=0.3)

    # --- 1. Plot Ground Truth (Always Top) ---
    ax_gt = fig.add_subplot(gs[0, 0])
    im_gt = ax_gt.imshow(
        np.abs(beam_gt), extent=extent_map, aspect="auto", cmap="inferno"
    )
    ax_gt.set_title("Ground Truth Beam (Fine Padé)")
    ax_gt.set_ylabel("X (um)")
    if n_solvers > 0:
        ax_gt.set_xticks([])  # Hide X-ticks if other plots follow
    else:
        ax_gt.set_xlabel("Z (um)")
    # Colorbar limits
    im_gt.set_clim(0, np.max(np.abs(beam_gt)) * 1.1)
    plt.colorbar(im_gt, ax=ax_gt, label="|u|")

    # --- 2. Plot Each Solver ---
    for idx, (name, beam) in enumerate(beam_histories.items(), start=1):
        ax = fig.add_subplot(gs[idx, 0])
        im = ax.imshow(np.abs(beam), extent=extent_map, aspect="auto", cmap="inferno")
        ax.set_title(f"Solver: {name} (N={step_count_disp})")
        ax.set_ylabel("X (um)")

        # Only show X-label on the very bottom plot
        if idx == n_solvers:
            ax.set_xlabel("Z (um)")
        else:
            ax.set_xticks([])
        # Colorbar limits
        im.set_clim(0, np.max(np.abs(beam_gt)) * 1.1)
        plt.colorbar(im, ax=ax, label="|u|")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{test_case_name}_2_beams.png"), dpi=150)
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
                print(f"Warning: No timing data found for {name}, skipping time plot.")

    ax2.set_xlabel("Execution Time (s)")
    ax2.set_ylabel("Relative Error")
    ax2.set_title("Efficiency: Accuracy vs Time")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{test_case_name}_3_convergence.png"), dpi=150
    )
    plt.close(fig)
