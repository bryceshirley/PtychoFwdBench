import argparse
import yaml
import os
import subprocess
import shutil
import numpy as np
from datetime import datetime
from time import process_time

# PyRAM imports
from pyram_ptycho.generators import (
    generate_blob_phantom,
    generate_gravel_phantom,
    generate_waveguide_phantom,
    generate_grin_phantom,
    generate_branching_phantom,
    generate_fiber_bundle_phantom,
    interpolate_to_coarse,
)
from pyram_ptycho.ptycho_solvers import PtychoPadeSolver, MultisliceSolver
import pyram_ptycho.plotters as plotters

# Map string names in YAML to actual function objects
GENERATOR_MAP = {
    "BLOBS": generate_blob_phantom,
    "GRAVEL": generate_gravel_phantom,
    "WAVEGUIDE": generate_waveguide_phantom,
    "GRIN": generate_grin_phantom,
    "BRANCHING": generate_branching_phantom,
    "FIBERS": generate_fiber_bundle_phantom,
}


def get_git_revision_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "Unknown (Git not found or not a repo)"


def setup_output_directory(config_path, test_case_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"{timestamp}_{test_case_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Copy config file for reproducibility
    shutil.copy(config_path, os.path.join(out_dir, "config_snapshot.yaml"))

    # Save git hash
    with open(os.path.join(out_dir, "commit_hash.txt"), "w") as f:
        f.write(get_git_revision_hash())

    return out_dir


def run_benchmark(config_path):
    # --- Load Configuration ---
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    test_case = cfg["experiment"]["name"]
    print(f"--- Starting Benchmark: {test_case} ---")

    # --- Setup Output ---
    out_dir = setup_output_directory(config_path, test_case)
    print(f"Output Directory: {out_dir}")

    # --- Physics & Grid ---
    um = 1e-6
    wavelength = cfg["physics"]["wavelength_um"] * um
    c0 = cfg["physics"].get("c0", 1500.0)

    n_physical = cfg["grid"]["n_physical"]
    n_padding = cfg["grid"]["n_padding"]
    n_total = n_physical + 2 * n_padding

    physical_width = cfg["grid"]["physical_width_um"] * um
    dx = physical_width / n_physical

    # --- Generator Setup ---
    sample_cfg = cfg["sample"]
    sample_thick = sample_cfg["thickness_um"] * um
    n_prop_fine = sample_cfg["n_prop_fine"]

    gen_func = GENERATOR_MAP.get(sample_cfg["type"])
    if not gen_func:
        raise ValueError(f"Unknown generator type: {sample_cfg['type']}")

    # Unpack specific parameters for the generator
    gen_params = sample_cfg.get("params", {})
    print(f"Generating Phantom ({sample_cfg['type']})...")

    # Note: Generators need to support **kwargs or exact matching keys in YAML
    # We pass n_total and n_prop_fine explicitly, rest via unpacking
    n_map_fine = gen_func(n_total, n_prop_fine, dx=dx / um, **gen_params)

    # --- Probe Setup ---
    probe_cfg = cfg["physics"]
    probe_dia = probe_cfg["probe_dia_um"] * um
    probe_focus = probe_cfg["probe_focus_um"] * um

    # --- Ground Truth Calculation ---
    print("Computing Ground Truth (Fine Padé)...")
    gt_solver = PtychoPadeSolver(
        n_map=n_map_fine,
        dx=dx,
        wavelength=wavelength,
        probe_dia=probe_dia,
        probe_focus=probe_focus,
        c0=c0,
        pade_order=cfg["solver"]["pade_order"],
        beam_store_resolution=(1, 1),  # Full res for GT plot
    )
    gt_solver.run()

    # Extract Data
    psi_gt = gt_solver.get_exit_wave(n_crop=n_total)[n_padding:-n_padding]
    beam_gt = gt_solver.get_beam_field()
    if beam_gt is not None:
        beam_gt = beam_gt[n_padding:-n_padding, :]

    # --- Benchmark Loop ---
    step_counts = cfg["benchmark"]["step_counts"]
    save_idx = cfg["benchmark"]["save_beam_idx"]
    # Handle negative indexing
    if save_idx < 0:
        save_idx += len(step_counts)

    methods = {
        "Multislice (Std)": {"err": [], "time": [], "style": "bo--"},
        "Padé [8,8]": {"err": [], "time": [], "style": "ms-"},
    }

    final_waves = {}
    beam_ms_fail = None
    beam_pade_coarse = None

    for idx, steps in enumerate(step_counts):
        print(f"Benchmarking N={steps}...")
        do_save_beam = idx == save_idx

        # Downsample Map
        n_map_coarse = interpolate_to_coarse(n_map_fine, steps)

        # 1. Multislice
        t0 = process_time()
        ms_solver = MultisliceSolver(
            n_map_coarse,
            dx,
            wavelength,
            probe_dia,
            probe_focus,
            symmetric=False,
            transform_type="DST",
            store_beam=do_save_beam,
        )

        # Run (passing psi_0 is technically not needed in refactored version if using run(),
        # but MultisliceSolver.run() generates its own probe internally now.
        ms_solver.run(sample_thick=sample_thick)

        t_ms = process_time() - t0
        psi_ms = ms_solver.get_exit_wave(n_crop=n_total)[n_padding:-n_padding]

        if do_save_beam:
            beam_raw = ms_solver.get_beam_field()
            if beam_raw is not None:
                beam_ms_fail = beam_raw[n_padding:-n_padding, :]

        methods["Multislice (Std)"]["err"].append(
            np.linalg.norm(psi_ms - psi_gt) / np.linalg.norm(psi_gt)
        )
        methods["Multislice (Std)"]["time"].append(t_ms)

        # 2. Padé
        t0 = process_time()
        pade_solver = PtychoPadeSolver(
            n_map=n_map_coarse,
            dx=dx,
            wavelength=wavelength,
            probe_dia=probe_dia,
            probe_focus=probe_focus,
            c0=c0,
            pade_order=cfg["solver"]["pade_order"],
            beam_store_resolution=(1, 1) if do_save_beam else None,
        )
        pade_solver.run()
        t_pade = process_time() - t0
        psi_pade = pade_solver.get_exit_wave(n_crop=n_total)[n_padding:-n_padding]

        if do_save_beam:
            beam_raw = pade_solver.get_beam_field()
            if beam_raw is not None:
                beam_pade_coarse = beam_raw[n_padding:-n_padding, :]

        methods["Padé [8,8]"]["err"].append(
            np.linalg.norm(psi_pade - psi_gt) / np.linalg.norm(psi_gt)
        )
        methods["Padé [8,8]"]["time"].append(t_pade)

        # Store specific waves for plotting
        if do_save_beam:
            final_waves[f"Coarse MS (N={steps})"] = psi_ms
            final_waves[f"Coarse Padé (N={steps})"] = psi_pade

    # --- Plotting ---
    print("Generating Plots...")

    # 1. Setup & Exit Wave
    plotters.plot_setup_and_exit_wave(
        n_map_fine,
        n_map_coarse,
        psi_gt,
        final_waves,
        physical_width / um,
        sample_thick / um,
        test_case,
        out_dir,
        n_padding,
    )

    # 2. Beams
    plotters.plot_beam_propagation(
        beam_gt,
        beam_ms_fail,
        beam_pade_coarse,
        physical_width / um,
        sample_thick / um,
        step_counts[save_idx],
        n_prop_fine,
        out_dir,
    )

    # 3. Metrics
    dz_values = [sample_thick / n for n in step_counts]
    plotters.plot_metrics_and_phase(
        dz_values, methods, psi_gt, final_waves, physical_width / um, out_dir
    )

    print(f"Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run X-ray Propagation Benchmark from YAML Config"
    )
    parser.add_argument("config", help="Path to .yml configuration file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        exit(1)

    run_benchmark(args.config)
