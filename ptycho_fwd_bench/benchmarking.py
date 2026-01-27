import yaml
import numpy as np
import logging
from time import process_time
from typing import Dict, Any, Tuple, Optional

# --- PyRAM Imports ---
from ptycho_fwd_bench.generators import (
    generate_blob_phantom,
    generate_gravel_phantom,
    generate_waveguide_phantom,
    generate_branching_phantom,
    generate_fiber_bundle_phantom,
    interpolate_to_coarse,
    get_probe_field,
)
from ptycho_fwd_bench.solver_factory import create_solver
from ptycho_fwd_bench.utils import (
    setup_output_directory,
    setup_logging,
    save_ground_truth,
    load_ground_truth,
)
from ptycho_fwd_bench.physics import (
    parse_simulation_parameters,
    validate_sampling_conditions,
)
import ptycho_fwd_bench.plotters as plotters


# --- Generator Map ---
GENERATOR_MAP = {
    "BLOBS": generate_blob_phantom,
    "GRAVEL": generate_gravel_phantom,
    "WAVEGUIDE": generate_waveguide_phantom,
    "BRANCHING": generate_branching_phantom,
    "FIBERS": generate_fiber_bundle_phantom,
}

# ==========================================
# Simulation Input Generation
# ==========================================


def generate_simulation_inputs(
    sim_params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates the high-res Refractive Index Map and the Initial Probe Field."""

    # 1. Refractive Index Map (Phantom)
    gen_func = GENERATOR_MAP.get(sim_params["sample_type"], generate_blob_phantom)
    logging.info(f"Generating Phantom ({sim_params['sample_type']})...")

    # Note: Generators expect dx in Microns, but sim_params has Meters. Convert back for generator.
    dx_um = sim_params["dx"] * 1e6

    n_map_fine = gen_func(
        sim_params["n_total"],
        sim_params["ground_truth_cfg"]["n_prop_fine"],
        dx=dx_um,
        **sim_params["sample_params"],
    )

    # 2. Probe Field
    logging.info("Generating Probe Field...")
    x_coords = np.arange(sim_params["n_total"]) * sim_params["dx"]
    psi_0 = get_probe_field(
        x_coords,
        sim_params["total_width"] / 2.0,
        sim_params["probe_dia"],
        sim_params["probe_focus"],
        sim_params["wavelength"],
    )

    return n_map_fine, psi_0


def compute_ground_truth(
    n_map_fine: np.ndarray, psi_0: np.ndarray, sim_params: Dict[str, Any]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Runs the high-resolution reference simulation."""

    gt_cfg = sim_params["ground_truth_cfg"]
    n_steps = gt_cfg["n_prop_fine"]
    dz_fine = sim_params["sample_thickness"] / n_steps

    logging.info(f"--- Computing Ground Truth ({gt_cfg['solver_type']}) ---")

    solver = create_solver(
        gt_cfg["solver_type"],
        gt_cfg.get("solver_params", {}),
        n_map_fine,
        sim_params,
        dz_fine,
        save_beam=True,
    )

    t0 = process_time()
    solver.run(psi_init=psi_0)
    elapsed = process_time() - t0
    logging.info(f"Ground Truth finished in {elapsed:.2f}s")

    # Extract results (crop padding)
    pad = sim_params["n_pad"]
    psi_gt = solver.get_exit_wave(n_crop=sim_params["n_total"])[pad:-pad]

    beam_field = solver.get_beam_field()
    if beam_field is not None:
        beam_field = beam_field[pad:-pad, :]

    return psi_gt, beam_field


# ==========================================
# 5. Benchmark Execution
# ==========================================


def run_benchmark_loop(
    cfg: Dict[str, Any],
    sim_params: Dict[str, Any],
    n_map_fine: np.ndarray,
    psi_0: np.ndarray,
    psi_gt: np.ndarray,
    out_dir: str,
    beam_gt: np.ndarray,
):
    """Iterates through step counts and solvers defined in YAML."""

    solvers_list = cfg["solvers"]
    step_counts = cfg["benchmark"]["step_counts"]
    save_idx = cfg["benchmark"]["save_beam_idx"]
    if save_idx < 0:
        save_idx += len(step_counts)

    # Initialize containers
    # Structure: methods[name] = {'err': [], 'times': [], 'style': ...}
    methods = {
        s["name"]: {"err": [], "times": [], "style": "o--"} for s in solvers_list
    }
    final_waves = {}
    beam_hists = {}  # To store beam propagation for the 'save_idx' run

    logging.info(f"Starting Benchmark Loop. Steps: {step_counts}")
    logging.info(f"Solvers to test: {[s['name'] for s in solvers_list]}")

    for idx, steps in enumerate(step_counts):
        do_save = idx == save_idx

        # Downsample Phantom for this step count
        n_map_coarse = interpolate_to_coarse(n_map_fine, steps)
        dz_coarse = sim_params["sample_thickness"] / steps

        logging.info(f"--- Iteration N={steps} (dz={dz_coarse * 1e6:.3f} um) ---")

        # Plot N_map preview on first iteration
        if idx == 0:
            plotters.plot_fine_vs_coarse(
                n_map_fine,
                n_map_coarse,
                cfg["experiment"]["name"],
                out_dir,
                sim_params["n_pad"],
                sim_params["physical_width"] * 1e6,
                sim_params["sample_thickness"] * 1e6,
            )

        # Run each solver in the list
        for s_conf in solvers_list:
            name = s_conf["name"]

            t0 = process_time()
            solver = create_solver(
                s_conf["type"],
                s_conf.get("solver_params", {}),
                n_map_coarse,
                sim_params,
                dz_coarse,
                save_beam=do_save,
            )
            solver.run(psi_init=psi_0)
            t_run = process_time() - t0

            # Extract Wave
            pad = sim_params["n_pad"]
            psi_out = solver.get_exit_wave(n_crop=sim_params["n_total"])[pad:-pad]

            # Metrics
            err = np.linalg.norm(psi_out - psi_gt) / np.linalg.norm(psi_gt)
            methods[name]["err"].append(err)
            methods[name]["times"].append(t_run)

            logging.info(f"  {name:<20}: Time={t_run:.4f}s, Error={err:.2e}")

            if do_save:
                key = f"{name} (N={steps})"
                final_waves[key] = psi_out
                beam = solver.get_beam_field()
                if beam is not None:
                    beam_hists[name] = beam[pad:-pad, :]

    # --- Plotting Phase ---
    logging.info("Generating plots...")
    um_scale = 1e6
    width_um = sim_params["physical_width"] * um_scale
    thick_um = sim_params["sample_thickness"] * um_scale

    # 1. Exit Waves (Amplitude & Phase)
    plotters.plot_exit_wave_comparison(
        psi_gt=psi_gt,
        final_waves=final_waves,
        physical_width_um=width_um,
        test_case_name=cfg["experiment"]["name"],
        output_dir=out_dir,
    )

    # 2. Convergence Metrics (Accuracy vs Steps & Time)
    dz_values = [sim_params["sample_thickness"] / n for n in step_counts]
    plotters.plot_convergence_metrics(
        dz_values=dz_values,
        methods_data=methods,
        output_dir=out_dir,
        test_case_name=cfg["experiment"]["name"],
    )

    # 3. Beam Propagation
    if beam_hists:  # Only plot if we actually saved beams
        plotters.plot_beam_propagation(
            beam_gt=beam_gt,
            beam_histories=beam_hists,
            physical_width_um=width_um,
            sample_thick_um=thick_um,
            step_count_disp=step_counts[save_idx],
            output_dir=out_dir,
            test_case_name=cfg["experiment"]["name"],
        )


# ==========================================
# Main Entry Point
# ==========================================


def run_full_benchmark(config_path: str):
    """Orchestrates the full benchmark process."""

    # 1. Load & Setup
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = setup_output_directory(config_path, cfg["experiment"]["name"])
    setup_logging(out_dir)

    logging.info(f"Loaded Config: {config_path}")
    logging.info(f"Description: {cfg['experiment'].get('description', '')}")

    # 2. Parse & Validate
    sim_params = parse_simulation_parameters(cfg)
    validate_sampling_conditions(sim_params)

    gt_cfg = cfg["sample"].get("ground_truth", {})
    load_path = gt_cfg.get("load_file", None)
    save_path = gt_cfg.get("save_file", None)

    # 3. Acquire Ground Truth (Load OR Compute)
    if load_path:
        # A. LOAD mode
        # If relative path, assume relative to cwd (or config dir, here using cwd)
        logging.info("--- Mode: LOAD Ground Truth ---")
        n_map_fine, psi_0, psi_gt, beam_gt = load_ground_truth(load_path)

        # Validation warning (basic)
        if n_map_fine.shape[0] != sim_params["n_total"]:
            logging.warning(
                f"WARNING: Loaded n_map shape {n_map_fine.shape} does not match current grid n_physical {sim_params['n_total']}. Errors may occur."
            )

    else:
        # B. COMPUTE mode
        logging.info("--- Mode: COMPUTE Ground Truth ---")
        n_map_fine, psi_0 = generate_simulation_inputs(sim_params)
        psi_gt, beam_gt = compute_ground_truth(n_map_fine, psi_0, sim_params)

        # Optional: Save Result
        if save_path:
            save_ground_truth(save_path, n_map_fine, psi_0, psi_gt, beam_gt)

    # 4. Run Comparison
    run_benchmark_loop(cfg, sim_params, n_map_fine, psi_0, psi_gt, out_dir, beam_gt)

    logging.info(f"Benchmark Complete. Results saved to: {out_dir}")
