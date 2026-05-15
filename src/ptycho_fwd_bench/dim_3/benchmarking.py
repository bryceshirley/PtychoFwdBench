import logging
import random
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import numpy as np

# --- 3D Plotters & Utils Imports ---
import ptycho_fwd_bench.dim_3.plotters_3d as plotters
from ptycho_fwd_bench.dim_3.generators_3d import (
    generate_3d_blob_phantom,
    get_2d_airy_probe,
    interpolate_to_coarse_3d,
)
from ptycho_fwd_bench.dim_3.solver_factory_3d import create_solver
from ptycho_fwd_bench.dim_3.utils.utils import (
    load_ground_truth,
    parse_simulation_parameters,
    save_ground_truth,
)

# ==========================================
# GPU Synchronization Helper
# ==========================================


def gpu_sync():
    """Synchronize GPU if available (supports PyTorch and CuPy)."""

    try:
        import cupy as cp

        cp.cuda.Device().synchronize()
    except Exception:
        pass


# ==========================================
# Simulation Input Generation
# ==========================================


def generate_simulation_inputs(
    sim_params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    logging.info("Generating 3D Blob Phantom...")

    if "delta_n" in sim_params["sample_params"]:
        sim_params["sample_params"]["delta_n"] = float(
            sim_params["sample_params"]["delta_n"]
        )
    if "beta_n" in sim_params["sample_params"]:
        sim_params["sample_params"]["beta_n"] = float(
            sim_params["sample_params"]["beta_n"]
        )

    n_total = sim_params["n_total"]
    n_prop_fine = sim_params["ground_truth_cfg"]["n_prop_fine"]

    n_map_fine = generate_3d_blob_phantom(
        n_total,
        n_total,
        n_prop_fine,
        **sim_params["sample_params"],
    )

    logging.info("Generating 2D Probe Field...")
    psi_0 = get_2d_airy_probe(
        nx=n_total,
        ny=n_total,
        dx=sim_params["dx"],
        diameter=sim_params["probe_dia"],
        focus=sim_params["probe_focus"],
        wavelength=sim_params["wavelength"],
    )

    return n_map_fine, psi_0


def compute_ground_truth(
    n_map_fine: np.ndarray, psi_0: np.ndarray, sim_params: Dict[str, Any]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    gt_cfg = sim_params["ground_truth_cfg"]
    n_steps = gt_cfg["n_prop_fine"]
    dz_fine = sim_params["sample_thickness"] / n_steps
    save_gt_beam = gt_cfg.get("save_beam", False)

    logging.info(f"--- Computing Ground Truth ({gt_cfg['solver_type']}) ---")

    solver = create_solver(
        gt_cfg["solver_type"],
        gt_cfg.get("solver_params", {}),
        n_map_fine,
        sim_params,
        dz_fine,
        save_beam=save_gt_beam,
    )

    gpu_sync()
    t0 = perf_counter()
    solver.run(psi_init=psi_0)
    gpu_sync()
    elapsed = perf_counter() - t0

    logging.info(f"Ground Truth finished in {elapsed:.2f}s")

    pad = sim_params["n_pad"]
    psi_gt_full = solver.get_exit_wave()
    beam_field_full = solver.get_beam_field()

    if pad > 0:
        psi_gt = psi_gt_full[pad:-pad, pad:-pad]
        beam_field = (
            beam_field_full[pad:-pad, pad:-pad, :]
            if beam_field_full is not None
            else None
        )
    else:
        psi_gt = psi_gt_full
        beam_field = beam_field_full

    return psi_gt, beam_field


# ==========================================
# Benchmark Execution (Unbiased Version)
# ==========================================


def run_benchmark_loop(
    cfg: Dict[str, Any],
    sim_params: Dict[str, Any],
    n_map_fine: np.ndarray,
    psi_0: np.ndarray,
    psi_gt: np.ndarray,
    out_dir: str,
    beam_gt: Optional[np.ndarray],
):
    solvers_list = cfg["solvers"]
    step_counts = cfg["benchmark"]["step_counts"]
    N_REPEATS = cfg["benchmark"].get("n_repeats", 1)

    save_idx = cfg["benchmark"].get("save_beam_idx", None)
    if isinstance(save_idx, int) and save_idx < 0:
        save_idx += len(step_counts)

    methods = {
        s["name"]: {"err": [], "times": [], "style": "o--"} for s in solvers_list
    }
    final_waves = {}
    beam_hists = {}

    logging.info("Performing processor warm-up...")

    for s_conf in solvers_list:
        solver = create_solver(
            s_conf["type"],
            s_conf.get("solver_params", {}),
            n_map_fine[:, :, :2],  # tiny dummy volume
            sim_params,
            sim_params["sample_thickness"] / 2,
            save_beam=False,
        )
        solver.run(psi_init=psi_0)
        gpu_sync()

    logging.info("Warm-up complete.")
    logging.info(f"Starting Benchmark Loop. Steps: {step_counts}")

    for idx, steps in enumerate(step_counts):
        do_save = (save_idx is not None) and (idx == save_idx)

        n_map_coarse = interpolate_to_coarse_3d(n_map_fine, steps)
        dz_coarse = sim_params["sample_thickness"] / steps

        logging.info(f"--- Iteration N={steps} ---")

        if idx == 0:
            plotters.plot_phantom_slices(
                n_map_fine,
                filepath=f"{out_dir}/phantom_slices_fine_N{step_counts[0]}.png",
            )
            plotters.plot_phantom_slices(
                n_map_coarse,
                filepath=f"{out_dir}/phantom_slices_coarse_N{steps}.png",
            )

        solvers_shuffled = solvers_list.copy()
        random.shuffle(solvers_shuffled)

        for s_conf in solvers_shuffled:
            name = s_conf["name"]

            solver = create_solver(
                s_conf["type"],
                s_conf.get("solver_params", {}),
                n_map_coarse,
                sim_params,
                dz_coarse,
                save_beam=do_save,
            )

            times = []

            for _ in range(N_REPEATS):
                gpu_sync()
                t0 = perf_counter()
                solver.run(psi_init=psi_0)
                gpu_sync()
                times.append(perf_counter() - t0)

            t_run = float(np.median(times))

            pad = sim_params["n_pad"]
            psi_out_full = solver.get_exit_wave()
            psi_out = psi_out_full[pad:-pad, pad:-pad] if pad > 0 else psi_out_full

            err = np.linalg.norm(psi_out - psi_gt) / np.linalg.norm(psi_gt)

            methods[name]["err"].append(err)
            methods[name]["times"].append(t_run)

            logging.info(
                f"  {name:<20}: Time={t_run:.4f}s (median of {N_REPEATS}), Error={err:.2e}"
            )
            plotters.plot_wave_slice(
                psi_out,
                filepath=f"{out_dir}/final_{name.replace(' ', '_')}.png",
                title=f"{name} Exit Wave",
            )

            if do_save:
                key = f"{name} (N={steps})"
                final_waves[key] = psi_out
                beam = solver.get_beam_field()
                if beam is not None:
                    beam_hists[name] = beam[pad:-pad, pad:-pad, :] if pad > 0 else beam

    logging.info("Generating plots...")

    plotters.plot_wave_slice(
        psi_0, filepath=f"{out_dir}/initial_probe.png", title="Initial Probe"
    )

    plotters.plot_wave_slice(
        psi_gt, filepath=f"{out_dir}/initial_gt.png", title="Ground Truth Exit Wave"
    )

    # plotters.plot_comparison(psi_gt, final_waves, output_dir=out_dir)

    dz_values = [sim_params["sample_thickness"] / n for n in step_counts]

    plotters.plot_convergence_metrics(
        dz_values=dz_values,
        methods_data=methods,
        output_dir=out_dir,
        test_case_name=cfg["experiment"]["name"],
    )


# ==========================================
# Main Entry Point
# ==========================================


def run_full_benchmark(cfg: dict, out_dir: str):
    logging.info(f"Description: {cfg['experiment'].get('description', '')}")

    sim_params = parse_simulation_parameters(cfg)

    gt_cfg = cfg["sample"].get("ground_truth", {})
    load_path = gt_cfg.get("load_file", None)
    save_path = gt_cfg.get("save_file", None)

    if load_path:
        logging.info("--- Mode: LOAD Ground Truth ---")
        n_map_fine, psi_0, psi_gt, beam_gt = load_ground_truth(load_path)
    else:
        logging.info("--- Mode: COMPUTE Ground Truth ---")
        n_map_fine, psi_0 = generate_simulation_inputs(sim_params)
        psi_gt, beam_gt = compute_ground_truth(n_map_fine, psi_0, sim_params)

        if save_path:
            save_ground_truth(save_path, n_map_fine, psi_0, psi_gt, beam_gt)

    run_benchmark_loop(cfg, sim_params, n_map_fine, psi_0, psi_gt, out_dir, beam_gt)

    logging.info(f"Benchmark Complete. Results saved to: {out_dir}")
