from typing import Any, Dict

import numpy as np

from ptycho_fwd_bench.solvers import (
    ExactParallelSolver,
    FiniteDifferencePadeSolver,
    MultisliceSolver,
    ParallelMultisliceSolver,
    ParallelMultisliceSolver_v2,
    SpectralPadeSolver,
    WaveletMultisliceSolver,
)

# --------------------------------------------
# Solver Factory
# --------------------------------------------


def create_solver(
    solver_type: str,
    solver_params: Dict[str, Any],
    n_map: np.ndarray,
    sim_params: Dict[str, Any],
    dz: float,
    save_beam: bool = False,
):
    s_type = solver_type.upper()

    common_args = {
        "n_map": n_map,
        "dx": sim_params["dx"],
        "wavelength": sim_params["wavelength"],
        "probe_dia": sim_params["probe_dia"],
        "probe_focus": sim_params["probe_focus"],
        "dz": dz,
        "store_beam": save_beam,
    }

    if s_type == "PADE":
        store_res = (1, 1) if save_beam else None
        # Remove 'store_beam' from common args as PyRAM uses specific resolution arg
        pyram_args = common_args.copy()
        del pyram_args["store_beam"]

        return FiniteDifferencePadeSolver(
            **pyram_args,
            pade_order=solver_params.get("pade_order", 8),
            beam_store_resolution=store_res,
        )

    elif s_type == "SPECTRAL_PADE":
        return SpectralPadeSolver(
            **common_args,
            pade_order=solver_params.get("pade_order", 8),
            max_iter=solver_params.get("max_iter", 4),
            envelope=solver_params.get("envelope", False),
            mode=solver_params.get("mode", "spectral"),
            transform_type=solver_params.get("transform_type", "FFT"),
            solver_type=solver_params.get("solver_type", "bicgstab"),
            preconditioner=solver_params.get("preconditioner", "split_step"),
        )

    elif s_type in ["MULTISLICE", "MS"]:
        return MultisliceSolver(
            **common_args,
            symmetric=solver_params.get("symmetric", True),
            transform_type=solver_params.get("transform_type", "FFT"),
            use_richardson=solver_params.get("use_richardson", False),
        )

    elif s_type == "PARAMS":
        return ParallelMultisliceSolver(
            **common_args,
            alpha=solver_params.get("alpha", 1e-3),
            n_iter=solver_params.get("n_iter", 2),
            solver_type=solver_params.get("solver_type", "richardson"),
        )
    elif s_type == "PARAMS2":
        return ParallelMultisliceSolver_v2(
            **common_args,
            alpha=solver_params.get("alpha", 1e-3),
            n_iter=solver_params.get("n_iter", 2),
            solver_type=solver_params.get("solver_type", "richardson"),
        )

    elif s_type == "WAVELET_MULTISLICE":
        wavelet_solver = WaveletMultisliceSolver(
            **common_args,
            symmetric=solver_params.get("symmetric", True),
            wavelet_name=solver_params.get("wavelet_name", "db4"),
            compression_threshold=solver_params.get("compression_threshold", 1e-5),
        )
        wavelet_solver.build_propagator_matrix()
        return wavelet_solver

    elif s_type == "EXACTPARALLEL":
        return ExactParallelSolver(
            **common_args,
            alpha=solver_params.get("alpha", 1e-6),
            n_iter=solver_params.get("n_iter", 2),
            solver_type=solver_params.get("solver_type", "gmres"),
        )

    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
