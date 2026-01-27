import numpy as np

from typing import Dict, Any
from ptycho_fwd_bench.solvers import SpectralPadeSolver, MultisliceSolver
from ptycho_fwd_bench.solvers import FiniteDifferencePadeSolver

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
            transform_type=solver_params.get("transform_type", "DST"),
            solver_type=solver_params.get("solver_type", "bicgstab"),
            preconditioner=solver_params.get("preconditioner", "split_step"),
        )

    elif s_type in ["MULTISLICE", "MS"]:
        return MultisliceSolver(
            **common_args,
            symmetric=solver_params.get("symmetric", False),
            transform_type=solver_params.get("transform_type", "DST"),
        )

    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
