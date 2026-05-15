from typing import Any, Dict

import numpy as np

# Adjust these imports to point to your actual 3D solver modules
from ptycho_fwd_bench.dim_3.solvers_3d import (
    FiniteDifferencePadeSumABCSolver3D,
    FiniteDifferencePadeSumSolver3D,
    MultisliceSolver3D,
    ParallelMultisliceSolver3D,
    ParallelMultisliceSolver3D_MGZ,
    ParallelMultisliceSolverASM3D,
    WaveletMultisliceSolver3D,
)

# --------------------------------------------
# 3D Solver Factory
# --------------------------------------------


def create_solver(
    solver_type: str,
    solver_params: Dict[str, Any],
    n_map: np.ndarray,
    sim_params: Dict[str, Any],
    dz: float,
    save_beam: bool = False,
):
    """
    Factory to instantiate 3D multislice solvers.
    """
    s_type = solver_type.upper()

    # Common arguments shared across all 3D solvers
    common_args = {
        "n_map": n_map,
        "dx": sim_params["dx"],
        "wavelength": sim_params["wavelength"],
        "probe_dia": sim_params["probe_dia"],
        "probe_focus": sim_params["probe_focus"],
        "dz": dz,
        "store_beam": save_beam,
    }

    if s_type in ["MULTISLICE", "MS"]:
        return MultisliceSolver3D(
            **common_args,
            use_gpu=solver_params.get("use_gpu", False),
            gpu_id=solver_params.get("gpu_id", 0),
            symmetric=solver_params.get("symmetric", True),
            transform_type=solver_params.get("transform_type", "FFT"),
            mode=solver_params.get("mode", "spectral"),
        )

    elif s_type == "PARAMS":
        return ParallelMultisliceSolver3D(
            **common_args,
            use_gpu=solver_params.get("use_gpu", False),
            gpu_id=solver_params.get("gpu_id", 0),
            alpha=solver_params.get("alpha", 1e-6),
            n_iter=solver_params.get("n_iter", 1),
            woodbury=solver_params.get("woodbury", True),
        )

    elif s_type == "PARAMS_MGZ":
        return ParallelMultisliceSolver3D_MGZ(
            **common_args,
            use_gpu=solver_params.get("use_gpu", False),
            gpu_id=solver_params.get("gpu_id", 0),
            n_iter=solver_params.get("n_iter", 1),
            alpha=solver_params.get("alpha", 1e-6),
            woodbury=solver_params.get("woodbury", True),
            coarsening_factor=solver_params.get("coarsening_factor", 8),
        )

    elif s_type == "PARAMS_ASM":
        return ParallelMultisliceSolverASM3D(
            **common_args,
            use_gpu=solver_params.get("use_gpu", False),
            gpu_id=solver_params.get("gpu_id", 0),
            n_iter=solver_params.get("n_iter", 2),
        )

    elif s_type in ["PADE"]:
        return FiniteDifferencePadeSumSolver3D(
            **common_args,
            pade_order=solver_params.get("pade_order", 4),
            cross_term_order=solver_params.get("cross_term_order", 1),
        )

    elif s_type in ["PADE_ABC"]:
        return FiniteDifferencePadeSumABCSolver3D(
            **common_args,
            boundary_type=solver_params.get("boundary_type", "abc"),
            boundary_width=solver_params.get("boundary_width", 32),
            boundary_strength=solver_params.get("boundary_strength", 2.0),
            pade_order=solver_params.get("pade_order", 4),
            cross_term_order=solver_params.get("cross_term_order", 1),
        )

    elif s_type == "WAVELET_MULTISLICE":
        wavelet_solver = WaveletMultisliceSolver3D(
            **common_args,
            symmetric=solver_params.get("symmetric", True),
            wv_family=solver_params.get("wv_family", "sym6"),
            wv_level=solver_params.get("wv_level", 3),
            v_s=solver_params.get("v_s", 1e-4),
        )
        return wavelet_solver

    else:
        raise ValueError(f"Unknown 3D solver type: {solver_type}")
