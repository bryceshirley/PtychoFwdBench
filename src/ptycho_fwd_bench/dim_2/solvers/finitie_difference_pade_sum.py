import logging

import numpy as np
from scipy.linalg import solve_banded

from ptycho_fwd_bench.dim_2.solvers.sssp.pade import pade_coefficients

from .base import OpticalWaveSolver

logger = logging.getLogger(__name__)


class FiniteDifferencePadeSumSolver(OpticalWaveSolver):
    """
    Finite Difference Pade Solver.
    Uses pre-allocated C-contiguous memory templates to eliminate
    Python overhead in the inner loop.
    """

    def __init__(
        self,
        n_map: np.ndarray,
        dz: float,
        dx: float,
        wavelength: float,
        probe_dia: float = 0,
        probe_focus: float = 0,
        pade_order: int = 4,
        store_beam: bool = False,
        envelope: bool = False,
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.pade_order = pade_order

        # Pade Coeffs
        hk0 = self.dz * self.k0
        self.b_coeffs_raw, self.d_coeffs = pade_coefficients(
            hk0, self.pade_order, envelope=envelope
        )

        # ---------------------------------------------------------
        # MEMORY PRE-ALLOCATION OPTIMIZATIONS
        # ---------------------------------------------------------
        self.k0sq_dxsq = (self.k0**2) * (self.dx**2)

        # Pre-calculate constant coefficients for all Pade terms
        self.coef_dx_arr = self.b_coeffs_raw / self.k0sq_dxsq
        self.const_diag_arr = 1.0 - 2.0 * self.coef_dx_arr

        # Pre-allocate LAPACK banded matrix templates (Fortran contiguous)
        # This completely stops Python from using `np.zeros` or `np.full` in the loop.
        self.ab_templates = []
        for j in range(self.pade_order):
            ab = np.zeros((3, self.nx), dtype=complex, order="F")
            ab[0, 1:] = self.coef_dx_arr[j]  # Upper diagonal
            ab[2, :-1] = self.coef_dx_arr[j]  # Lower diagonal
            self.ab_templates.append(ab)

    def run(self, psi_init: np.ndarray = None) -> "FiniteDifferencePadeSumSolver":
        """
        Propagates the wavefront through the medium.
        """
        psi = self.initialize_wavefront(psi_init)

        if self.store_beam:
            self.beam_history = np.zeros((self.nx, self.nz_steps), dtype=complex)
            self.beam_history[:, 0] = psi

        # Propagation loop (marching in z)
        for i in range(self.nz_steps - 1):
            psi_next = self.d_coeffs[0] * psi

            # 2D Refractive Term N = n^2 - 1
            N_vals = (self.n_map[:, i] ** 2) - 1.0

            # Tight sequential loop over Pade branches
            for j in range(self.pade_order):
                # 1. Fast C-level copy of the pre-allocated template for this branch
                ab = self.ab_templates[j].copy(order="F")

                # 2. Update ONLY the main diagonal in-place
                ab[1, :] = self.const_diag_arr[j] + self.b_coeffs_raw[j] * N_vals

                # 3. Solve directly
                # overwrite_ab=True prevents SciPy from allocating hidden workspace arrays
                w_j = solve_banded((1, 1), ab, psi, overwrite_ab=True)

                # 4. Accumulate
                psi_next += self.d_coeffs[j + 1] * w_j

            psi = psi_next
            if self.store_beam:
                self.beam_history[:, i + 1] = psi

        self.psi_final = psi
        return self
