import logging

import numba as nb
import numpy as np

from ptycho_fwd_bench.dim_2.solvers.sssp.pade import pade_coefficients

from .base import OpticalWaveSolver3D

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# NUMBA JIT KERNELS (COUPLED REFRACTIVE INDEX)
# ---------------------------------------------------------
@nb.njit(parallel=True, fastmath=True)
def _numba_sweep_x(psi_in, d_coeffs, b_coeffs_raw, coef_dx, n_term):
    """
    Parallelizes X-operator sweeps across the Y spatial dimension.
    Couples the spatially varying refractive index (n_term) into the Thomas diagonal.
    """
    ny, nx = psi_in.shape
    pade_order = len(coef_dx)
    out = np.empty((ny, nx), dtype=np.complex128)

    # prange dynamically balances the workload across all CPU cores
    for y in nb.prange(ny):
        line_in = psi_in[y, :]
        n_line = n_term[y, :]
        line_out = d_coeffs[0] * line_in

        # Thread-local memory allocation
        w = np.empty(nx, dtype=np.complex128)
        c_prime = np.empty(nx, dtype=np.complex128)
        d_prime = np.empty(nx, dtype=np.complex128)

        for j in range(pade_order):
            a = coef_dx[j]
            c = coef_dx[j]
            b_j = b_coeffs_raw[j]

            # Base diagonal (vacuum diffraction)
            b_base = 1.0 - 2.0 * a

            # Inline Thomas Algorithm (Forward Elimination)
            # Spatially varying diagonal: b_i = b_base + b_j * 0.5 * (n^2 - 1)
            b_0 = b_base + b_j * n_line[0]
            c_prime[0] = c / b_0
            d_prime[0] = line_in[0] / b_0

            for i in range(1, nx):
                b_i = b_base + b_j * n_line[i]
                m = 1.0 / (b_i - a * c_prime[i - 1])
                c_prime[i] = c * m
                d_prime[i] = (line_in[i] - a * d_prime[i - 1]) * m

            # Back Substitution
            w[nx - 1] = d_prime[nx - 1]
            for i in range(nx - 2, -1, -1):
                w[i] = d_prime[i] - c_prime[i] * w[i + 1]

            line_out += d_coeffs[j + 1] * w

        out[y, :] = line_out
    return out


@nb.njit(parallel=True, fastmath=True)
def _numba_sweep_y(psi_in, d_coeffs, b_coeffs_raw, coef_dy, n_term):
    """
    Parallelizes Y-operator sweeps across the X spatial dimension.
    Couples the spatially varying refractive index (n_term) into the Thomas diagonal.
    """
    ny, nx = psi_in.shape
    pade_order = len(coef_dy)
    out = np.empty((ny, nx), dtype=np.complex128)

    for x in nb.prange(nx):
        # Extract memory-contiguous column for cache efficiency
        line_in = np.empty(ny, dtype=np.complex128)
        n_line = np.empty(ny, dtype=np.complex128)
        for y in range(ny):
            line_in[y] = psi_in[y, x]
            n_line[y] = n_term[y, x]

        line_out = d_coeffs[0] * line_in

        w = np.empty(ny, dtype=np.complex128)
        c_prime = np.empty(ny, dtype=np.complex128)
        d_prime = np.empty(ny, dtype=np.complex128)

        for j in range(pade_order):
            a = coef_dy[j]
            c = coef_dy[j]
            b_j = b_coeffs_raw[j]

            b_base = 1.0 - 2.0 * a

            b_0 = b_base + b_j * n_line[0]
            c_prime[0] = c / b_0
            d_prime[0] = line_in[0] / b_0

            for i in range(1, ny):
                b_i = b_base + b_j * n_line[i]
                m = 1.0 / (b_i - a * c_prime[i - 1])
                c_prime[i] = c * m
                d_prime[i] = (line_in[i] - a * d_prime[i - 1]) * m

            w[ny - 1] = d_prime[ny - 1]
            for i in range(ny - 2, -1, -1):
                w[i] = d_prime[i] - c_prime[i] * w[i + 1]

            line_out += d_coeffs[j + 1] * w

        for y in range(ny):
            out[y, x] = line_out[y]

    return out


# ---------------------------------------------------------
# SOLVER CLASS
# ---------------------------------------------------------
class FiniteDifferencePadeSumABCSolver3D(OpticalWaveSolver3D):
    """
    3D Finite Difference Pade Solver (Fully Coupled Refractive Index).
    Uses Numba @njit(parallel=True) to distribute spatial sweeps across CPU cores.
    Supports standard Dirichlet ("none") or Absorbing Boundary Conditions ("abc").
    """

    def __init__(
        self,
        n_map: np.ndarray,
        dx: float,
        wavelength: float,
        probe_dia: float = 0,
        probe_focus: float = 0,
        dz: float = 0.0,
        pade_order: int = 4,
        cross_term_order: int = 0,
        store_beam: bool = False,
        boundary_type: str = "abc",  # "none" or "abc"
        boundary_width: int = 32,
        boundary_strength: float = 3.0,
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.pade_order = pade_order
        self.cross_term_order = cross_term_order

        self.boundary_type = boundary_type.lower()
        self.boundary_width = boundary_width
        self.boundary_strength = boundary_strength

        if self.boundary_type not in ["none", "abc"]:
            logger.warning(
                f"Unknown boundary type '{self.boundary_type}'. Defaulting to 'none'."
            )
            self.boundary_type = "none"

        hk0 = self.dz * self.k0
        self.b_coeffs_raw, self.d_coeffs = pade_coefficients(
            hk0, self.pade_order, envelope=True
        )

        self.k0sq_dxsq = (self.k0**2) * (self.dx**2)
        self.coef_dx_arr = self.b_coeffs_raw / self.k0sq_dxsq

        self._setup_boundaries()

    def _setup_boundaries(self):
        """Initializes ABC absorption masks."""
        self.abc_mask = None

        if self.boundary_type == "abc":
            mask_x = np.ones(self.nx)
            mask_y = np.ones(self.ny)
            w = self.boundary_width

            ramp = np.linspace(0, 1, w) ** 2
            mask_x[:w] = ramp
            mask_x[-w:] = ramp[::-1]
            mask_y[:w] = ramp
            mask_y[-w:] = ramp[::-1]

            mask_2d = np.outer(mask_y, mask_x).astype(np.complex128)
            self.abc_mask = np.exp(-self.boundary_strength * (1.0 - mask_2d))

    def _apply_X_operator(self, psi: np.ndarray, n_term: np.ndarray) -> np.ndarray:
        """Applies the continuous X operator: 0.5*(n^2-1) + (1/k0^2) d^2/dx^2"""
        out = np.zeros_like(psi)
        out[:, 1:-1] = (psi[:, 2:] - 2 * psi[:, 1:-1] + psi[:, :-2]) / self.k0sq_dxsq
        out += n_term * psi
        return out

    def _apply_Y_operator(self, psi: np.ndarray, n_term: np.ndarray) -> np.ndarray:
        """Applies the continuous Y operator: 0.5*(n^2-1) + (1/k0^2) d^2/dy^2"""
        out = np.zeros_like(psi)
        out[1:-1, :] = (psi[2:, :] - 2 * psi[1:-1, :] + psi[:-2, :]) / self.k0sq_dxsq
        out += n_term * psi
        return out

    def _apply_cross_correction(
        self, psi: np.ndarray, n_term: np.ndarray
    ) -> np.ndarray:
        """Applies diffractive/refractive cross-terms to mitigate splitting error."""
        if self.cross_term_order < 1:
            return psi

        sigma = self.k0 * self.dz
        psi_corrected = psi.copy()
        delta_m = psi.copy()

        for m in range(1, self.cross_term_order + 1):
            phi_Y = self._apply_Y_operator(delta_m, n_term)
            phi_XY = self._apply_X_operator(phi_Y, n_term)
            delta_m = (-1j * sigma / (4.0 * m)) * phi_XY
            psi_corrected += delta_m

        return psi_corrected

    def run(self, psi_init: np.ndarray = None) -> "FiniteDifferencePadeSumABCSolver3D":
        psi = self.initialize_wavefront(psi_init)

        if self.store_beam:
            self.beam_history = np.zeros(
                (self.ny, self.nx, self.nz_steps), dtype=complex
            )
            self.beam_history[:, :, 0] = psi

        for i in range(self.nz_steps - 1):
            n_slice = self.n_map[:, :, i]

            # Formulate the spatially varying refractive term: 0.5 * (n^2 - 1)
            # Cast to complex128 to match Numba loop types
            n_term = (0.5 * (n_slice**2 - 1.0)).astype(np.complex128)

            # Phase 1: Diffractive/Refractive Cross-Term Correction
            psi_cross = self._apply_cross_correction(psi, n_term)

            # Phase 2 & 3: Implicit X and Y Operator Splits via Numba
            psi_x = _numba_sweep_x(
                psi_cross, self.d_coeffs, self.b_coeffs_raw, self.coef_dx_arr, n_term
            )
            psi_next = _numba_sweep_y(
                psi_x, self.d_coeffs, self.b_coeffs_raw, self.coef_dx_arr, n_term
            )

            # Apply ABC mask if using Sponge layer (applied after the step)
            if self.boundary_type == "abc" and self.abc_mask is not None:
                psi_next *= self.abc_mask

            psi = psi_next
            if self.store_beam:
                self.beam_history[:, :, i + 1] = psi

        self.psi_final = psi
        return self
