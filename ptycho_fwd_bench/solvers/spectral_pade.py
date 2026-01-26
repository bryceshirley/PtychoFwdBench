import numpy as np
from .base import OpticalWaveSolver
from .utils import apply_spectral_kernel, get_spectral_coords
from ptycho_fwd_bench.sssp.pade import pade_coefficients


class SpectralPadeSolver(OpticalWaveSolver):
    """
    Spectral Pade Solver implementing High-Order Spectral Split-Step Pade.

    Parameters
    ----------
    n_map : np.ndarray
        Complex refractive index map.
    dz : float
        Step size in propagation direction.
    dx : float
        Spatial grid spacing.
    wavelength : float
        Wavelength of the wave.
    probe_dia : float, optional
        Diameter of the probe beam.
    probe_focus : float, optional
        Focus position of the probe beam.
    pade_order : int, optional
        Order of the Pade approximation.
    transform_type : str, optional
        Type of spectral transform ("DST" or "FFT").
    max_iter : int, optional
        Maximum iterations for Richardson solver.
    store_beam : bool, optional
        Whether to store beam history.
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
        transform_type: str = "DST",
        max_iter: int = 4,
        store_beam: bool = False,
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.pade_order = pade_order
        self.max_iter = max_iter
        self.transform_type = transform_type

        # Pade Coeffs
        hk0 = self.dz * self.k0
        self.b_coeffs_raw, self.d_coeffs = pade_coefficients(hk0, self.pade_order)

        # Spectral Operator Lambda = -kx^2
        kx = get_spectral_coords(self.nx, self.dx, self.transform_type)
        self.Lambda = -(kx**2)

    def _apply_diffraction(self, psi: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return apply_spectral_kernel(psi, kernel, self.transform_type)

    def run(self, psi_init: np.ndarray = None) -> "SpectralPadeSolver":
        """
        Propagates the wavefront through the medium using Spectral Pade method.
        Parameters:
            psi_init: Initial wavefront (np.ndarray)
        Returns:
            self: Updated solver with final wavefront
        """
        # Base class handles creation or validation
        psi = self.initialize_wavefront(psi_init)

        if self.store_beam:
            self.beam_history = np.zeros((self.nx, self.nz_steps), dtype=complex)
            self.beam_history[:, 0] = psi

        # Propagation loop
        for i in range(self.nz_steps - 1):
            # 1. Initialize Summation: d0 * psi
            psi_next = self.d_coeffs[0] * psi

            # 2. Define Refractive Term N = n^2 - 1
            N_vals = (self.n_map[:, i] ** 2) - 1.0

            # 3. Add partial fraction terms sum(d_j * w_j)
            for j in range(self.pade_order):
                b_j = self.b_coeffs_raw[j]
                d_j = self.d_coeffs[j + 1]
                w_j = self._solve_pade_term(psi, b_j, N_vals)
                psi_next += d_j * w_j

            psi = psi_next
            if self.store_beam:
                self.beam_history[:, i + 1] = psi

        self.psi_final = psi
        return self

    def _solve_pade_term(
        self, psi: np.ndarray, b_j: complex, N_vals: np.ndarray
    ) -> np.ndarray:
        """
        Solves (1 + b_j * X) w = psi using Preconditioned Richardson Iteration.
        where X = L_scaled + N, L_scaled = (1/k0^2) * L

        Parameters:
        ----------
        psi : np.ndarray
            Input wavefront.
        b_j : complex
            Pade coefficient.
        N_vals : np.ndarray
            Refractive index squared deviation values.
        Returns:
        ---------
        w : np.ndarray
            Solution wavefront.
        """
        # Preconditioner M = (1 + b_j/k0^2 * L)(1 + b_j * N)
        b_diff = b_j / self.k0sq
        M_L_kernel = 1.0 + b_diff * self.Lambda
        M_N_vals = 1.0 + b_j * N_vals

        # Split-Step Preconditioned Initial Guess w0 = M^-1 psi
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_M_L = 1.0 / M_L_kernel
            inv_M_L[np.isclose(M_L_kernel, 0)] = 0.0
        w = (self._apply_diffraction(psi, inv_M_L)) / M_N_vals

        # Richardson Iteration
        for _ in range(self.max_iter):
            # A. Compute A * w = (1 + b L_scaled + b N) w
            # L term
            term_L = b_diff * self._apply_diffraction(w, self.Lambda)

            # N term
            term_N = b_j * N_vals * w

            # A_w
            A_w = w + term_L + term_N

            # Residual and Correction
            residual = psi - A_w

            # C. Apply Preconditioner M^-1 to residual
            correction_L = self._apply_diffraction(residual, inv_M_L)
            correction = correction_L / M_N_vals

            # Update
            w += correction

        return w
