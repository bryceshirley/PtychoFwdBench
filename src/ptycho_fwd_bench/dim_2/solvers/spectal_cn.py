from typing import Literal, Optional

import numpy as np

from .base import OpticalWaveSolver


class SpectralCrankNicolsonSolver(OpticalWaveSolver):
    """
    Crank-Nicolson Solver for the Paraxial Wave Equation using
    Preconditioned Richardson Iteration.

    Fixes for Extrapolation Stability:
    1. Increased default `n_iter` to ensure the Coarse step converges.
    2. Explicit alignment of spectral coordinates with FFT frequencies.
    3. Correct handling of the Richardson loop.
    """

    def __init__(
        self,
        n_map: np.ndarray,
        dx: float,
        wavelength: float,
        probe_dia: float,
        probe_focus: float,
        dz: float,
        n_iter: int = 8,  # Increased from 4 to 8 for Coarse step stability
        use_extrapolation: bool = False,
        method: Literal["PARAXIAL", "ASM"] = "PARAXIAL",
        store_beam: bool = False,
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)

        self.n_iter = n_iter
        self.use_extrapolation = use_extrapolation
        self.method = method.upper()

        # Ensure k0 squared exists (handle potential base class missing attr)
        if not hasattr(self, "k0sq"):
            self.k0sq = self.k0**2

        # --- Precompute Spectral Laplacian ---
        # Using standard FFT frequency order (0, 1, ..., -N/2, ...)
        # This matches the corner-aligned FFT data layout.
        fx = np.fft.fftfreq(self.nx, d=self.dx)
        kx = 2 * np.pi * fx

        if self.method == "ASM":
            # Exact Helmholtz: i * (sqrt(k^2 - kx^2) - k)
            inside = self.k0sq - kx**2
            sqrt_term = np.sqrt(np.clip(inside, 0.0, None))
            self.L_eigenvalues = 1j * (sqrt_term - self.k0)
        else:
            # Paraxial: -i * kx^2 / (2k)
            self.L_eigenvalues = (-1j * kx**2) / (2 * self.k0)

        # Ensure complex type for operators
        self.L_eigenvalues = self.L_eigenvalues.astype(np.complex128)

    def _get_operators(self, C_slice: np.ndarray, dz: float):
        """
        Constructs M^{-1} (Preconditioner) and delta_C (Residual term).
        M = I - 0.5 * L * dz - 0.5 * C_bar * dz
        """
        # C_bar = np.mean(C_slice)
        # delta_C = C_slice - C_bar

        # M is diagonal in Fourier space.
        M_eigenvalues = 1.0 - 0.5 * (self.L_eigenvalues * dz)  # - 0.5 * (C_bar * dz)

        # M_inv is simply 1 / Eigenvalues
        M_inv_eigenvalues = 1.0 / M_eigenvalues
        delta_C_dz = C_slice * dz

        return M_inv_eigenvalues, delta_C_dz

    def _step_crank_nicolson(
        self, u_in: np.ndarray, C_slice: np.ndarray, dz: float
    ) -> np.ndarray:
        """
        Solves (I - 0.5(L+C)dz) x = u_in using Richardson Iteration.
        Returns u_out = 2x - u_in.
        """
        M_inv_eig, delta_C_dz = self._get_operators(C_slice, dz)

        def apply_M_inv(v):
            return np.divide(
                np.fft.ifft(np.fft.fft(v) * M_inv_eig), (1 + 0.5 * delta_C_dz)
            )

        def apply_P(v):
            V = np.fft.fft(v)
            Mv = np.fft.ifft(V / M_inv_eig)
            return Mv - 0.5 * (delta_C_dz * v)

        b = u_in

        # Initial Guess: 0th order Neumann (x0 = M^{-1} b)
        x = apply_M_inv(b)  # Free space propagation only guess

        for _ in range(self.n_iter):
            # Residual: r = b - Px
            Px = apply_P(x)
            r = b - Px

            # Correction: dx = M^{-1} r
            dx = apply_M_inv(r)
            x = x + dx

        # --- OUTER UPDATE (Cayley Transform) ---
        # u^{n+1} = 2x - u^n
        u_out = 2 * x - u_in

        return u_out

    def _propagate_step(
        self, u_in: np.ndarray, C_slice: np.ndarray, dz: float
    ) -> np.ndarray:
        if not self.use_extrapolation:
            # Single O(dz^2) step
            return self._step_crank_nicolson(u_in, C_slice, dz)
        else:
            # Richardson Extrapolation O(dz^4)
            # 1. Coarse Step (dz) - The hardest to converge
            u_coarse = self._step_crank_nicolson(u_in, C_slice, dz)

            # 2. Fine Steps (dz/2)
            u_fine_mid = self._step_crank_nicolson(u_in, C_slice, dz / 2.0)
            u_fine = self._step_crank_nicolson(u_fine_mid, C_slice, dz / 2.0)

            # 3. Combine: (4 * fine - coarse) / 3
            return (4.0 * u_fine - u_coarse) / 3.0

    def run(
        self, psi_init: Optional[np.ndarray] = None
    ) -> "SpectralCrankNicolsonSolver":
        psi = self.initialize_wavefront(psi_init)

        C = (self.k0 / 1j) * (self.n_map**2 - 1.0)

        C = (self.dz / 2) * (C[:, :-1] + C[:, 1:]) / 2

        if self.store_beam:
            self.beam_history = np.zeros((self.nx, self.nz_steps), dtype=complex)
            self.beam_history[:, 0] = psi

        for z_idx in range(self.nz_steps - 1):
            psi = self._propagate_step(psi, C[:, z_idx], self.dz)

            if self.store_beam:
                self.beam_history[:, z_idx + 1] = psi

        self.psi_final = psi
        return self
