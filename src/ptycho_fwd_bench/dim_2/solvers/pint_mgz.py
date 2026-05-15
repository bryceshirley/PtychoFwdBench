import logging

import numpy as np
import scipy.ndimage

from .pint_multislice import ParallelMultisliceSolver


class ParallelMultisliceSolver_MGZ(ParallelMultisliceSolver):
    """
    2D (X, Z) MG Solver - Z-Coarsening Only.
    """

    def __init__(
        self,
        n_map: np.ndarray,
        dx: float,
        wavelength: float,
        dz: float,
        probe_dia: float = 0,
        probe_focus: float = 0,
        store_beam: bool = False,
        alpha: float = 1e-6,
        n_iter: int = 1,
        woodbury: bool = True,
        coarsening_factor: int = 8,
    ):
        super().__init__(
            n_map,
            dx,
            wavelength,
            dz,
            probe_dia,
            probe_focus,
            store_beam,
            alpha,
            n_iter,
            woodbury,
        )
        self.coarsening_factor = int(coarsening_factor)

    def _restrict_map_factor(
        self, n_map: np.ndarray, coarsening_factor: int
    ) -> np.ndarray:
        """
        Downsamples ONLY in Z by a specific integer factor.
        (Nx, L) -> (Nx, L/factor).
        """
        sh = n_map.shape
        # Ensure divisible by factor
        sx = sh[0]
        sz = sh[1] - (sh[1] % coarsening_factor)
        n_map = n_map[:, :sz]
        # Reshape to (Nx, L_new, factor) and mean over the last axis
        return n_map.reshape(sx, sz // coarsening_factor, coarsening_factor).mean(
            axis=2
        )

    def _prolong_wave(self, u_coarse: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Upsamples Wavefunction ONLY in Z using Polar Interpolation.
        (Nx, Nz_coarse) -> (Nx, Nz_fine)
        """
        # Calculate zoom factors: 1.0 for X, Ratio for Z
        zoom_fac = (
            1.0,  # X dimension stays 1:1
            target_shape[1] / u_coarse.shape[1],  # Z dimension scales
        )

        # 1. Extract Amplitude and Phase
        amp = np.abs(u_coarse)
        phase = np.angle(u_coarse)

        # 2. Unwrap the phase along the Z-axis to prevent interpolation artifacts at boundaries
        phase_unwrapped = np.unwrap(phase, axis=1)

        # 3. Interpolate Amplitude and Unwrapped Phase separately
        amp_fine = scipy.ndimage.zoom(amp, zoom_fac, order=1, grid_mode=False)
        phase_fine = scipy.ndimage.zoom(
            phase_unwrapped, zoom_fac, order=1, grid_mode=False
        )

        # 4. Recombine into a complex wavefield
        u_fine = amp_fine * np.exp(1j * phase_fine)

        return u_fine.astype(np.complex64)

    def _solve_pass_MG(self, psi_0: np.ndarray) -> np.ndarray:
        """
        Two-Level Cascaded Multigrid Driver (Z Coarsening).
        """
        # Base case: If no coarsening is requested, just run standard solver
        if self.coarsening_factor == 1:
            return self._solve_pass(psi_0)

        # ==========================================
        # LEVEL 0: COARSE GRID
        # ==========================================
        # Restrict map in both X and Z
        coarse_map = self._restrict_map_factor(self.n_map, self.coarsening_factor)
        coarse_dz = self.dz * float(self.coarsening_factor)

        nx_lvl, nz_lvl = coarse_map.shape
        logging.info(f"FMG Level 0 ('Coarse'): {nx_lvl}x{nz_lvl}")

        # Coarse Grid: Heavy smoothing
        coarse_solver = type(self)(
            n_map=coarse_map,
            dx=self.dx,  # Constant dx
            wavelength=self.wavelength,
            dz=coarse_dz,
            alpha=self.alpha,
            n_iter=self.n_iter,
            woodbury=self.woodbury,
            coarsening_factor=1,  # No further FMG recursion
        )

        # Solve Coarse
        u_prev = coarse_solver._solve_pass(psi_0)

        # ==========================================
        # LEVEL 1: FINE GRID
        # ==========================================
        logging.info(f"FMG Level 1 ('Fine'): {self.nx}x{self.nz_steps}")

        # Prolongate (Upsample Z)
        u_guess = self._prolong_wave(u_prev, target_shape=(self.nx, self.nz_steps))

        # Solve Fine
        u_prev = self._solve_pass(psi_0, u_guess=u_guess)
        return u_prev

    def _propagate_and_store(self, psi_0: np.ndarray):
        """Wrapper to run the FMG solver and store history if enabled."""
        u_sol = self._solve_pass_MG(psi_0)

        if self.store_beam:
            self.beam_history = u_sol

        self.psi_final = u_sol[:, -1]
