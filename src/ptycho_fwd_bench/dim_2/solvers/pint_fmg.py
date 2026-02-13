import logging

import numpy as np
import scipy.ndimage

from .pint_multislice import ParallelMultisliceSolver


class ParallelMultisliceSolver_FMG(ParallelMultisliceSolver):
    """
    2D (X, Z) Full Multigrid Solver.
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
        """Downsamples (Nx, Nz) -> (Nx/n, Nz/n)."""
        sh = n_map.shape
        sx = sh[0] - (sh[0] % coarsening_factor)
        sz = sh[1] - (sh[1] % coarsening_factor)
        n_map = n_map[:sx, :sz]
        return n_map.reshape(
            sx // coarsening_factor,
            coarsening_factor,
            sz // coarsening_factor,
            coarsening_factor,
        ).mean(axis=(1, 3))

    def _prolong_wave(self, u_coarse: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Upsamples Wavefunction in X AND Z using Polar Interpolation.
        (Nx_coarse, Nz_coarse) -> (Nx_fine, Nz_fine)
        """
        zoom_fac = (
            target_shape[0] / u_coarse.shape[0],  # Scale X
            target_shape[1] / u_coarse.shape[1],  # Scale Z
        )

        # Extract Amplitude and Phase
        amp = np.abs(u_coarse)
        phase = np.angle(u_coarse)

        # Unwrap phase along Z, then along X to prevent 2D interpolation artifacts
        phase_unwrapped = np.unwrap(phase, axis=1)
        phase_unwrapped = np.unwrap(phase_unwrapped, axis=0)

        # Interpolate Amplitude and Unwrapped Phase separately
        amp_fine = scipy.ndimage.zoom(amp, zoom_fac, order=1, grid_mode=False)
        phase_fine = scipy.ndimage.zoom(
            phase_unwrapped, zoom_fac, order=1, grid_mode=False
        )

        # Recombine
        u_fine = amp_fine * np.exp(1j * phase_fine)
        return u_fine.astype(np.complex64)

    def _solve_pass_MG(self, psi_0: np.ndarray) -> np.ndarray:
        """
        Two-Level Cascaded Multigrid Driver (X and Z Coarsening).
        """
        # Base case: If no coarsening is requested, just run standard solver
        if self.coarsening_factor <= 1:
            return self._solve_pass(psi_0)

        # ==========================================
        # LEVEL 0: COARSE GRID
        # ==========================================
        logging.info(
            f"FMG Level 0 ('Coarse'): Restricting X and Z by factor {self.coarsening_factor}..."
        )

        # Restrict map in both X and Z
        coarse_map = self._restrict_map_factor(self.n_map, self.coarsening_factor)
        coarse_dz = self.dz * float(self.coarsening_factor)
        coarse_dx = self.dx * float(self.coarsening_factor)

        nx_lvl, nz_lvl = coarse_map.shape
        logging.info(f"FMG Level 0 ('Coarse'): {nx_lvl}x{nz_lvl}")

        # Decimate the initial wave in X so its shape matches the coarse grid
        psi_coarse = psi_0[:: self.coarsening_factor]
        if len(psi_coarse) > nx_lvl:
            psi_coarse = psi_coarse[:nx_lvl]

        coarse_solver = type(self)(
            n_map=coarse_map,
            dx=coarse_dx,  # Passed adjusted dx
            wavelength=self.wavelength,
            dz=coarse_dz,  # Passed adjusted dz
            alpha=self.alpha,
            n_iter=self.n_iter,
            woodbury=self.woodbury,
            coarsening_factor=1,
        )

        # Solve Coarse
        u_prev = coarse_solver._solve_pass(psi_coarse)

        # ==========================================
        # LEVEL 1: FINE GRID
        # ==========================================
        logging.info(f"FMG Level 1 ('Fine'): {self.nx}x{self.nz_steps}")

        # Prolongate (Upsamples both X and Z)
        u_guess = self._prolong_wave(u_prev, target_shape=(self.nx, self.nz_steps))

        # Solve Fine
        u_fine = self._solve_pass(psi_0, u_guess=u_guess)
        return u_fine

    def _propagate_and_store(self, psi_0: np.ndarray):
        """Wrapper to run the FMG solver and store history if enabled."""
        u_sol = self._solve_pass_MG(psi_0)

        if self.store_beam:
            self.beam_history = u_sol

        self.psi_final = u_sol[:, -1]
