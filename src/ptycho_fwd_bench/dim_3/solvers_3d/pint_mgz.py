import logging
from typing import Optional

import numpy as np
import scipy.ndimage

from .pint_multislice import ParallelMultisliceSolver3D

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndimage

    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False

logger = logging.getLogger(__name__)


class ParallelMultisliceSolver3D_MGZ(ParallelMultisliceSolver3D):
    """
    3D (Y, X, Z) MG Solver - Z-Coarsening Only.
    Uses a coarse Z-grid to generate a high-quality initial guess for the fine solver.
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
        use_gpu: bool = True,
        gpu_id: int = 0,
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
            use_gpu,
            gpu_id,
        )
        self.coarsening_factor = int(coarsening_factor)

    def _restrict_map_factor(self, n_map, factor: int):
        """
        Downsamples ONLY in Z by a specific integer factor.
        (Ny, Nx, L) -> (Ny, Nx, L/factor).
        """
        sh = n_map.shape
        sy = sh[0]
        sx = sh[1]
        sz = sh[2] - (sh[2] % factor)

        # Crop to divisible size
        cropped_map = n_map[:, :, :sz]

        # Reshape to (Ny, Nx, L_new, factor) and mean over the last axis
        return cropped_map.reshape(sy, sx, sz // factor, factor).mean(axis=3)

    def _prolong_wave(self, u_coarse, target_shape: tuple):
        """
        Upsamples Wavefunction ONLY in Z using Polar Interpolation.
        (Ny, Nx, Nz_coarse) -> (Ny, Nx, Nz_fine)
        Handles both CuPy (GPU) and NumPy (CPU) arrays.
        """
        ny, nx, nz_fine = target_shape
        _, _, nz_coarse = u_coarse.shape

        # Calculate zoom factors: 1.0 for Y, X, Ratio for Z
        zoom_fac = (1.0, 1.0, nz_fine / nz_coarse)

        if self.use_gpu:
            # --- GPU Interpolation using CuPy ---
            amp = self.xp.abs(u_coarse)
            phase = self.xp.angle(u_coarse)

            # Unwrap phase along Z
            phase_unwrapped = self.xp.unwrap(phase, axis=2)

            # CuPy doesn't have a direct `zoom` equivalent, so we build coordinates for map_coordinates
            y_coords = self.xp.arange(ny)[:, None, None]
            x_coords = self.xp.arange(nx)[None, :, None]

            # Scale Z coordinates to map from Fine grid to Coarse grid
            z_coords = self.xp.linspace(0, nz_coarse - 1, nz_fine)[None, None, :]

            # Broadcast to full 3D coordinate grids
            Y, X, Z = self.xp.broadcast_arrays(y_coords, x_coords, z_coords)
            coords = self.xp.stack([Y, X, Z])

            # Interpolate Amplitude and Phase
            amp_fine = cpx_ndimage.map_coordinates(amp, coords, order=1, mode="nearest")
            phase_fine = cpx_ndimage.map_coordinates(
                phase_unwrapped, coords, order=1, mode="nearest"
            )

        else:
            # --- CPU Interpolation using SciPy ---
            amp = np.abs(u_coarse)
            phase = np.angle(u_coarse)
            phase_unwrapped = np.unwrap(phase, axis=2)

            amp_fine = scipy.ndimage.zoom(amp, zoom_fac, order=1, grid_mode=False)
            phase_fine = scipy.ndimage.zoom(
                phase_unwrapped, zoom_fac, order=1, grid_mode=False
            )

        # Recombine into a complex wavefield
        u_fine = amp_fine * self.xp.exp(1j * phase_fine)
        return u_fine.astype(self.complex_t)

    def _solve_pass_MG(self, psi_0: np.ndarray) -> np.ndarray:
        """
        Two-Level Cascaded Multigrid Driver (Z Coarsening).
        """
        # Base case: If no coarsening is requested, just run standard solver
        if self.coarsening_factor <= 1:
            return self._solve_pass(psi_0)

        # ==========================================
        # LEVEL 0: COARSE GRID
        # ==========================================
        logger.info(
            f"FMG Level 0 ('Coarse'): Restricting Z by factor {self.coarsening_factor}..."
        )

        with self._device_context():
            coarse_map = self._restrict_map_factor(self.n_map, self.coarsening_factor)

        coarse_dz = self.dz * float(self.coarsening_factor)

        ny_lvl, nx_lvl, nz_lvl = coarse_map.shape
        logger.info(
            f"FMG Level 0 ('Coarse'): Grid=({ny_lvl}x{nx_lvl}x{nz_lvl}), dz={coarse_dz:.2f}nm"
        )

        # Coarse Grid: Create a temporary solver instance
        # We pass the GPU map directly to avoid redundant host/device transfers
        coarse_solver = type(self)(
            n_map=coarse_map
            if not self.use_gpu
            else cp.asnumpy(coarse_map),  # __init__ expects numpy array to initialize
            dx=self.dx,
            wavelength=self.wavelength,
            dz=coarse_dz,
            alpha=self.alpha,
            n_iter=self.n_iter,
            woodbury=self.woodbury,
            use_gpu=self.use_gpu,
            gpu_id=self.gpu_id,
            coarsening_factor=1,  # No further FMG recursion
        )

        # Override the n_map if it was on GPU to prevent double-transfer
        if self.use_gpu:
            coarse_solver.n_map = coarse_map

        # Solve Coarse
        with self._device_context():
            u_prev = coarse_solver._solve_pass(psi_0)

        # ==========================================
        # LEVEL 1: FINE GRID
        # ==========================================
        logger.info(
            f"FMG Level 1 ('Fine'): Grid=({self.ny}x{self.nx}x{self.nz_steps}), dz={self.dz:.2f}nm"
        )

        with self._device_context():
            # Prolongate (Upsample Z)
            u_guess = self._prolong_wave(
                u_prev, target_shape=(self.ny, self.nx, self.nz_steps)
            )

            # Solve Fine using the prolonged wave as the initial guess
            u_prev = self._solve_pass(psi_0, u_guess=u_guess)

        return u_prev

    def _propagate_and_store(
        self, psi_0: np.ndarray, u_guess: Optional[np.ndarray] = None
    ):
        """Wrapper to run the FMG solver and store history if enabled."""
        with self._device_context():
            u_sol = self._solve_pass_MG(psi_0)

            if self.store_beam:
                self.beam_history = cp.asnumpy(u_sol) if self.use_gpu else u_sol

            self.psi_final = (
                cp.asnumpy(u_sol[:, :, -1]) if self.use_gpu else u_sol[:, :, -1]
            )
