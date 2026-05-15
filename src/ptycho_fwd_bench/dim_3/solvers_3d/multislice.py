import logging
from typing import Optional

import numpy as np

from ptycho_fwd_bench.dim_2.solvers.utils import get_spectral_coords

from .base import OpticalWaveSolver3D

try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False

logger = logging.getLogger(__name__)


class MultisliceSolver3D(OpticalWaveSolver3D):
    """
    3D Multislice beam propagation solver using spectral methods (Split-Step Fourier).
    Uses CuPy for GPU acceleration if available.

    Parameters
    ----------
    n_map : np.ndarray
        Complex refractive index map of shape (ny, nx, nz_steps).
    dx : float
        Spatial sampling interval in x/y (um). Assumes isotropic transverse grid.
    wavelength : float
        Wavelength of the optical wave (um).
    probe_dia : float
        Diameter of the probe beam (um).
    probe_focus : float
        Focal distance of the probe beam (um).
    dz : float
        Slice thickness in z (um).
    symmetric : bool, optional
        If True, uses symmetric propagation (half-step vacuum, full-step phase, half-step vacuum).
    store_beam : bool, optional
        If True, stores the wavefield at each slice. Defaults to False.
    use_gpu : bool, optional
        Whether to attempt to use CuPy for GPU acceleration.
    gpu_id : int, optional
        Device ID to run the GPU computations on.
    """

    def __init__(
        self,
        n_map: np.ndarray,
        dx: float,
        wavelength: float,
        probe_dia: float,
        probe_focus: float,
        dz: float,
        symmetric: bool = True,
        store_beam: bool = False,
        use_gpu: bool = True,
        gpu_id: int = 0,
        transform_type: str = "FFT",
        mode: str = "spectral",
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.symmetric = symmetric
        self.transform_type = transform_type
        self.mode = mode

        # GPU Setup
        self.use_gpu = use_gpu and HAS_GPU
        self.gpu_id = gpu_id
        self.xp = cp if self.use_gpu else np

        if self.use_gpu:
            logger.debug(
                f"MultisliceSolver3D initialized: Using GPU (ID={self.gpu_id})"
            )
        else:
            logger.debug("MultisliceSolver3D initialized: Using CPU (NumPy)")

        # Precompute Free-Space Propagators
        with self._device_context():
            self._perp_kernel = self._get_propagation_kernel(self.dz)
            if self.symmetric:
                self._perp_kernel_half = self._get_propagation_kernel(self.dz / 2.0)

    def _device_context(self):
        """Context manager to ensure operations happen on the correct GPU."""
        if self.use_gpu:
            return cp.cuda.Device(self.gpu_id)
        return np.errstate(all="ignore")

    def _get_propagation_kernel(self, step_dz: float) -> np.ndarray:
        """
        Calculates the 2D transverse free-space propagator for a given Z-step.
        """
        # Get coordinates on CPU, then transfer to target device
        kx_cpu = get_spectral_coords(self.nx, self.dx, self.transform_type, self.mode)
        ky_cpu = get_spectral_coords(self.ny, self.dx, self.transform_type, self.mode)

        kx = self.xp.asarray(kx_cpu)
        ky = self.xp.asarray(ky_cpu)

        KX, KY = self.xp.meshgrid(kx, ky, indexing="ij")
        K_sq = KX**2 + KY**2
        del KX, KY  # Free memory

        # Convert to complex type BEFORE sqrt to allow imaginary results for negative values
        inside = (self.k0**2 - K_sq).astype(np.complex128)
        sqrt_term = self.xp.sqrt(inside)

        # For evanescent modes (K_sq > k0^2), sqrt_term becomes i*alpha.
        # phase = i*(i*alpha - k0)*dz = -alpha*dz - i*k0*dz
        # exp(phase) will naturally exponentially decay
        phase = 1j * (sqrt_term - self.k0) * step_dz

        return self.xp.exp(phase).astype(np.complex128)

    def _apply_propagation(self, psi, kernel):
        """Helper to apply a propagation kernel in Fourier space."""
        psi_k = self.xp.fft.fft2(psi, axes=(0, 1))
        psi_k *= kernel
        return self.xp.fft.ifft2(psi_k, axes=(0, 1))

    def _propagate_and_store(self, psi: np.ndarray):
        """
        Internal 3D multislice physics loop with optimized Strang splitting.
        Chains P_{1/2} S P_{1/2} into P_{1/2} S P S P ... S P_{1/2}.
        """
        # Ensure array is on correct device
        if self.use_gpu and not isinstance(psi, cp.ndarray):
            psi = cp.asarray(psi)

        # Reset history storage
        if self.store_beam:
            self.beam_history = self.xp.zeros(
                (self.ny, self.nx, self.nz_steps), dtype=np.complex128
            )

        with self._device_context():
            n_map_device = self.xp.asarray(self.n_map)

            # --- PRE-LOOP: Initial Half-Step ---
            # If symmetric, we start with a half-step propagation
            if self.symmetric:
                psi = self._apply_propagation(psi, self._perp_kernel_half)

            for i in range(self.nz_steps):
                # 1. Phase (Refraction) - The 'S' operator
                n_slice = n_map_device[:, :, i]
                psi *= self.xp.exp(1j * self.k0 * (n_slice - 1.0) * self.dz)

                if self.store_beam:
                    self.beam_history[:, :, i] = psi

                # 2. Propagation - The 'P' operator
                if i < self.nz_steps - 1:
                    # Intermediate steps:
                    # If symmetric: Half-step out + Half-step in = Full-step (P)
                    # If not symmetric: Standard full-step (P)
                    psi = self._apply_propagation(psi, self._perp_kernel)
                else:
                    # --- POST-LOOP: Final Step ---
                    # Last slice: If symmetric, apply final half-step (P_{1/2})
                    if self.symmetric:
                        psi = self._apply_propagation(psi, self._perp_kernel_half)
                    # If not symmetric, no final propagation is needed as
                    # it was P -> S in each iteration.

        # Transfer final exit wave back
        self.psi_final = cp.asnumpy(psi) if self.use_gpu else psi

        if self.store_beam and self.use_gpu:
            self.beam_history = cp.asnumpy(self.beam_history)

    def run(self, psi_init: Optional[np.ndarray] = None):
        """
        Runs the 3D solver.
        """
        psi_0 = self.initialize_wavefront(psi_init)

        self._propagate_and_store(psi_0)

        return self
