from typing import Optional

# Use cupy if available, otherwise fallback to numpy
try:
    import cupy as xp
except ImportError:
    import numpy as xp

from .base import OpticalWaveSolver
from .utils import apply_spectral_kernel, get_prop_kernel_perp, get_spectral_coords


class MultisliceSolver(OpticalWaveSolver):
    """
    Multislice beam propagation solver using spectral methods.

    Parameters
    ----------
    n_map : xp.ndarray
        Complex refractive index map of shape (nx, nz_steps).
    dx : float
        Spatial sampling interval in x (um).
    dz : float
        Slice thickness in z (um).
    wavelength : float
        Wavelength of the optical wave (um).
    probe_dia : float
        Diameter of the probe beam (um).
    probe_focus : float
        Focal distance of the probe beam (um).
    symmetric : bool, optional
        If True, uses symmetric propagation (half-step before and after phase). Defaults to True.
    transform_type : str, optional
        Type of spectral transform to use ("DST" or "FFT"). Defaults to "DST".
    store_beam : bool, optional
        If True, stores the wavefield at each slice. Defaults to False.
    """

    def __init__(
        self,
        n_map: xp.ndarray,
        dx: float,
        wavelength: float,
        probe_dia: float,
        probe_focus: float,
        dz: float,
        symmetric: bool = True,
        transform_type: str = "FFT",
        store_beam: bool = False,
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.symmetric = symmetric
        self.transform_type = transform_type
        self._perp_kernel = self._get_propagation_kernel(self.dz)
        if self.symmetric:
            self._perp_kernel_half = self._get_propagation_kernel(self.dz / 2.0)

    def _get_propagation_kernel(self, dz: float) -> xp.ndarray:
        kx = get_spectral_coords(self.nx, self.dx, self.transform_type)

        # Standard vacuum propagator: exp(i * sqrt(k0^2 - kx^2) * z)
        return get_prop_kernel_perp(self.k0, kx, dz)

    def _propagate_and_store(
        self,
        psi: xp.ndarray,
    ) -> xp.ndarray:
        """
        Internal multislice physics loop.
        """

        # Reset history storage if saving is enabled for this pass
        if self.store_beam:
            self.beam_history = xp.zeros((self.nx, self.nz_steps), dtype=complex)
            self.beam_history[:, 0] = psi

        for i in range(self.nz_steps):
            # 1. Half-step Propagate (Vacuum)
            H = self._perp_kernel_half if self.symmetric else self._perp_kernel
            psi = apply_spectral_kernel(psi, H, self.transform_type)

            # 2. Phase (Refraction)
            # Apply phase shift for this slice
            n_slice = self.n_map[:, i]
            psi *= xp.exp(1j * self.k0 * (n_slice - 1.0) * self.dz)

            # 3. Half-step Propagate (Vacuum)
            if self.symmetric:
                psi = apply_spectral_kernel(psi, H, self.transform_type)

            if self.store_beam:
                self.beam_history[:, i] = psi

        self.psi_final = psi

    def run(
        self,
        psi_init: Optional[xp.ndarray] = None,
    ):
        """
        Runs the solver.
        """
        psi_0 = self.initialize_wavefront(psi_init)

        self._propagate_and_store(psi_0)

        return self
