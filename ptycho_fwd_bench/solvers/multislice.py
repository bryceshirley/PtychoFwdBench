import numpy as np
from typing import Optional
from .base import OpticalWaveSolver
from .utils import apply_spectral_kernel, get_spectral_coords


class MultisliceSolver(OpticalWaveSolver):
    """
    Multislice beam propagation solver using spectral methods.

    Parameters
    ----------
    n_map : np.ndarray
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
        n_map: np.ndarray,
        dx: float,
        wavelength: float,
        probe_dia: float,
        probe_focus: float,
        dz: float,
        symmetric: bool = True,
        transform_type: str = "DST",
        store_beam: bool = False,
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.symmetric = symmetric
        self.transform_type = transform_type
        self._kernel_cache = {}

    def _get_propagation_kernel(self, dz: float) -> np.ndarray:
        key = (self.transform_type, dz)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        kx = get_spectral_coords(self.nx, self.dx, self.transform_type)

        # Standard vacuum propagator: exp(i * sqrt(k0^2 - kx^2) * z)
        inside = self.k0sq - kx**2
        sqrt_term = np.sqrt(np.clip(inside, 0.0, None))
        Lambda = 1j * (sqrt_term - self.k0)
        H = np.exp(Lambda * dz).astype(np.complex128)

        self._kernel_cache[key] = H
        return H

    def run(self, psi_init: Optional[np.ndarray] = None) -> "MultisliceSolver":
        # Base class handles creation or validation
        psi = self.initialize_wavefront(psi_init)

        if self.store_beam:
            self.beam_history = np.zeros((self.nx, self.nz_steps), dtype=complex)
            self.beam_history[:, 0] = psi

        # Propagation Loop
        for i in range(self.nz_steps):
            step_dist = (self.dz / 2.0) if self.symmetric else self.dz

            # Half-step
            H_half = self._get_propagation_kernel(step_dist)
            psi = apply_spectral_kernel(psi, H_half, self.transform_type)

            # Phase
            n_slice = self.n_map[:, i]
            psi *= np.exp(1j * self.k0 * (n_slice - 1.0) * self.dz)

            # Half-step
            if self.symmetric:
                psi = apply_spectral_kernel(psi, H_half, self.transform_type)

            if self.store_beam:
                self.beam_history[:, i] = psi

        self.psi_final = psi
        return self
