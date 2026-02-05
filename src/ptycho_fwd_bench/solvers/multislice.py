from typing import Optional

import numpy as np

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
        transform_type: str = "FFT",
        store_beam: bool = False,
        use_richardson: bool = False,
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.symmetric = symmetric
        self.transform_type = transform_type
        self._kernel_cache = {}
        self.use_richardson = use_richardson

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

    def _execute_propagation(
        self,
        psi: np.ndarray,
        n_map_local: np.ndarray,
        dz_local: float,
        save_history: bool = False,
    ) -> np.ndarray:
        """
        Internal physics loop.
        Separated to allow running Coarse/Fine passes independently.
        """
        nz_local = n_map_local.shape[1]

        # Reset history storage if saving is enabled for this pass
        if save_history and self.store_beam:
            self.beam_history = np.zeros((self.nx, nz_local), dtype=complex)
            self.beam_history[:, 0] = psi

        for i in range(nz_local):
            step_dist = (dz_local / 2.0) if self.symmetric else dz_local

            # 1. Half-step Propagate (Vacuum)
            H_half = self._get_propagation_kernel(step_dist)
            psi = apply_spectral_kernel(psi, H_half, self.transform_type)

            # 2. Phase (Refraction)
            # Apply phase shift for this slice
            n_slice = n_map_local[:, i]
            psi *= np.exp(1j * self.k0 * (n_slice - 1.0) * dz_local)

            # 3. Half-step Propagate (Vacuum)
            if self.symmetric:
                psi = apply_spectral_kernel(psi, H_half, self.transform_type)

            if save_history and self.store_beam:
                self.beam_history[:, i] = psi

        return psi

    def run(self, psi_init: Optional[np.ndarray] = None) -> "MultisliceSolver":
        """
        Runs the solver.

        Parameters
        ----------
        psi_init : np.ndarray, optional
            Initial wavefront.
        """
        # Base class handles creation or validation
        psi_0 = self.initialize_wavefront(psi_init)

        if not self.use_richardson:
            # Standard Run (2nd Order)
            self.psi_final = self._execute_propagation(
                psi_0, self.n_map, self.dz, save_history=True
            )
        else:
            # Richardson Extrapolation Run (4th Order)

            # 1. Coarse Simulation (Step = dz)
            # We disable history saving here to save memory/avoid conflicts
            psi_coarse = self._execute_propagation(
                psi_0.copy(), self.n_map, self.dz, save_history=False
            )

            # 2. Fine Simulation (Step = dz/2)
            # Upsample the map: Split every slice into two identical thinner slices
            n_map_fine = np.repeat(self.n_map, 2, axis=1)
            psi_fine = self._execute_propagation(
                psi_0.copy(), n_map_fine, self.dz / 2.0, save_history=False
            )

            # 3. Extrapolate
            # For 2nd-order methods (symmetric split), the error scales as dz^2.
            # Richardson Formula: (4 * Fine - Coarse) / 3
            self.psi_final = (4.0 * psi_fine - psi_coarse) / 3.0

            # Note: beam_history is not populated in Richardson mode
            # as the z-axis grids of the two runs do not match.

        return self
