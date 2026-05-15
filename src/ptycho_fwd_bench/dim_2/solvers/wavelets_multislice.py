from typing import Dict, Optional, Tuple

import numpy as np
import pywt
from scipy import sparse

from .base import OpticalWaveSolver


class WaveletMultisliceSolver(OpticalWaveSolver):
    """
    Multislice beam propagation solver using the Split-Step Wavelet (SSW) method.
    Uses an exact Sparse Transfer Matrix for lightning-fast 1D CPU propagation.
    """

    # Global cache prevents rebuilding the matrix for identical grid setups
    _global_cache: Dict[Tuple, sparse.csr_matrix] = {}

    def __init__(
        self,
        n_map: np.ndarray,
        dx: float,
        wavelength: float,
        probe_dia: float,
        probe_focus: float,
        dz: float,
        symmetric: bool = True,
        wv_family: str = "sym6",
        wv_level: int = 3,
        v_s: float = 1e-4,
        store_beam: bool = False,
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)

        self.symmetric = symmetric
        self.wavelet_name = wv_family
        self.mode = "per"  # Periodic boundaries to perfectly match FFT

        # FORCE CAST TO FLOAT to fix numpy ufunc string comparison error from YAML
        self.threshold = float(v_s)
        self.P_matrix: Optional[sparse.csr_matrix] = None

        # Init Wavelets
        self.wavelet = pywt.Wavelet(self.wavelet_name)

        # Safety check: Cap the requested level to the maximum mathematically allowed
        max_allowed = pywt.dwt_max_level(self.nx, self.wavelet.dec_len)
        self.wv_level = min(int(wv_level), max_allowed)

        # Pre-compute spectral frequencies for the vacuum kernel
        self.kx = np.fft.fftfreq(self.nx, d=self.dx) * 2 * np.pi

    def build_propagator_matrix(self) -> None:
        """
        Builds the Sparse Transfer Matrix by passing unit impulses through the
        exact FFT vacuum propagator and transforming them into the wavelet domain.
        """
        step_dist = (self.dz / 2.0) if self.symmetric else self.dz
        cache_key = (self.nx, self.dx, step_dist, self.wavelet_name, self.threshold)

        # Use cached matrix if available to save initialization time
        if cache_key in self._global_cache:
            self.P_matrix = self._global_cache[cache_key]
            return

        # Figure out the exact size of the flattened wavelet array
        dummy_coeffs = pywt.wavedec(
            np.zeros(self.nx), self.wavelet, level=self.wv_level, mode=self.mode
        )
        dummy_arr, slices = pywt.coeffs_to_array(dummy_coeffs)
        num_coeffs = dummy_arr.shape[0]

        # Exact optical vacuum propagator kernel
        inside = self.k0sq - self.kx**2
        sqrt_term = np.sqrt(np.clip(inside, 0.0, None))
        lambda_vac = 1j * (sqrt_term - self.k0)
        spectral_filter = np.exp(lambda_vac * step_dist)

        # Build dense transfer matrix
        dense_matrix = np.zeros((num_coeffs, num_coeffs), dtype=np.complex128)

        for i in range(num_coeffs):
            unit_vec = np.zeros(num_coeffs)
            unit_vec[i] = 1.0

            # Reconstruct spatial impulse from wavelet impulse
            c_struct = pywt.array_to_coeffs(unit_vec, slices, output_format="wavedec")
            spatial = pywt.waverec(c_struct, self.wavelet, mode=self.mode)

            # Propagate exactly in vacuum
            prop_f = np.fft.fft(spatial) * spectral_filter
            prop_s = np.fft.ifft(prop_f)

            # Transform back to wavelet domain and store column
            res_coeffs = pywt.wavedec(
                prop_s, self.wavelet, level=self.wv_level, mode=self.mode
            )
            res_flat, _ = pywt.coeffs_to_array(res_coeffs)
            dense_matrix[:, i] = res_flat

        # Threshold and Sparsify
        mask = np.abs(dense_matrix) > self.threshold
        self.P_matrix = sparse.csr_matrix(dense_matrix * mask)
        self._global_cache[cache_key] = self.P_matrix

    def run(self, psi_init: Optional[np.ndarray] = None) -> "WaveletMultisliceSolver":
        """Runs the multislice solver."""
        if self.P_matrix is None:
            self.build_propagator_matrix()

        psi = self.initialize_wavefront(psi_init)

        if self.store_beam:
            self.beam_history = np.zeros((self.nx, self.nz_steps), dtype=complex)
            self.beam_history[:, 0] = psi

        # Pre-allocate slice structure for maximum speed in the loop
        coeffs_template = pywt.wavedec(
            psi, self.wavelet, level=self.wv_level, mode=self.mode
        )
        _, slices = pywt.coeffs_to_array(coeffs_template)

        for i in range(self.nz_steps):
            # 1. Diffraction (Half-step if symmetric)
            coeffs = pywt.wavedec(
                psi, self.wavelet, level=self.wv_level, mode=self.mode
            )
            psi_w, _ = pywt.coeffs_to_array(coeffs)

            # State Sparsification: zero out insignificant coefficients
            psi_w[np.abs(psi_w) < self.threshold] = 0.0j

            # Sparse Matrix Propagator
            psi_w = self.P_matrix.dot(psi_w)

            coeffs_rec = pywt.array_to_coeffs(psi_w, slices, output_format="wavedec")
            psi = pywt.waverec(coeffs_rec, self.wavelet, mode=self.mode)

            # 2. Phase Shift (Refraction)
            n_slice = self.n_map[:, i]
            psi *= np.exp(1j * self.k0 * (n_slice - 1.0) * self.dz)

            # 3. Diffraction (Second Half-step if symmetric)
            if self.symmetric:
                coeffs = pywt.wavedec(
                    psi, self.wavelet, level=self.wv_level, mode=self.mode
                )
                psi_w, _ = pywt.coeffs_to_array(coeffs)

                psi_w[np.abs(psi_w) < self.threshold] = 0.0j

                psi_w = self.P_matrix.dot(psi_w)

                coeffs_rec = pywt.array_to_coeffs(
                    psi_w, slices, output_format="wavedec"
                )
                psi = pywt.waverec(coeffs_rec, self.wavelet, mode=self.mode)

            if self.store_beam:
                self.beam_history[:, i] = psi

        self.psi_final = psi
        return self
