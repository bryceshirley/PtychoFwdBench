from typing import Dict, Optional, Tuple

import numpy as np
import pywt
from scipy import sparse

from .base import OpticalWaveSolver


class WaveletMultisliceSolver(OpticalWaveSolver):
    """
    Multislice beam propagation solver using SSW method.
    Includes Global Cache and Sparsity Diagnostics.
    """

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
        wavelet_name: str = "db4",
        compression_threshold: float = 1e-5,
        store_beam: bool = False,
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.symmetric = symmetric
        self.wavelet_name = wavelet_name
        self.threshold = float(compression_threshold)
        self.P_matrix: Optional[sparse.csr_matrix] = None

        # Init Wavelets
        self.wavelet = pywt.Wavelet(self.wavelet_name)
        self.max_level = pywt.dwt_max_level(self.nx, self.wavelet.dec_len)
        self.kx = np.fft.fftfreq(self.nx, d=self.dx) * 2 * np.pi

    def build_propagator_matrix(self) -> None:
        """Builds or retrieves matrix with diagnostics."""
        step_dist = (self.dz / 2.0) if self.symmetric else self.dz
        cache_key = (self.nx, self.dx, step_dist, self.wavelet_name, self.threshold)

        if cache_key in self._global_cache:
            self.P_matrix = self._global_cache[cache_key]
            return

        # Build it
        dummy_coeffs = pywt.wavedec(
            np.zeros(self.nx), self.wavelet, level=self.max_level
        )
        dummy_arr, slices = pywt.coeffs_to_array(dummy_coeffs)
        num_coeffs = dummy_arr.shape[0]

        inside = self.k0sq - self.kx**2
        sqrt_term = np.sqrt(np.clip(inside, 0.0, None))
        lambda_vac = 1j * (sqrt_term - self.k0)
        spectral_filter = np.exp(lambda_vac * step_dist)

        dense_matrix = np.zeros((num_coeffs, num_coeffs), dtype=np.complex128)

        for i in range(num_coeffs):
            unit_vec = np.zeros(num_coeffs)
            unit_vec[i] = 1.0
            c_struct = pywt.array_to_coeffs(unit_vec, slices, output_format="wavedec")
            spatial = pywt.waverec(c_struct, self.wavelet)

            if len(spatial) < self.nx:
                spatial = np.pad(spatial, (0, self.nx - len(spatial)))
            elif len(spatial) > self.nx:
                spatial = spatial[: self.nx]

            prop_f = np.fft.fft(spatial) * spectral_filter
            prop_s = np.fft.ifft(prop_f)

            res_coeffs = pywt.wavedec(prop_s, self.wavelet, level=self.max_level)
            res_flat, _ = pywt.coeffs_to_array(res_coeffs)
            dense_matrix[:, i] = res_flat

        # Compress
        mask = np.abs(dense_matrix) > self.threshold
        self.P_matrix = sparse.csr_matrix(dense_matrix * mask)

        # Save to Cache
        self._global_cache[cache_key] = self.P_matrix

        # --- DIAGNOSTICS ---
        nz_per_row = self.P_matrix.nnz / num_coeffs
        if nz_per_row > 20:
            pass

    def run(self, psi_init: Optional[np.ndarray] = None) -> "WaveletMultisliceSolver":
        if self.P_matrix is None:
            self.build_propagator_matrix()  # Fallback

        psi = self.initialize_wavefront(psi_init)

        if self.store_beam:
            self.beam_history = np.zeros((self.nx, self.nz_steps), dtype=complex)
            self.beam_history[:, 0] = psi

        # Pre-allocate for speed
        coeffs_template = pywt.wavedec(psi, self.wavelet, level=self.max_level)
        _, slices = pywt.coeffs_to_array(coeffs_template)

        for i in range(self.nz_steps):
            # 1. Diffraction
            coeffs = pywt.wavedec(psi, self.wavelet, level=self.max_level)
            psi_w, _ = pywt.coeffs_to_array(coeffs)
            psi_w = self.P_matrix.dot(psi_w)
            coeffs_rec = pywt.array_to_coeffs(psi_w, slices, output_format="wavedec")
            psi = pywt.waverec(coeffs_rec, self.wavelet)

            if len(psi) > self.nx:
                psi = psi[: self.nx]
            elif len(psi) < self.nx:
                psi = np.pad(psi, (0, self.nx - len(psi)))

            # 2. Refraction
            n_slice = self.n_map[:, i]
            psi *= np.exp(1j * self.k0 * (n_slice - 1.0) * self.dz)

            # 3. Symmetric
            if self.symmetric:
                coeffs = pywt.wavedec(psi, self.wavelet, level=self.max_level)
                psi_w, _ = pywt.coeffs_to_array(coeffs)
                psi_w = self.P_matrix.dot(psi_w)
                coeffs_rec = pywt.array_to_coeffs(
                    psi_w, slices, output_format="wavedec"
                )
                psi = pywt.waverec(coeffs_rec, self.wavelet)

                if len(psi) > self.nx:
                    psi = psi[: self.nx]
                elif len(psi) < self.nx:
                    psi = np.pad(psi, (0, self.nx - len(psi)))

            if self.store_beam:
                self.beam_history[:, i] = psi

        self.psi_final = psi
        return self
