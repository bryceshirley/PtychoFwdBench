import logging
from typing import Dict, Optional, Tuple

import numba as nb
import numpy as np
import pywt
from scipy import sparse

from .base import OpticalWaveSolver3D

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# NUMBA JIT KERNELS FOR WAVELETS
# ---------------------------------------------------------
@nb.njit(parallel=True, fastmath=True)
def _numba_wavelet_sweep_x(psi_w, thresh_sq, data, indices, indptr):
    """Parallelized CSR Matrix Multiply for the X-axis (Rows)."""
    ny, n_coeffs = psi_w.shape
    out = np.empty((ny, n_coeffs), dtype=np.complex128)

    # prange distributes the rows perfectly across all cores
    for y in nb.prange(ny):
        # Thread-local buffer (stays in L1 cache)
        row_in = np.empty(n_coeffs, dtype=np.complex128)

        # Inline thresholding
        for i in range(n_coeffs):
            v = psi_w[y, i]
            if (v.real * v.real + v.imag * v.imag) < thresh_sq:
                row_in[i] = 0.0j
            else:
                row_in[i] = v

        # Inline CSR Matrix-Vector Multiply
        for row in range(n_coeffs):
            val = 0.0j
            for p in range(indptr[row], indptr[row + 1]):
                val += data[p] * row_in[indices[p]]
            out[y, row] = val

    return out


@nb.njit(parallel=True, fastmath=True)
def _numba_wavelet_sweep_y(psi_w, thresh_sq, data, indices, indptr):
    """Parallelized CSR Matrix Multiply for the Y-axis (Columns)."""
    n_coeffs, nx = psi_w.shape
    out = np.empty((n_coeffs, nx), dtype=np.complex128)

    for x in nb.prange(nx):
        # Extract column to local buffer to fix strided memory access
        col_in = np.empty(n_coeffs, dtype=np.complex128)
        col_out = np.empty(n_coeffs, dtype=np.complex128)

        for i in range(n_coeffs):
            v = psi_w[i, x]
            if (v.real * v.real + v.imag * v.imag) < thresh_sq:
                col_in[i] = 0.0j
            else:
                col_in[i] = v

        # Inline CSR Matrix-Vector Multiply
        for row in range(n_coeffs):
            val = 0.0j
            for p in range(indptr[row], indptr[row + 1]):
                val += data[p] * col_in[indices[p]]
            col_out[row] = val

        # Write contiguous block back to strided memory
        for i in range(n_coeffs):
            out[i, x] = col_out[i]

    return out


# ---------------------------------------------------------
# MAIN SOLVER CLASS
# ---------------------------------------------------------
class WaveletMultisliceSolver3D(OpticalWaveSolver3D):
    """
    3D Multislice beam propagation solver using the Split-Step Wavelet (SSW) method.
    Uses Numba @njit(parallel=True) to distribute CSR matrix sweeps across CPU cores.
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
        wv_family: str = "sym6",
        wv_level: int = 3,
        v_s: float = 1e-4,
        store_beam: bool = False,
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)

        self.symmetric = symmetric
        self.wavelet_name = wv_family
        self.mode = "per"

        # Square threshold for fast complex magnitude checking in Numba
        self.threshold = float(v_s)
        self.thresh_sq = self.threshold**2

        self.wavelet = pywt.Wavelet(self.wavelet_name)
        max_allowed = pywt.dwt_max_level(min(self.nx, self.ny), self.wavelet.dec_len)
        self.wv_level = min(int(wv_level), max_allowed)

        # Pre-calculate Phase Map
        self.refraction_map = np.exp(1j * self.k0 * (self.n_map - 1.0) * self.dz)

        # Pre-calculate Split Indices for fast un-flattening
        dummy_x = pywt.wavedec(
            np.zeros(self.nx), self.wavelet, level=self.wv_level, mode=self.mode
        )
        self.split_idx_x = np.cumsum([len(c) for c in dummy_x])[:-1]

        dummy_y = pywt.wavedec(
            np.zeros(self.ny), self.wavelet, level=self.wv_level, mode=self.mode
        )
        self.split_idx_y = np.cumsum([len(c) for c in dummy_y])[:-1]

        # ---------------------------------------------------------
        # BUILD MATRICES IN INIT (Ensures clean benchmark timings)
        # ---------------------------------------------------------
        P_x = self._build_1d_propagator_matrix(self.nx)
        self.P_data_x, self.P_indices_x, self.P_indptr_x = (
            P_x.data,
            P_x.indices,
            P_x.indptr,
        )

        if self.nx == self.ny:
            self.P_data_y, self.P_indices_y, self.P_indptr_y = (
                self.P_data_x,
                self.P_indices_x,
                self.P_indptr_x,
            )
        else:
            P_y = self._build_1d_propagator_matrix(self.ny)
            self.P_data_y, self.P_indices_y, self.P_indptr_y = (
                P_y.data,
                P_y.indices,
                P_y.indptr,
            )

    def _build_1d_propagator_matrix(self, N: int) -> sparse.csr_matrix:
        step_dist = (self.dz / 2.0) if self.symmetric else self.dz
        cache_key = (N, self.dx, step_dist, self.wavelet_name, self.threshold)

        if cache_key in self._global_cache:
            return self._global_cache[cache_key]

        dummy_coeffs = pywt.wavedec(
            np.zeros(N), self.wavelet, level=self.wv_level, mode=self.mode
        )
        num_coeffs = sum(len(c) for c in dummy_coeffs)
        split_idx = np.cumsum([len(c) for c in dummy_coeffs])[:-1]

        kx = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        inside = self.k0sq - kx**2
        sqrt_term = np.sqrt(np.clip(inside, 0.0, None))
        lambda_vac = 1j * (sqrt_term - self.k0)
        spectral_filter = np.exp(lambda_vac * step_dist)

        I_w = np.eye(num_coeffs, dtype=np.complex128)
        coeffs_batched = np.split(I_w, split_idx, axis=1)
        spatial_batched = pywt.waverec(
            coeffs_batched, self.wavelet, mode=self.mode, axis=1
        )

        prop_f = np.fft.fft(spatial_batched, axis=1) * spectral_filter[np.newaxis, :]
        prop_s = np.fft.ifft(prop_f, axis=1)

        res_coeffs_batched = pywt.wavedec(
            prop_s, self.wavelet, level=self.wv_level, mode=self.mode, axis=1
        )
        dense_matrix = np.concatenate(res_coeffs_batched, axis=1).T

        # Threshold and Sparsify
        mask = np.abs(dense_matrix) > self.threshold
        P_matrix = sparse.csr_matrix(dense_matrix * mask)

        self._global_cache[cache_key] = P_matrix
        return P_matrix

    def _propagate_vacuum_2d(self, psi: np.ndarray) -> np.ndarray:
        """Applies Dimensional Operator Splitting to propagate a 2D transverse plane."""

        # --- 1. Propagate along X (Rows) ---
        coeffs_x = pywt.wavedec(
            psi, self.wavelet, level=self.wv_level, mode=self.mode, axis=1
        )
        psi_w_x = np.concatenate(coeffs_x, axis=1)

        # Numba Parallel Sweep
        psi_w_x = _numba_wavelet_sweep_x(
            psi_w_x, self.thresh_sq, self.P_data_x, self.P_indices_x, self.P_indptr_x
        )

        coeffs_rec_x = np.split(psi_w_x, self.split_idx_x, axis=1)
        psi = pywt.waverec(coeffs_rec_x, self.wavelet, mode=self.mode, axis=1)

        # --- 2. Propagate along Y (Columns) ---
        coeffs_y = pywt.wavedec(
            psi, self.wavelet, level=self.wv_level, mode=self.mode, axis=0
        )
        psi_w_y = np.concatenate(coeffs_y, axis=0)

        # Numba Parallel Sweep
        psi_w_y = _numba_wavelet_sweep_y(
            psi_w_y, self.thresh_sq, self.P_data_y, self.P_indices_y, self.P_indptr_y
        )

        coeffs_rec_y = np.split(psi_w_y, self.split_idx_y, axis=0)
        psi = pywt.waverec(coeffs_rec_y, self.wavelet, mode=self.mode, axis=0)

        return psi

    def run(self, psi_init: Optional[np.ndarray] = None) -> "WaveletMultisliceSolver3D":
        logger.debug("WaveletMultisliceSolver3D initialized: Using CPU")

        psi = self.initialize_wavefront(psi_init)

        if self.store_beam:
            self.beam_history = np.zeros(
                (self.ny, self.nx, self.nz_steps), dtype=np.complex128
            )
            self.beam_history[:, :, 0] = psi

        for i in range(self.nz_steps):
            # 1. Vacuum Propagation
            psi = self._propagate_vacuum_2d(psi)

            # 2. Phase Shift (Refraction)
            psi *= self.refraction_map[:, :, i]

            # 3. Vacuum Propagation
            if self.symmetric:
                psi = self._propagate_vacuum_2d(psi)

            if self.store_beam:
                self.beam_history[:, :, i] = psi

        self.psi_final = psi
        return self
