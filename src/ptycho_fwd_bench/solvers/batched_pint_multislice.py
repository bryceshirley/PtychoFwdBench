import logging
from typing import Optional

import numpy as np

from ptycho_fwd_bench.generators import get_probe_field

from .utils import get_spectral_coords


class ParallelMultisliceSolverBatched:
    def __init__(
        self,
        dx: float,
        wavelength: float,
        dz: float,
        nx: int,  # Window Size
        nz_steps: int,  # Depth Steps
        probe_dia: float = 50e-9,  # Default non-zero to avoid RuntimeWarning
        probe_focus: float = 0,
        alpha: float = 1e-6,
    ):
        self.dx = dx
        self.dz = dz
        self.wavelength = wavelength
        self.nx = nx
        self.nz_steps = nz_steps
        self.probe_dia = probe_dia
        self.probe_focus = probe_focus
        self.total_width = self.nx * self.dx

        self.k0 = 2 * np.pi / wavelength
        self.k0sq = self.k0**2
        self.alpha = float(alpha)
        logging.info("Initializing spectral physics engine...")

    def setup_solver(self, n_map: np.ndarray):
        """Sets up the global refractive index map, ensuring it is 2D."""
        # Force squeeze to remove singleton dimensions (e.g., (2048, 1, 256) -> (2048, 256))
        n_map_squeezed = np.squeeze(n_map)

        if n_map_squeezed.ndim != 2:
            raise ValueError(
                f"n_map must be 2D after squeezing. Got shape {n_map.shape}"
            )

        self.n_mean = np.mean(n_map_squeezed)
        self.delta_n_global = n_map_squeezed - self.n_mean

        # Kernel is computed based on window size nx
        self.K_3d = self._get_3d_kernel()

    def _get_3d_kernel(self) -> np.ndarray:
        kx = get_spectral_coords(self.nx, self.dx, "FFT")
        inside = self.k0sq - kx**2
        sqrt_term = np.sqrt(np.clip(inside, 0.0, None))
        lambda_vac = 1j * (sqrt_term - self.k0)
        phi_mean = 1j * self.k0 * (self.n_mean - 1.0)

        self.L_step = np.exp((lambda_vac + phi_mean) * self.dz)[
            np.newaxis, :, np.newaxis
        ]

        L = self.nz_steps
        kz = np.arange(L)
        lam_alpha = (self.alpha ** (1 / L) * np.exp(-2j * np.pi * kz / L))[
            np.newaxis, np.newaxis, :
        ]
        return 1.0 / (1.0 - (lam_alpha * self.L_step))

    def _setup_batch_operators(self, scan_indices: np.ndarray):
        L = self.nz_steps
        Nx = self.nx
        batch_offsets = scan_indices[:, np.newaxis]
        window_indices = np.arange(Nx)
        gather_indices = batch_offsets + window_indices

        # Vectorized gather from global map [cite: 85, 230]
        local_delta_n = self.delta_n_global[gather_indices, :]

        N_phase = 1j * self.k0 * local_delta_n * self.dz / 2
        z_idx = np.arange(L)
        gamma = (self.alpha ** (z_idx / L))[np.newaxis, np.newaxis, :]
        gamma_inv = (self.alpha ** (-z_idx / L))[np.newaxis, np.newaxis, :]

        self._pre_mul = np.exp(-N_phase) * gamma
        self._post_mul = np.exp(-N_phase) * gamma_inv
        self._fwd_N = np.exp(N_phase)
        self._Error_Diag = 1.0 - np.exp(2 * N_phase)

    def _apply_M_inv(self, rhs_batch: np.ndarray) -> np.ndarray:
        v = rhs_batch * self._pre_mul
        v_k = np.fft.fft2(v, axes=(1, 2))
        v_k *= self.K_3d
        v = np.fft.ifft2(v_k, axes=(1, 2))
        return v * self._post_mul

    def _apply_Error(self, u_batch: np.ndarray) -> np.ndarray:
        err = self._Error_Diag * u_batch
        u_last = u_batch[:, :, -1]
        val = u_last * self._fwd_N[:, :, -1]
        val_k = np.fft.fft(val, axis=1)
        val_k *= self.L_step.squeeze(-1)
        val = np.fft.ifft(val_k, axis=1)
        err[:, :, 0] += val * self._fwd_N[:, :, 0] * self.alpha
        return err

    def _apply_M_inv_adjoint(self, rhs_batch: np.ndarray) -> np.ndarray:
        v = rhs_batch * np.conj(self._post_mul)
        v_k = np.fft.fft2(v, axes=(1, 2))
        v_k *= np.conj(self.K_3d)
        v = np.fft.ifft2(v_k, axes=(1, 2))
        return v * np.conj(self._pre_mul)

    def _apply_Error_adjoint(self, v_batch: np.ndarray) -> np.ndarray:
        err = np.conj(self._Error_Diag) * v_batch
        v_first = v_batch[:, :, 0]
        val = v_first * np.conj(self._fwd_N[:, :, 0])
        val_k = np.fft.fft(val, axis=1)
        val_k *= np.conj(self.L_step.squeeze(-1))
        val = np.fft.ifft(val_k, axis=1)
        err[:, :, -1] += val * np.conj(self._fwd_N[:, :, -1]) * self.alpha
        return err

    def compute_gradient(
        self, u_sol: np.ndarray, v_sol: np.ndarray, scan_indices: np.ndarray
    ):
        """Assembles complex gradient components via scatter-add."""
        # 1. Compute the raw overlap term: X = u * v*
        # This contains both Phase info (in Imag part) and Absorp info (in Real part)
        local_overlap = u_sol * np.conj(v_sol)

        batch_offsets = scan_indices[:, np.newaxis]
        window_indices = np.arange(self.nx)
        gather_indices = batch_offsets + window_indices

        # Initialize complex global gradient container
        global_grad_overlap = np.zeros_like(self.delta_n_global, dtype=np.complex128)

        # Accumulate the raw overlap
        np.add.at(global_grad_overlap, (gather_indices, slice(None)), local_overlap)

        return global_grad_overlap

    def run_ptycho_batch(
        self,
        psi_init_batch: np.ndarray,
        scan_indices: np.ndarray,
        n_map: np.ndarray,
        n_iter: int = 5,
        mode: str = "forward",
    ) -> np.ndarray:
        self.setup_solver(n_map)
        B = scan_indices.shape[0]
        self._setup_batch_operators(scan_indices)

        # Initialize Source Term Volume
        S = np.zeros((B, self.nx, self.nz_steps), dtype=np.complex128)

        if mode == "adjoint":
            # In adjoint mode, the "initial" wave is the residual at the DETECTOR (last slice)
            S[:, :, -1] = psi_init_batch
        else:
            # In forward mode, the initial wave is the probe at the SOURCE (first slice)
            S[:, :, 0] = psi_init_batch

        # Apply the Preconditioner (M_inv) to get the initial guess 'b'
        # For adjoint, this effectively "smears" the residual backwards through the volume
        # based on the average bulk refractive index (the preconditioner's job).
        b = self._apply_M_inv_adjoint(S) if mode == "adjoint" else self._apply_M_inv(S)

        u_sol = b.copy()

        # Richardson Iteration / Iterative Refinement
        for _ in range(n_iter):
            if mode == "adjoint":
                # Note: ensure _apply_Error_adjoint correctly handles the full volume
                A_u = u_sol + self._apply_M_inv_adjoint(
                    self._apply_Error_adjoint(u_sol)
                )
            else:
                A_u = u_sol + self._apply_M_inv(self._apply_Error(u_sol))

            u_sol += b - A_u

        return u_sol

    def initialize_wavefront(self, psi_init: Optional[np.ndarray]) -> np.ndarray:
        if psi_init is not None:
            return psi_init.astype(complex)
        x_coords = np.arange(self.nx) * self.dx
        psi = get_probe_field(
            x_coords,
            self.total_width / 2.0,
            self.probe_dia,
            self.probe_focus,
            self.wavelength,
        )
        return psi.astype(complex)
