import logging
from typing import Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from ptycho_fwd_bench.dim_2.generators import get_probe_field

from .utils import get_spectral_coords


class ParallelMultisliceSolverBatched:
    """
    Batched Parallel-in-Time Solver for 2D Ptychography (x, z).
    Solves for B probes simultaneously.
    """

    def __init__(
        self,
        dx: float,
        wavelength: float,
        dz: float,
        nx: int,  # Window Size
        nz_steps: int,  # Depth Steps
        probe_dia: float,
        probe_focus: float = 0,
        alpha: float = 1e-6,
        solver_type: str = "richardson",
        n_iter: int = 1,
        **kwargs,
    ):
        self.dx = dx
        self.dz = dz
        self.wavelength = wavelength
        self.nx = nx
        self.nz_steps = nz_steps
        self.probe_dia = probe_dia
        self.probe_focus = probe_focus
        self.total_width = self.nx * self.dx
        self.solver_type = solver_type
        self.n_iter = n_iter
        self.k0 = 2 * np.pi / wavelength
        self.k0sq = self.k0**2
        self.alpha = float(alpha)
        logging.info("Initializing spectral physics engine...")

    def setup_solver(self, n_map: np.ndarray):
        """Sets up the global refractive index map, ensuring it is 2D."""
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
        """
        Computes the spectral propagator K.
        Shape: (Nx, Nz) broadcastable to (Batch, Nx, Nz).
        """
        kx = get_spectral_coords(self.nx, self.dx, "FFT")

        # Transverse Propagator P (Vacuum)
        inside = self.k0sq - kx**2
        sqrt_term = np.sqrt(np.clip(inside, 0.0, None))

        # lambda_vac: vacuum phase advance per dz
        lambda_vac = 1j * (sqrt_term - self.k0)

        # phi_mean: mean potential phase advance per dz
        phi_mean = 1j * self.k0 * (self.n_mean - 1.0)

        # P_shifted = exp(Lambda_bar * dz)
        # Shape: (Nx, 1)
        self._P_shifted = np.exp((lambda_vac + phi_mean) * self.dz)[:, np.newaxis]

        # Longitudinal Shifts (Twist)
        L = self.nz_steps
        kz = np.arange(L)

        # lam_alpha = alpha^(1/L) * exp(-i 2pi k/L)
        # Shape: (1, Nz)
        lam_alpha = (self.alpha ** (1 / L) * np.exp(-2j * np.pi * kz / L))[
            np.newaxis, :
        ]

        # Kernel: P / (1 - lambda * P)
        denom = 1.0 - (lam_alpha * self._P_shifted)

        # Expand dims to (1, Nx, Nz) for batch broadcasting
        return self._P_shifted / (denom + 1e-15)

    def _setup_batch_operators(self, scan_indices: np.ndarray):
        """
        Prepares the local object slices and twist operators for the batch.
        scan_indices: (Batch,) integers indicating start x index.
        """
        L = self.nz_steps
        Nx = self.nx

        # 1. Gather Local Object
        # Create grid of indices: (Batch, Nx)
        batch_offsets = scan_indices[:, np.newaxis]
        window_indices = np.arange(Nx)[np.newaxis, :]
        gather_indices = batch_offsets + window_indices

        # Fetch local delta_n: (Batch, Nx, Nz)
        # Assumes delta_n_global is (Global_Nx, Nz)
        local_delta_n = self.delta_n_global[gather_indices, :]

        # 2. Physics Terms (Phase Shift)
        # exp(i * k * delta_n * dz / 2)
        N_phase = (1j * self.k0 * local_delta_n * self.dz / 2).astype(np.complex128)

        # 3. Twist Terms (Gamma)
        z_idx = np.arange(L)
        # Shapes: (1, 1, Nz)
        gamma = (self.alpha ** (-z_idx / L))[np.newaxis, np.newaxis, :]
        gamma_inv = (self.alpha ** (z_idx / L))[np.newaxis, np.newaxis, :]

        # 4. Operator Construction
        self._half_obj = np.exp(N_phase)
        self._inv_half_obj = np.exp(-N_phase)

        # Pre/Post Multipliers for M^-1
        self._pre_mul = self._half_obj * gamma_inv
        self._post_mul = self._half_obj * gamma

        # Error Terms (Using exact form: InvObj * shift(InvObj) - I)
        # E1: Sub-diagonal
        shifted_inv = np.roll(self._inv_half_obj, shift=1, axis=2)
        self._E1_coeff = (self._inv_half_obj * shifted_inv) - 1.0

        # E2: Corner (Wrap-around)
        # Note: self._inv_half_obj[:, :, 0] is slice z=0
        self._E2_coeff = (
            self.alpha * self._inv_half_obj[:, :, 0] * self._inv_half_obj[:, :, -1]
        )

    def _apply_M_inv(self, rhs_batch: np.ndarray) -> np.ndarray:
        """Forward Preconditioner: (Untwist+Refract) -> IFFT -> K -> FFT -> (Refract+Twist)"""

        # 1. Entry Multiplier
        v = rhs_batch * self._pre_mul

        # 2. FFT along Z (axis 2) to get spectral z
        v_k = np.fft.fft2(v, axes=(1, 2))

        # 3. Kernel Multiply
        v_k *= self.K_3d

        # 4. Inverse FFT back to real space
        v = np.fft.ifft2(v_k, axes=(1, 2))

        # 5. Exit Multiplier
        return v * self._post_mul

    def _apply_M_inv_adjoint(self, rhs_batch: np.ndarray) -> np.ndarray:
        """
        Adjoint of M^-1.
        Order is reversed: conj(Post) -> FFT -> conj(K) -> IFFT -> conj(Pre)
        """
        # 1. Apply conjugate of Post-multiplier (Exit becomes Entry)
        v = rhs_batch * np.conj(self._post_mul)

        # B. FFT (Inverse of IFFT step in forward)
        v_k = np.fft.fft2(v, axes=(1, 2))

        # C. Conjugate Kernel
        v_k *= np.conj(self.K_3d)

        # D. IFFT (Inverse of FFT step in forward)
        v = np.fft.ifft2(v_k, axes=(1, 2))

        # E. Conjugate Pre-Multiply
        return v * np.conj(self._pre_mul)

    def _apply_A(self, u_batch: np.ndarray) -> np.ndarray:
        """Applies A = I + M^-1 * E"""
        # E1 acts on u_{j-1} (shift z by +1)
        u_prev = np.roll(u_batch, 1, axis=2)
        err = self._E1_coeff * u_prev

        # Zero out the wrap-around from roll (since E1 is strictly sub-diagonal)
        err[:, :, 0] = 0.0

        # E2 acts on u_{L-1} and adds to slice 0
        err[:, :, 0] += self._E2_coeff * u_batch[:, :, -1]

        return u_batch + self._apply_M_inv(err)

    def _apply_A_adjoint(self, v_batch: np.ndarray) -> np.ndarray:
        """
        Computes A^H v = v + M^-H E^H v
         where E^H has E1^H (super-diagonal) and E2^H (corner).
        """
        # 1. E1^H: Acts on v_{j+1}. So we roll -1 (shift left)
        v_next = np.roll(v_batch, -1, axis=2)
        E1_conj_shifted = np.roll(np.conj(self._E1_coeff), -1, axis=2)
        err = E1_conj_shifted * v_next

        # The last slice (L-1) shouldn't get input from slice 0 via E1 (boundary)
        err[:, :, -1] = 0.0

        # 2. E2^H: Acts on v_0 and adds to slice L-1
        err[:, :, -1] += np.conj(self._E2_coeff) * v_batch[:, :, 0]

        return v_batch + self._apply_M_inv_adjoint(err)

    def compute_gradient_object(
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

    def compute_gradient_probe(self, v_sol: np.ndarray) -> np.ndarray:
        """Computes the probe gradient by integrating the adjoint field at the source plane."""
        # Extract the adjoint field at the source plane (first slice)
        return np.mean(v_sol[:, :, 0], axis=0)

    def run_ptycho_batch(
        self,
        psi_init_batch: np.ndarray,
        scan_indices: np.ndarray,
        n_map: np.ndarray,
        mode: str = "forward",
    ) -> np.ndarray:
        """
        Main runner for Forward or Adjoint pass.
        psi_init_batch: (Batch, Nx)
        """
        self.setup_solver(n_map)
        B = scan_indices.shape[0]
        self._setup_batch_operators(scan_indices)

        # Initialize Volume S
        S = np.zeros((B, self.nx, self.nz_steps), dtype=np.complex128)

        if mode == "adjoint":
            # In adjoint mode, the "initial" wave is the residual at the DETECTOR (last slice)
            S[:, :, -1] = psi_init_batch
        else:
            # In forward mode, the initial wave is the probe at the SOURCE (first slice)
            S[:, :, 0] = psi_init_batch

        # Apply the Preconditioner (M_inv) to get the initial guess 'b'
        b = self._apply_M_inv_adjoint(S) if mode == "adjoint" else self._apply_M_inv(S)
        u_sol = b.copy()

        # Richardson Iteration / Iterative Refinement
        if self.n_iter > 0:
            if self.solver_type == "richardson":
                for _ in range(self.n_iter):
                    res = b - (
                        self._apply_A_adjoint(u_sol)
                        if mode == "adjoint"
                        else self._apply_A(u_sol)
                    )
                    u_sol += res
            elif self.solver_type in ["gmres"]:

                def matvec(u_flat):
                    u_reshaped = u_flat.reshape(self.nx, self.nz_steps)
                    return (
                        self._apply_A_adjoint(u_reshaped)
                        if mode == "adjoint"
                        else self._apply_A(u_reshaped)
                    ).flatten()

                A_op = LinearOperator(
                    (self.nx * self.nz_steps, self.nx * self.nz_steps),
                    matvec=matvec,
                    dtype=np.complex128,
                )

                u_flat, _ = gmres(
                    A_op, b.flatten(), x0=b.flatten(), maxiter=self.n_iter
                )
                u_sol = u_flat.reshape(self.nx, self.nz_steps)

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
