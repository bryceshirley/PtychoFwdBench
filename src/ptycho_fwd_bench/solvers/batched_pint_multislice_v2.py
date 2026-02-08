import logging

import numpy as np

# Assuming these exist in your utils/generators
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
        nx: int,  # Window Size (local x)
        nz_steps: int,  # Depth Steps (local z)
        probe_dia: float = 50e-9,
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

        # Initialize global map placeholders
        self.n_mean = 1.0
        self.delta_n_global = None

        logging.info("Initializing spectral physics engine...")

    def setup_solver(self, n_map: np.ndarray):
        """
        Sets up the global refractive index map.
        n_map shape should be (Global_Nx, Nz) for 2D ptycho.
        """
        n_map_squeezed = np.squeeze(n_map)

        if n_map_squeezed.ndim != 2:
            raise ValueError(
                f"n_map must be 2D (Global_Nx, Nz). Got shape {n_map.shape}"
            )

        self.n_mean = np.mean(n_map_squeezed)
        self.delta_n_global = n_map_squeezed - self.n_mean

        # Precompute the spectral kernel (Depends only on grid, not object)
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

        # Phase advance per dz
        lambda_vac = 1j * (sqrt_term - self.k0)
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
        K = self._P_shifted / (denom + 1e-15)
        return K[np.newaxis, :, :]

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
        # Note: Your original code had FFT/IFFT swapped relative to standard definitions,
        # or relative to my previous explanation.
        # Standard spectral solvers usually do: IFFT_z( K * FFT_z( u ) )
        # Here we follow the logic: Input is x-z space. Transform to x-k_z.

        # 1. Entry Multiplier
        v = rhs_batch * self._pre_mul

        # 2. FFT along Z (axis 2) to get spectral z
        #    We usually keep X in real space (or spectral x), but K_3d handles x spectrally?
        #    Wait, K_3d was computed with kx. So we need FFT along X (axis 1) too.
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

        # 2. FFT (Forward FFT is unitary-ish, adjoint is IFFT-ish, but if we use
        #    fft2/ifft2 pairs, we just swap them or use conj).
        #    Mathematical Adjoint of (F K F^-1) is (F^-H K^H F^H).
        #    Since F is unitary (up to scale), F^H = F^-1.
        #    So Adjoint is: F^-1 [ conj(K) * F [ v ] ]
        #    Wait! The forward was: v_out = Post * F^-1 * (K * F * (Pre * v_in))
        #    Adjoint: v_adj = Pre^H * F^H * (K^H * F^-H * (Post^H * v_in))
        #    F^H is IFFT (scaled). F^-H is FFT.

        # Correct Adjoint Sequence:

        # A. Conjugate Post-Multiply
        # v is now (Post^H * v_in)

        # B. FFT (Inverse of IFFT step in forward)
        v_k = np.fft.fft2(v, axes=(1, 2))

        # C. Conjugate Kernel
        v_k *= np.conj(self.K_3d)

        # D. IFFT (Inverse of FFT step in forward)
        v = np.fft.ifft2(v_k, axes=(1, 2))

        # E. Conjugate Pre-Multiply
        return v * np.conj(self._pre_mul)

    def _apply_Error(self, u_batch: np.ndarray) -> np.ndarray:
        """Computes E * u"""
        # E1 acts on u_{j-1} (shift z by +1)
        u_prev = np.roll(u_batch, 1, axis=2)
        err = self._E1_coeff * u_prev

        # Zero out the wrap-around from roll (slice 0 shouldn't get slice L-1 via E1)
        # But E1 is strictly sub-diagonal, so row 0 is 0.
        err[:, :, 0] = 0.0

        # E2 acts on u_{L-1} and adds to slice 0
        err[:, :, 0] += self._E2_coeff * u_batch[:, :, -1]

        return err

    def _apply_Error_adjoint(self, v_batch: np.ndarray) -> np.ndarray:
        """
        Computes E^H * v.
        E1 is Lower Diagonal -> E1^H is Upper Diagonal.
        E2 is Bottom-Left Corner -> E2^H is Top-Right Corner.
        """
        # 1. E1^H: Acts on v_{j+1}. So we roll -1 (shift left)
        #    err_j = conj(E1_{j+1}) * v_{j+1}

        # Use roll -1 to bring v_{j+1} to pos j
        v_next = np.roll(v_batch, -1, axis=2)

        # We need the coeff at j+1 aligned with v_{j+1}.
        # E1_coeff stored at 'j' corresponds to interaction (j, j-1).
        # We need interaction (j+1, j).
        # So we align conj(E1) shifted by -1.
        E1_conj_shifted = np.roll(np.conj(self._E1_coeff), -1, axis=2)

        err = E1_conj_shifted * v_next

        # The last slice (L-1) shouldn't get input from slice 0 via E1 (boundary)
        err[:, :, -1] = 0.0

        # 2. E2^H: Acts on v_0 and adds to slice L-1
        #    (Corner element (0, L-1) transposed is (L-1, 0))
        err[:, :, -1] += np.conj(self._E2_coeff) * v_batch[:, :, 0]

        return err

    def compute_gradient_object(
        self, u_sol: np.ndarray, v_sol: np.ndarray, scan_indices: np.ndarray
    ):
        """Assembles complex gradient components via scatter-add."""
        # u * v_conj
        local_overlap = u_sol * np.conj(v_sol)

        # Flatten logic for safe accumulation
        B, Nx, Nz = local_overlap.shape

        # Create flat indices for the global map rows
        # batch_offsets: (B, 1) -> (B, Nx)
        global_x_indices = scan_indices[:, np.newaxis] + np.arange(Nx)[np.newaxis, :]
        flat_indices = global_x_indices.ravel()  # Size B*Nx

        # Reshape overlap to (B*Nx, Nz)
        flat_overlap = local_overlap.reshape(B * Nx, Nz)

        # Initialize container
        global_grad = np.zeros_like(self.delta_n_global, dtype=np.complex128)

        # Add safely
        np.add.at(global_grad, flat_indices, flat_overlap)

        return global_grad

    def compute_gradient_probe(self, v_sol: np.ndarray):
        """
        Computes gradient for the probe at z=0.
        Gradient = Sum_over_batch( v_adjoint(z=0) )
        """
        # v_sol: (Batch, Nx, Nz)
        # Take z=0 slice -> (Batch, Nx)
        # Sum over Batch -> (Nx,)
        probe_grad = np.sum(v_sol[:, :, 0], axis=0)
        return probe_grad

    def run_ptycho_batch(
        self,
        psi_init_batch: np.ndarray,
        scan_indices: np.ndarray,
        n_map: np.ndarray,
        n_iter: int = 5,
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
            # Adjoint Source: Residual at detector (z=End)
            S[:, :, -1] = psi_init_batch
            # Invert: Adjoint Solve
            b = self._apply_M_inv_adjoint(S)
            u_sol = b.copy()

            for _ in range(n_iter):
                # Apply (I + M^-H E^H)
                # Res = b - (I + M^-H E^H) u
                correction_term = self._apply_M_inv_adjoint(
                    self._apply_Error_adjoint(u_sol)
                )
                A_u = u_sol + correction_term
                u_sol += b - A_u

        else:
            # Forward Source: Probe at entrance (z=0)
            S[:, :, 0] = psi_init_batch
            # Invert: Forward Solve
            b = self._apply_M_inv(S)
            u_sol = b.copy()

            for _ in range(n_iter):
                # Apply (I + M^-1 E)
                correction_term = self._apply_M_inv(self._apply_Error(u_sol))
                A_u = u_sol + correction_term
                u_sol += b - A_u

        return u_sol
