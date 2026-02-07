import logging
from typing import Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres

from .base import OpticalWaveSolver
from .utils import get_spectral_coords


class ParallelMultisliceSolver2(OpticalWaveSolver):
    """
    Symmetric Parallel-in-Time Multislice Solver.

    Corrections applied:
    1. Gamma is defined as alpha^(-z/L) (Growth) to cancel eigenvalue damping.
    2. E1 Error sign flipped to (O^-1 O^-1 - I).
    3. Kernel uses P / (1 - lambda P) form for stability.
    """

    def __init__(
        self,
        n_map: np.ndarray,
        dx: float,
        wavelength: float,
        dz: float,
        probe_dia: float = 0,
        probe_focus: float = 0,
        store_beam: bool = False,
        alpha: float = 1e-6,
        n_iter: int = 5,
        solver_type: str = "richardson",
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.alpha = float(alpha)
        self.n_iter = n_iter
        self.solver_type = solver_type.lower()

        # Physics setup
        self.n_mean = np.mean(self.n_map)
        self.delta_n = self.n_map - self.n_mean

        self._setup_operators()

    def _setup_operators(self):
        L = self.nz_steps
        z_idx = np.arange(L)

        # A. Physics Terms (Refraction)
        # Use complex128 to prevent underflow/overflow with alpha terms
        N_phase = (1j * self.k0 * self.delta_n * self.dz / 2).astype(np.complex128)
        self._half_obj = np.exp(N_phase)
        self._inv_half_obj = np.exp(-N_phase)

        # 2. Parallel Twist (Gamma)

        # Exit Twist (Gamma): Growth term alpha^(-z/L)
        self._gamma = (self.alpha ** (-z_idx / L))[np.newaxis, :]

        # Entry Twist (Gamma^-1): Decay term alpha^(z/L)
        self._gamma_inv = (self.alpha ** (z_idx / L))[np.newaxis, :]

        # 3. Pre/Post Multipliers
        # M^-1 = (O_1/2 * G) * [Spectral] * (G^-1 * O_1/2)
        # Pre-multiply applies G^-1 (Entry Twist)
        self._pre_mul = self._half_obj * self._gamma_inv

        # Post-multiply applies G (Exit Twist)
        self._post_mul = self._half_obj * self._gamma

        # 4. Error Term E1 (Sub-diagonal)
        # E1 = Exact - Approx = (-I) - (-InvObj * InvObj) = InvObj*InvObj - I
        shifted_inv = np.roll(self._inv_half_obj, shift=1, axis=1)
        self._E1_coeff = (self._inv_half_obj * shifted_inv) - 1.0

        # 5. Error Term E2 (Corner)
        # E2 = Corner term alpha * InvObj[0] * InvObj[L]
        self._E2_coeff = (
            self.alpha * self._inv_half_obj[:, 0] * self._inv_half_obj[:, -1]
        )

        # 6. Compute Spectral Kernel
        self._K_3d = self._get_3d_kernel()

    def _get_3d_kernel(self) -> np.ndarray:
        """
        Computes K = P / (1 - lambda * P)
        This matches the forward recurrence u_n = P * u_{n-1}
        """
        kx = get_spectral_coords(self.nx, self.dx, "FFT")

        # Transverse Propagator P
        inside = self.k0sq - kx**2
        sqrt_term = np.sqrt(np.clip(inside, 0.0, None))

        # lambda_vac: vacuum phase advance per dz
        lambda_vac = 1j * (sqrt_term - self.k0)

        # phi_mean: mean potential phase advance per dz
        phi_mean = 1j * self.k0 * (self.n_mean - 1.0)

        # P_shifted corresponds to exp(Lambda_bar) in your derivation
        self._P_shifted = np.exp((lambda_vac + phi_mean) * self.dz)[:, np.newaxis]

        # Longitudinal Eigenvalues (decaying)
        L = self.nz_steps
        kz = np.arange(L)

        # lam_alpha contains the shift operator eigenvalues
        # exp(-2j * pi * kz / L) is standard FFT definition
        lam_alpha = (self.alpha ** (1 / L) * np.exp(-2j * np.pi * kz / L))[
            np.newaxis, :
        ]

        # Kernel calculation
        denom = 1.0 - (lam_alpha * self._P_shifted)

        # Numerical safeguard: avoid division by zero
        return self._P_shifted / (denom + 1e-15)

    def _apply_M_inv(self, rhs_vector: np.ndarray) -> np.ndarray:
        """
        Applies M^-1 using standard 2D FFT.
        Sequence: (Refract+Twist) -> FFT2 -> Kernel -> IFFT2 -> (Untwist+Refract)
        """
        # 1. Combined Real-Space Multiply (Entry)
        v = rhs_vector * self._pre_mul

        # 2. Standard 2D FFT (Space X, Time Z)
        # Note: Ensure axes align with your kernel broadcasting (Nx, L)
        v_k = np.fft.fft2(v, axes=(0, 1))

        # 3. Kernel Multiply (Spectral)
        v_k *= self._K_3d

        # 4. Standard 2D IFFT
        v = np.fft.ifft2(v_k, axes=(0, 1))

        # 5. Combined Real-Space Multiply (Exit)
        return v * self._post_mul

    def _apply_A(self, u: np.ndarray) -> np.ndarray:
        """Applies A = I + M^-1 * E"""
        # E1 acts on u_{j-1}
        u_prev = np.roll(u, 1, axis=1)
        err = self._E1_coeff * u_prev
        err[:, 0] = 0.0  # E1 is strictly sub-diagonal

        # E2 acts on u_{L-1} and adds to slice 0
        err[:, 0] += self._E2_coeff * u[:, -1]

        # Apply M^-1 to error
        correction = self._apply_M_inv(err)
        return u + correction

    def run(self, psi_init: Optional[np.ndarray] = None) -> "ParallelMultisliceSolver2":
        psi_0 = self.initialize_wavefront(psi_init)

        # Source S at slice 0
        S = np.zeros((self.nx, self.nz_steps), dtype=np.complex128)

        # Exact first step for Source: psi_0
        # Note: We apply the P operator in spectral space for accuracy
        S[:, 0] = psi_0

        # Initial Guess
        b = self._apply_M_inv(S)
        u_sol = b.copy()

        # Iterative Solver
        if self.n_iter > 0:
            if self.solver_type == "richardson":
                logging.info(f"Running Richardson (n={self.n_iter})...")
                for _ in range(self.n_iter):
                    res = b - self._apply_A(u_sol)
                    u_sol += res

            elif self.solver_type in ["gmres", "bicgstab"]:
                logging.info(f"Running {self.solver_type.upper()}...")

                def matvec(u_flat):
                    return self._apply_A(
                        u_flat.reshape(self.nx, self.nz_steps)
                    ).flatten()

                A_op = LinearOperator(
                    (self.nx * self.nz_steps, self.nx * self.nz_steps),
                    matvec=matvec,
                    dtype=np.complex128,
                )

                solver = gmres if self.solver_type == "gmres" else bicgstab
                u_flat, _ = solver(
                    A_op, b.flatten(), x0=b.flatten(), maxiter=self.n_iter
                )
                u_sol = u_flat.reshape(self.nx, self.nz_steps)

        self.beam_history = u_sol if self.store_beam else None
        self.psi_final = u_sol[:, -1]
        return self
